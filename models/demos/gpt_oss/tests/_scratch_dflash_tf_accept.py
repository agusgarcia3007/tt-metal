# Scratch: teacher-forced DFlash acceptance: rounds follow a fixed reference continuation (plain greedy, computed once),
# so the drafter variants (ttnn / fused) are compared on the same text: accepted = matching drafts + 1 per round.
import os
from pathlib import Path

import pytest
import torch

from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric

NEW = int(os.environ.get("TF_NEW", "96"))


@pytest.mark.timeout(7200)
@parametrize_mesh_with_fabric([(1, 4)])
def test_tf_accept(mesh_device, device_params):
    import models.demos.gpt_oss.demo.text_demo as text_demo
    from models.demos.gpt_oss.config import MeshConfig, ModeConfig
    from models.demos.gpt_oss.tt.dflash.draft import DFlashDraft
    from models.demos.gpt_oss.tt.dflash.runner import DFlashRunner, FastDFlashRunner
    from models.tt_transformers.tt.generator import Generator

    ctx = 4096
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=4, ep=1))
    model_args, model, page_table, kv_cache, tokenizer, processor, _ = text_demo.prepare_gpt_oss_generator_args(
        num_devices=4,
        data_parallel=1,
        mesh_device=mesh_device,
        global_batch_size=1,
        optimizations=None,
        max_seq_len=ctx,
        page_params={"page_block_size": 64, "page_max_num_blocks_per_dp": ctx // 64},
        paged_attention=True,
        mesh_config=mesh_config,
        state_dict=None,
    )
    generator = Generator(model, model_args, mesh_device, processor=processor, tokenizer=tokenizer)
    m = model[0]
    draft = DFlashDraft(mesh_device, m, max_positions=ctx)
    fast = FastDFlashRunner(generator, m, page_table, kv_cache, draft)
    fused = draft.fused
    plain = DFlashRunner.__new__(DFlashRunner)
    plain.generator, plain.page_table, plain.kv_cache, plain.V = generator, page_table, kv_cache, m.vocab_size
    refs = torch.load(Path(__file__).parents[4] / "generated" / "dflash_reference.pt")
    res = {}
    unions = []
    for w, r in enumerate(refs):
        ids = r["prompt_ids"]
        L = len(ids)
        ref_file = Path(__file__).parents[4] / "generated" / f"dflash_tf_ref_{w}.pt"
        if ref_file.exists() and len(torch.load(ref_file)) >= NEW + 9:
            ref = torch.load(ref_file)  # fixed reference text (plain greedy flips on ties from run to run)
        else:
            first = fast.prefill(ids)
            ref = [first]
            for k in range(NEW + 8):
                ref.append(plain._decode(ref[-1], L + k))
            torch.save(ref, ref_file)
        for name, f in (("fused", fused),) if os.environ.get("TF_FUSED_ONLY") else (("ttnn", None), ("fused", fused)):
            draft.fused = f
            fast.prefill(ids)
            pos, acc = 0, []  # ref[pos] is the block's first token, at position L + pos
            while pos < NEW:
                start = L + pos
                draft.set_draft_inputs(ref[pos], start)
                fast.verify.set_positions(start)
                draft._draft_device()
                if os.environ.get("TF_UNIONS"):
                    fast.verify.union_sets = []
                fast.verify.forward()
                if os.environ.get("TF_UNIONS"):
                    unions.append(fast.verify.union_sets)
                    fast.verify.union_sets = None
                block, post = fast.verify.read_result()
                d = block[1:]
                k = 0
                while k < 7 and d[k] == ref[pos + 1 + k]:
                    k += 1
                acc.append(k + 1)
                draft.set_context_positions(range(start, start + k + 1))
                draft._context_device()
                pos += k + 1
            res.setdefault(name, []).extend(acc)
            print(f"TF prompt {w} {name}: accept {sum(acc) / len(acc):.3f} over {len(acc)} rounds", flush=True)
    if unions:
        # per layer: |U| and, vs the previous round's union of that layer, the hit rate of its first p experts
        L = len(unions[0])
        for p in (1, 2, 3, 4):
            hits = tot = 0
            for a, b in zip(unions, unions[1:]):
                for l in range(L):
                    prev = a[l][:p]
                    hits += sum(e in b[l] for e in prev)
                    tot += len(prev)
            print(f"TF unions: previous round's first {p} experts in this round's union: {hits / tot:.3f}", flush=True)
        ov = [len(set(a[l]) & set(b[l])) / len(b[l]) for a, b in zip(unions, unions[1:]) for l in range(L)]
        print(
            f"TF unions: mean |U| {sum(len(u[l]) for u in unions for l in range(L)) / (len(unions) * L):.2f}; "
            f"fraction of this round's union also in the previous round's {sum(ov) / len(ov):.3f}",
            flush=True,
        )
    for name, acc in res.items():
        print(f"TF {name}: mean accept {sum(acc) / len(acc):.3f} over {len(acc)} rounds", flush=True)
        hist = [acc.count(k) for k in range(1, 9)]
        print(
            f"TF {name}: accept histogram 1..8 {hist}; E[min(a, R)] for R = 2..8: "
            f"{[round(sum(min(a, R) for a in acc) / len(acc), 3) for R in range(2, 9)]}",
            flush=True,
        )
