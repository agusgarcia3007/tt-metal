# Scratch: teacher-forced acceptance of the torch reference drafter (tt/dflash/reference.py, fp32 CPU) on the same
# fixed reference texts as _scratch_dflash_tf_accept.py, with the target's context features from one device prefill
# of prompt + reference text (exact target hidden states): the acceptance the drafter itself allows.
import os
from pathlib import Path

import pytest
import torch

from models.demos.gpt_oss.tests._scratch_dflash_draft import _lm_head_cpu
from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric

NEW = int(os.environ.get("TF_NEW", "96"))


@pytest.mark.timeout(7200)
@parametrize_mesh_with_fabric([(1, 4)])
def test_tf_cpu(mesh_device, device_params):
    import models.demos.gpt_oss.demo.text_demo as text_demo
    from models.demos.gpt_oss.config import MeshConfig, ModeConfig
    from models.demos.gpt_oss.tt.dflash.capture import TargetCapture
    from models.demos.gpt_oss.tt.dflash.reference import DraftReference
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
    ref_d = DraftReference()
    cap = TargetCapture(m, ref_d.target_layer_ids)
    head = _lm_head_cpu()
    lm, emb = head["lm_head.weight"], head["model.embed_tokens.weight"]
    refs = torch.load(Path(__file__).parents[4] / "generated" / "dflash_reference.pt")
    allacc = []
    allacc_v = {}
    for w, r in enumerate(refs):
        ids = r["prompt_ids"]
        L = len(ids)
        ref = torch.load(Path(__file__).parents[4] / "generated" / f"dflash_tf_ref_{w}.pt")
        full = list(ids) + ref[: NEW + 8]
        generator.prefill_forward_text(
            torch.tensor([full]),
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=[len(full)],
            enable_trace=False,
            warmup_prefill=False,
        )
        feats = cap.prefill_features(len(full))  # [n, 5 * hidden]
        vocabs = [int(v) for v in os.environ.get("TF_VOCAB", "0").split(",")]  # 0: full; N: ids < N only
        for V in vocabs:
            pos, acc = 0, []
            while pos < NEW:
                start = L + pos
                block = [ref[pos]] + [ref_d.mask_token_id] * 7
                hid = ref_d.forward(feats[:start], emb[block], range(start), range(start, start + 8))
                if V:  # ids < V plus the special tokens (>= 199998)
                    sub = torch.cat([torch.arange(V), torch.arange(199998, lm.shape[0])])
                    d = sub[(hid[1:] @ lm[sub].T).argmax(-1)].tolist()
                else:
                    d = (hid[1:] @ lm.T).argmax(-1).tolist()
                k = 0
                while k < 7 and d[k] == ref[pos + 1 + k]:
                    k += 1
                acc.append(k + 1)
                pos += k + 1
            allacc_v.setdefault(V, []).extend(acc)
            print(f"TFCPU prompt {w} vocab {V}: accept {sum(acc) / len(acc):.3f} over {len(acc)} rounds", flush=True)
        allacc = allacc_v[vocabs[0]]
    for V, a in allacc_v.items():
        print(f"TFCPU vocab {V}: mean accept {sum(a) / len(a):.3f} over {len(a)} rounds", flush=True)
    hist = [allacc.count(k) for k in range(1, 9)]
    print(f"TFCPU mean accept {sum(allacc) / len(allacc):.3f} over {len(allacc)} rounds; histogram {hist}", flush=True)
