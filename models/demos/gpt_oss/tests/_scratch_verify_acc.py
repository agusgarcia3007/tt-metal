# Scratch: teacher-forced accuracy of the 8-row FusedVerify vs plain greedy b1 decode: for each prompt, plain decode
# N tokens (full logits kept), then verify the same tokens in blocks of 8 at the same positions; per-row argmax
# agreement, and for every disagreement the plain decode's top-1 / top-2 logit gap.
import os
from pathlib import Path

import pytest
import torch

from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric

N = int(os.environ.get("ACC_N", "64"))


@pytest.mark.timeout(7200)
@parametrize_mesh_with_fabric([(1, 4)])
def test_verify_acc(mesh_device, device_params):
    import models.demos.gpt_oss.demo.text_demo as text_demo
    from models.demos.gpt_oss.config import MeshConfig, ModeConfig
    from models.demos.gpt_oss.tt.dflash.verify import FusedVerify
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
    V = m.vocab_size
    kv = kv_cache[0] if isinstance(kv_cache[0][0], list) else kv_cache
    ver = FusedVerify(m, page_table, kv)
    refs = torch.load(Path(__file__).parents[4] / "generated" / "dflash_reference.pt")
    if os.environ.get("ACC_BENCH"):  # the prompts of _scratch_dflash_bench.py
        from models.demos.gpt_oss.tests._scratch_dflash_bench import PROMPTS

        def enc(text):
            e = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True
            )
            if isinstance(e, dict) or hasattr(e, "input_ids"):
                e = e["input_ids"]
            return [int(t) for t in e]

        refs = [dict(prompt_ids=enc(t)) for _, t in PROMPTS]
    tot = agree = 0
    unions = []
    gaps = []
    for w, r in enumerate(refs):
        ids = r["prompt_ids"]
        L = len(ids)
        lg = generator.prefill_forward_text(
            torch.tensor([ids]),
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=[L],
            enable_trace=False,
            warmup_prefill=False,
        )
        g = [int(lg.reshape(-1)[:V].float().argmax())]
        logits = []
        for k in range(N):
            out = generator.decode_forward(
                torch.tensor([g[-1]]),
                torch.tensor([L + k]),
                enable_trace=False,
                page_table=page_table,
                kv_cache=kv_cache,
                sampling_params=None,
                reload_inputs=True,
                reload_page_table=False,
                reload_sampling_params=False,
                reset_sampling_state=False,
            )
            lo = (out[0] if isinstance(out, tuple) else out).reshape(-1)[:V].float()
            logits.append(lo)
            g.append(int(lo.argmax()))
        for k0 in range(0, N - 7, 8):
            ver.set_inputs(g[k0 : k0 + 8], L + k0)
            ver.union_log = []
            ver.forward()
            unions.append(sum(ver.union_log) / max(1, len(ver.union_log)))
            ver.union_log = None
            got = ver.argmax()
            for j in range(8):
                tot += 1
                exp = g[k0 + j + 1]
                if got[j] == exp:
                    agree += 1
                else:
                    top = logits[k0 + j].topk(2)
                    gap = float(top.values[0] - top.values[1])
                    vlog = float(logits[k0 + j][got[j]])
                    gaps.append(gap)
                    print(
                        f"ACC prompt {w} pos {L + k0 + j}: verify {got[j]} vs plain {exp}; plain top-2 gap "
                        f"{gap:.4f}, plain logit of verify's pick {vlog:.3f} (top {float(top.values[0]):.3f})",
                        flush=True,
                    )
    print(
        f"ACC mean union experts per layer {sum(unions) / len(unions):.2f} (per block: {[round(u, 1) for u in unions]})",
        flush=True,
    )
    print(
        f"ACC verify top-1 == plain greedy {agree}/{tot} ({100 * agree / tot:.2f}%); disagreement gaps {gaps}",
        flush=True,
    )
