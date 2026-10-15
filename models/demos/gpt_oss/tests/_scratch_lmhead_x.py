# Scratch: capture the verify pass's final normed hidden rows (the LM head input) over eager DFlash rounds on the
# bench prompts -> generated/dflash_lmhead_x.pt ([N, hidden] FP32), for an offline BFP4-vs-BF8 LM head ranking study.
import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric

NEW = int(os.environ.get("LMX_NEW", "48"))


@pytest.mark.timeout(7200)
@parametrize_mesh_with_fabric([(1, 4)])
def test_lmhead_x(mesh_device, device_params):
    import models.demos.gpt_oss.demo.text_demo as text_demo
    from models.demos.gpt_oss.config import MeshConfig, ModeConfig
    from models.demos.gpt_oss.tests._scratch_dflash_bench import PROMPTS
    from models.demos.gpt_oss.tt.dflash.draft import DFlashDraft
    from models.demos.gpt_oss.tt.dflash.runner import FastDFlashRunner
    from models.demos.gpt_oss.tt.fused_decode.boundary import rowtile_to_rows
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
    v = fast.verify
    rows = []
    lm = v.lm_head

    def wrapped(p, w, lists):
        lm(p, w, lists)
        x = p if isinstance(p, ttnn.Tensor) else p.x
        t = ttnn.to_torch(ttnn.get_device_tensors(x)[0]).float()[0, 0]
        rows.append(rowtile_to_rows(t, v.M, v.H))

    v.lm_head = wrapped

    def encode(text):
        e = tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True)
        e = e["input_ids"] if isinstance(e, dict) or hasattr(e, "input_ids") else e
        return [int(t) for t in e]

    for _, text in PROMPTS:
        fast.generate(encode(text), NEW)
    X = torch.cat(rows)
    out = Path(__file__).parents[4] / "generated" / "dflash_lmhead_x.pt"
    torch.save(X, out)
    print(f"LMX {X.shape[0]} rows -> {out}", flush=True)
