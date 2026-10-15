# Scratch: traced FastDFlashRunner round time at position ISL (decode-only: no prefill, caches past the round's
# positions hold no real data; attention reads ISL positions). tok/s/user at acceptance ACC = ACC / round.
import os
import statistics
import time

import pytest

import ttnn
from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric

ACC = float(os.environ.get("DFLASH_ACC", "3.46"))


@pytest.mark.timeout(7200)
@parametrize_mesh_with_fabric([(1, 4)])
def test_dflash_fast_isl(mesh_device, device_params):
    import models.demos.gpt_oss.demo.text_demo as text_demo
    from models.demos.gpt_oss.config import MeshConfig, ModeConfig
    from models.demos.gpt_oss.tt.dflash.draft import DFlashDraft
    from models.demos.gpt_oss.tt.dflash.runner import FastDFlashRunner
    from models.tt_transformers.tt.generator import Generator

    isl = int(os.environ["ISL_TOKENS"])
    ctx = max(4096, 1 << (isl + 256).bit_length())
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
    mesh = mesh_device

    def one_round(start, traced):
        fast.draft.set_draft_inputs(1234, start)
        fast.verify.set_positions(start)
        if traced:
            ttnn.execute_trace(mesh, fast.draft.trace_draft, cq_id=0, blocking=False)
            ttnn.execute_trace(mesh, fast._verify_trace_for(start), cq_id=0, blocking=False)
        else:
            fast.draft._draft_device()
            fast.verify.forward()
        fast.verify.read_result()
        fast.draft.set_context_positions(range(start, start + 4))
        if traced:
            ttnn.execute_trace(mesh, fast.draft.trace_ctx, cq_id=0, blocking=False)
        else:
            fast.draft._context_device()

    one_round(isl, False)
    fast.capture_traces()
    ts = []
    for k in range(24):
        t = time.perf_counter()
        one_round(isl + 4 * k, True)
        ttnn.synchronize_device(mesh)
        ts.append(time.perf_counter() - t)
    rnd = statistics.median(ts[4:])
    print(
        f"DFISL isl={isl} ctx={ctx}: round {rnd * 1e3:.2f} ms -> {ACC / rnd:.1f} tok/s/user at accept {ACC} "
        f"({rnd * 1e3 / ACC:.3f} ms/token)",
        flush=True,
    )
