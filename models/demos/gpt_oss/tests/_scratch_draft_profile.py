# Scratch: device profile of the traced drafter's draft pass and context pass (2 eager passes each between signposts).
import os

import pytest

import ttnn
from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric


@pytest.mark.timeout(3600)
@parametrize_mesh_with_fabric([(1, 4)])
def test_draft_profile(mesh_device, device_params):
    from tracy import signpost

    import models.demos.gpt_oss.demo.text_demo as text_demo
    from models.demos.gpt_oss.config import MeshConfig, ModeConfig
    from models.demos.gpt_oss.tt.dflash.draft import DFlashDraft
    from models.demos.gpt_oss.tt.dflash.verify import FusedVerify

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
    m = model[0]
    kv = kv_cache[0] if isinstance(kv_cache[0][0], list) else kv_cache
    ver = FusedVerify(m, page_table, kv)
    draft = DFlashDraft(mesh_device, m, max_positions=ctx)
    draft.enable_trace()
    ver.enable_device_argmax()
    draft.feed_verify(ver.tokens)
    from models.demos.gpt_oss.tt.dflash.fused_draft import FusedDraft

    draft.window = int(os.environ.get("DP_WINDOW", "512")) or None  # the DFlash rounds' drafter window
    draft.fused = FusedDraft(draft, ver)
    draft.fused.use_shared = os.environ.get("DP_SHARED", "1") == "1"  # block attention (as the traced rounds)
    ver.set_context_target_rowtile(draft.fused.ctx_buf, draft.fused.ctx_split)
    draft.set_draft_inputs(1234, 128)
    draft.set_context_positions(range(120, 124))
    draft._draft_device()
    draft._context_device()
    ttnn.synchronize_device(mesh_device)
    traced = os.environ.get("DP_TRACED", "1") == "1"  # the draft pass replayed from a trace (as the rounds run it)
    if traced:
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        draft._draft_device()
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    signpost("start")
    for _ in range(2):
        if traced:
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        else:
            draft._draft_device()
    ttnn.synchronize_device(mesh_device)
    signpost("stop")
    signpost("start")
    for _ in range(2):
        draft._context_device()
    ttnn.synchronize_device(mesh_device)
    signpost("stop")
