# Scratch: device profile of the 8-row FusedVerify pass (2 eager passes between signposts) at position ISL.
import os

import pytest

import ttnn
from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric


@pytest.mark.timeout(3600)
@parametrize_mesh_with_fabric([(1, 4)])
def test_verify8_profile(mesh_device, device_params):
    from tracy import signpost

    import models.demos.gpt_oss.demo.text_demo as text_demo
    from models.demos.gpt_oss.config import MeshConfig, ModeConfig
    from models.demos.gpt_oss.tt.dflash.verify import FusedVerify

    isl = int(os.environ.get("ISL", "128"))
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
    m = model[0]
    kv = kv_cache[0] if isinstance(kv_cache[0][0], list) else kv_cache
    ver = FusedVerify(m, page_table, kv)
    toks = [1234, 200008, 976, 1825, 31064, 25, 392, 10930]
    ver.enable_device_argmax()
    ver.set_inputs(toks, isl)
    if os.environ.get("VP_SHARED", "1") == "1":  # the shared K / V path (block attention), as the DFlash rounds run it
        ver.use_shared, ver.use_row_attn = True, False
    ver.forward()  # compile
    ttnn.synchronize_device(mesh_device)
    traced = os.environ.get("PROF_TRACED", "1") == "1"
    if traced:
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        ver.forward()
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    signpost("start")
    for _ in range(2):
        if traced:
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        else:
            ver.forward()
    ttnn.synchronize_device(mesh_device)
    signpost("stop")
    print("VP done", ver.argmax(), flush=True)
