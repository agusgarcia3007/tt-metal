# Scratch (device profiler, BA_PROF=1): one BlockAttention call per position for the compute zones of ba_compute.cpp
# (TT_METAL_DEVICE_PROFILER=1; $CLAUDE_JOB_DIR/tmp/bazones.py parses profile_log_device.csv).
import os

import pytest

import ttnn
from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric

CTX = int(os.environ.get("SKV_CTX", "131072"))
POSITIONS = [int(p) for p in os.environ.get("SKV_POS", "1000,4096,16384").split(",")]
S = int(os.environ.get("BA_S", "16"))


@pytest.mark.timeout(7200)
@parametrize_mesh_with_fabric([(1, 4)])
def test_ba_devprof(mesh_device, device_params):
    import models.demos.gpt_oss.demo.text_demo as text_demo
    from models.demos.gpt_oss.config import MeshConfig, ModeConfig
    from models.demos.gpt_oss.tt.dflash.block_attention import BlockAttention, free_cores
    from models.demos.gpt_oss.tt.dflash.shared_kv_attention import SharedKVAttention
    from models.demos.gpt_oss.tt.dflash.verify import FusedVerify

    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=4, ep=1))
    model_args, model, page_table, kv_cache, tokenizer, processor, _ = text_demo.prepare_gpt_oss_generator_args(
        num_devices=4,
        data_parallel=1,
        mesh_device=mesh_device,
        global_batch_size=1,
        optimizations=None,
        max_seq_len=CTX,
        page_params={"page_block_size": 64, "page_max_num_blocks_per_dp": CTX // 64},
        paged_attention=True,
        mesh_config=mesh_config,
        state_dict=None,
    )
    m = model[0]
    kv = kv_cache[0] if isinstance(kv_cache[0][0], list) else kv_cache
    v = FusedVerify(m, page_table, kv, rows=8)
    li = next(i for i, L in enumerate(m.layers) if not L.self_attn.config.sliding_window)
    attn = m.layers[li].self_attn
    k_cache, v_cache = kv[li]
    sinks, scale = attn.weights.decode_sinks, attn.config.scaling
    sp = SharedKVAttention(mesh_device, v.M, v.nq, v.nkv, v.hd, page_table, 64, None, ttnn.L1_MEMORY_CONFIG)
    sp.prepare([k_cache, v_cache], [sinks])
    ba = BlockAttention(mesh_device, v.M, v.nq, v.nkv, v.hd, 64, free_cores(mesh_device, set(), v.nkv * S), S)
    for p in POSITIONS:
        v.set_positions(p)
        ba(sp.q, k_cache, v_cache, v.pos, v.page_table, sinks, scale)  # compile
        ttnn.synchronize_device(mesh_device)
        ttnn.ReadDeviceProfiler(mesh_device)
        ba(sp.q, k_cache, v_cache, v.pos, v.page_table, sinks, scale)
        ttnn.synchronize_device(mesh_device)
        ttnn.ReadDeviceProfiler(mesh_device)
        print(f"DEVPROF pos {p}", flush=True)
