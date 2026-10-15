# Scratch (device profiler): one call each of per-row SDPA, spec SDPA (packed, R=8) and spec SDPA (uniform, R=32) at
# one position, for kernel-level durations (TT_METAL_DEVICE_PROFILER=1; parse profile_log_device.csv per run id).
import os

import pytest

import ttnn
from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric

CTX = int(os.environ.get("SKV_CTX", "4096"))
POS = int(os.environ.get("SKV_POS", "1000"))


@pytest.mark.timeout(7200)
@parametrize_mesh_with_fabric([(1, 4)])
def test_skv_devprof(mesh_device, device_params):
    import models.demos.gpt_oss.demo.text_demo as text_demo
    from models.demos.gpt_oss.config import MeshConfig, ModeConfig
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
    sinks, scale, ck = attn.weights.decode_sinks, attn.config.scaling, attn.program_config.get_compute_kernel_config()
    sp = SharedKVAttention(mesh_device, v.M, v.nq, v.nkv, v.hd, page_table, 64, None, ttnn.L1_MEMORY_CONFIG)
    sp.prepare([k_cache, v_cache], [sinks])
    su = SharedKVAttention(
        mesh_device, v.M, v.nq, v.nkv, v.hd, page_table, 64, None, ttnn.L1_MEMORY_CONFIG, uniform=True
    )
    su.prepare([k_cache, v_cache], [])
    v.set_positions(POS)
    sp.begin(v.pos, v.page_table)
    su.begin(v.pos, v.page_table)

    def per_row():
        ttnn.transformer.paged_scaled_dot_product_attention_decode(
            v.q,
            k_cache,
            v_cache,
            cur_pos_tensor=v.pos,
            attention_sink=sinks,
            page_table_tensor=v.page_table,
            scale=scale,
            program_config=v._sdpa_pc(None),
            compute_kernel_config=ck,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        ).deallocate(True)

    def ra():
        v.row_attn(v.q, k_cache, v_cache, v.pos, v.page_table, sinks, scale)

    fns = [
        ("per-row", per_row),
        ("spec packed", lambda: sp.attend(k_cache, v_cache, sinks, scale, ck).deallocate(True)),
        ("spec uniform", lambda: su.attend(k_cache, v_cache, None, scale, ck).deallocate(True)),
    ]
    if POS + v.M <= v.row_attn_max:
        fns.append(("row attention", ra))
    for name, fn in fns:  # compile
        fn()
    ttnn.synchronize_device(mesh_device)
    ttnn.ReadDeviceProfiler(mesh_device)
    for name, fn in fns:
        fn()
        ttnn.synchronize_device(mesh_device)
        print(f"DEVPROF op {name}", flush=True)
    ttnn.ReadDeviceProfiler(mesh_device)
