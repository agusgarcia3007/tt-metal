# Scratch: device time per call (traced: 20 calls per trace, replayed) of the verify block's attention pieces on one
# full-attention layer with random K / V: per-row ttnn SDPA (the old long path), row attention (the old short path,
# unfused), and the shared-KV path's pieces (Q pack copy, spec SDPA at several cores per kv head, output unpack copy).
import os
import time

import pytest
import torch

import ttnn
from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric

CTX = int(os.environ.get("SKV_CTX", "131072"))
POSITIONS = [int(p) for p in os.environ.get("SKV_POS", "300,1000,4096,16384,65536").split(",")]
CORES = [int(c) for c in os.environ.get("SKV_CORES_LIST", "2,4,8,16,32,55").split(",")]
N = 20


@pytest.mark.timeout(7200)
@parametrize_mesh_with_fabric([(1, 4)])
def test_skv_profile(mesh_device, device_params):
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
    shard = ttnn.ShardTensorToMesh(mesh_device, dim=1)
    for c in (k_cache, v_cache):
        s = list(c.shape)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                torch.randn(s[0], s[1] * 4, s[2], s[3]), dtype=c.dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=shard
            ),
            c,
        )
    sinks, scale, ck = attn.weights.decode_sinks, attn.config.scaling, attn.program_config.get_compute_kernel_config()
    shared = {}
    for c in CORES:
        os.environ["SKV_CORES"] = str(c)
        shared[c] = SharedKVAttention(mesh_device, v.M, v.nq, v.nkv, v.hd, page_table, 64, None, ttnn.L1_MEMORY_CONFIG)
        shared[c].prepare([k_cache, v_cache], [sinks])
    s0 = shared[CORES[-1]]
    uni = SharedKVAttention(
        mesh_device, v.M, v.nq, v.nkv, v.hd, page_table, 64, None, ttnn.L1_MEMORY_CONFIG, uniform=True
    )
    uni.prepare([k_cache, v_cache], [])

    def per_row():
        o = ttnn.transformer.paged_scaled_dot_product_attention_decode(
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
        )
        o.deallocate(True)

    def row_attn():
        v.row_attn(v.q, k_cache, v_cache, v.pos, v.page_table, sinks, scale)

    def spec_sdpa(s_, rows_per_cand=8):
        def f():
            o = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                s_.q,
                s_.views[k_cache.buffer_address()],
                s_.views[v_cache.buffer_address()],
                cur_pos_tensor=s_.pos,
                attention_sink=s_.sinks[sinks.buffer_address()] if rows_per_cand == 8 else None,
                page_table_tensor=s_.page_table,
                scale=scale,
                program_config=s_.pc,
                compute_kernel_config=ck,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                spec_multi_pos_tiles=s_.nkv * s_.Tt // s_.B,
                spec_rows_per_candidate=rows_per_cand,
            )
            o.deallocate(True)

        return f

    o_tmp = ttnn.from_torch(
        torch.zeros(1, v.nkv, s0.Tt * 32, 64),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def copy_q():
        s0._copy("SPEC_Q", (v.q, s0.q))

    def copy_out():
        s0._copy("SPEC_OUT", (o_tmp, s0.out))

    def traced_us(fn):
        fn()  # compile
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for _ in range(N):
            fn()
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        t = time.perf_counter()
        for _ in range(5):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        us = (time.perf_counter() - t) / (5 * N) * 1e6
        ttnn.release_trace(mesh_device, tid)
        return us

    for p in POSITIONS:
        v.set_positions(p)
        for s_ in shared.values():
            s_.begin(v.pos, v.page_table)
        row = {
            "per-row ttnn": traced_us(per_row),
            "copy Q": traced_us(copy_q),
            "copy out": traced_us(copy_out),
        }
        if p + v.M <= v.row_attn_max:
            row["row attention"] = traced_us(row_attn)
        for c, s_ in shared.items():
            row[f"spec cores={c}"] = traced_us(spec_sdpa(s_))
        uni.begin(v.pos, v.page_table)
        row["spec uniform (standard mask)"] = traced_us(spec_sdpa(uni, 32))
        print(f"SKVP pos={p}: " + ", ".join(f"{k} {u:.1f} us" for k, u in row.items()), flush=True)
