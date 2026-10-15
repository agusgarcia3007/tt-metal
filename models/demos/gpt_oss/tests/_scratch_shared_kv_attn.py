# Scratch: the verify block's long-context attention, per-row ttnn SDPA (every row reads the K / V cache) vs
# SharedKVAttention (dflash/shared_kv_attention.py: one read for the 8 rows), on one full-attention layer with random
# K / V and Q: output agreement (PCC per row) and device time per call at several block positions.
import os
import time

import pytest
import torch

import ttnn
from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric

CTX = int(os.environ.get("SKV_CTX", "131072"))
POSITIONS = [int(p) for p in os.environ.get("SKV_POS", "1100,4096,16384,65536,130000").split(",")]


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


@pytest.mark.timeout(7200)
@parametrize_mesh_with_fabric([(1, 4)])
def test_shared_kv_attn(mesh_device, device_params):
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
    torch.manual_seed(0)
    shard = ttnn.ShardTensorToMesh(mesh_device, dim=1)
    for c in (k_cache, v_cache):
        s = list(c.shape)
        host = ttnn.from_torch(
            torch.randn(s[0], s[1] * 4, s[2], s[3]), dtype=c.dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=shard
        )
        ttnn.copy_host_to_device_tensor(host, c)
    q = torch.randn(1, v.M, v.nq, v.hd)
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(
            q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
        ),
        v.q,
    )
    W = int(os.environ.get("SKV_WINDOW", "512"))
    shared = SharedKVAttention(mesh_device, v.M, v.nq, v.nkv, v.hd, page_table, 64, None, ttnn.L1_MEMORY_CONFIG)
    win = SharedKVAttention(mesh_device, v.M, v.nq, v.nkv, v.hd, page_table, 64, None, ttnn.L1_MEMORY_CONFIG, window=W)
    for s_ in (shared, win):
        s_.prepare([k_cache, v_cache], [attn.weights.decode_sinks])
    v.row_attn = None  # _sdpa: the ttnn per-row path
    sinks, scale, ck = attn.weights.decode_sinks, attn.config.scaling, attn.program_config.get_compute_kernel_config()

    def per_row(window=None):
        return ttnn.transformer.paged_scaled_dot_product_attention_decode(
            v.q,
            k_cache,
            v_cache,
            cur_pos_tensor=v.pos,
            sliding_window_size=window,
            attention_sink=sinks,
            page_table_tensor=v.page_table,
            scale=scale,
            program_config=v._sdpa_pc(None),
            compute_kernel_config=ck,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    def one_read(s_=shared):
        s_.begin(v.pos, v.page_table)
        return s_(v.q, k_cache, v_cache, sinks, scale, ck)

    def dev0(t):
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()

    def timed(fn):
        for _ in range(3):
            o = fn()
            if o is not shared.out and o is not win.out:
                o.deallocate(True)
        ttnn.synchronize_device(mesh_device)
        n = 20
        t = time.perf_counter()
        for _ in range(n):
            o = fn()
            if o is not shared.out and o is not win.out:
                o.deallocate(True)
        ttnn.synchronize_device(mesh_device)
        return (time.perf_counter() - t) / n * 1e3

    for p in POSITIONS:
        v.set_positions(p)
        a = per_row()
        ra = dev0(a).clone()
        a.deallocate(True)
        rb = dev0(one_read()).clone()
        pccs = [_pcc(ra[0, r, :16], rb[0, r, :16]) for r in range(v.M)]
        # window: row r attends [shift * 64, p + r] (shift = the whole blocks before p + 7 + 1 - W): per-row SDPA with
        # window p + r - shift * 64 + 1, row r of call r
        shift = max(0, p + v.M - W) // 64
        rw = dev0(one_read(win)).clone()
        wp = []
        for r in range(v.M):
            o = per_row(p + r - shift * 64 + 1)
            wp.append(_pcc(dev0(o)[0, r, :16], rw[0, r, :16]))
            o.deallocate(True)
        t_row, t_one, t_win = timed(per_row), timed(one_read), timed(lambda: one_read(win))
        print(
            f"SKV pos={p}: per-row {t_row:.3f} ms, one-read {t_one:.3f} ms, one-read window {W} {t_win:.3f} ms "
            f"(per-row window {timed(lambda: per_row(W)):.3f} ms); PCC per row min {min(pccs):.5f} (window "
            f"{min(wp):.5f}); max |diff| {float((ra[0, :, :16] - rb[0, :, :16]).abs().max()):.4f}",
            flush=True,
        )
