# Scratch: dflash/row_attention.py RowAttention vs torch (8 rows at consecutive positions, paged BF8 K / V cache,
# sinks, full and sliding-window attention) and vs paged_scaled_dot_product_attention_decode; timings.
import os
import time

import pytest
import torch

import ttnn
from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric


def pcc(a, b):
    return float(torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1])


@pytest.mark.timeout(1800)
@parametrize_mesh_with_fabric([(1, 4)])
def test_row_attn(mesh_device, device_params):
    from models.demos.gpt_oss.tt.dflash.row_attention import RowAttention

    M, nq, nkv, hd, bs = 8, 16, 2, 64, 64
    nblk = 64
    torch.manual_seed(0)
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    dev = lambda t, dt=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT: ttnn.from_torch(
        t, dtype=dt, layout=layout, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep
    )
    kc = torch.randn(nblk, nkv, bs, hd) * 2
    vc = torch.randn(nblk, nkv, bs, hd)
    k = dev(kc, ttnn.bfloat8_b)
    v = dev(vc, ttnn.bfloat8_b)
    kq = ttnn.to_torch(ttnn.get_device_tensors(k)[0]).float()  # the BF8-rounded cache
    vq = ttnn.to_torch(ttnn.get_device_tensors(v)[0]).float()
    perm = torch.randperm(nblk).int()
    pt = ttnn.from_torch(perm[None].repeat(M, 1), dtype=ttnn.int32, device=mesh_device, mesh_mapper=rep)
    scale = hd**-0.5
    sinks = torch.randn(nq) * 2
    sink_rows = torch.zeros(nq, 32)
    sink_rows[:, 0] = sinks / scale
    sinks_t = dev(sink_rows)
    ra = RowAttention(
        mesh_device, M, nq, nkv, hd, bs, max_chunks=int(os.environ.get("RA_NK", "32")), cores_start=(0, 2)
    )
    ck = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    for p0, window in ((5, 0), (37, 0), (100, 0), (300, 0), (700, 0), (100, 128), (300, 128)):
        q = torch.randn(1, M, nq, hd)
        qt = dev(q)
        qq = ttnn.to_torch(ttnn.get_device_tensors(qt)[0]).float()[0]  # [M, nq, hd]
        pos = ttnn.from_torch(
            torch.arange(p0, p0 + M, dtype=torch.int32), dtype=ttnn.int32, device=mesh_device, mesh_mapper=rep
        )
        out = ra(qt, k, v, pos, pt, sinks_t, scale, window)
        got = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()[0][:, :nq]
        # torch reference
        ref = torch.zeros(M, nq, hd)
        for r in range(M):
            p = p0 + r
            lo = max(0, p + 1 - window) if window else 0
            ks = torch.stack([kq[perm[t // bs], :, t % bs] for t in range(lo, p + 1)])  # [L, nkv, hd]
            vs = torch.stack([vq[perm[t // bs], :, t % bs] for t in range(lo, p + 1)])
            for h in range(nq):
                g = h // (nq // nkv)
                s = (qq[r, h] @ ks[:, g].T) * scale
                s = torch.cat([s, sinks[h : h + 1]])
                w = torch.softmax(s, -1)[:-1]
                ref[r, h] = w @ vs[:, g]
        sd = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            qt,
            k,
            v,
            cur_pos_tensor=pos,
            attention_sink=sinks_t,
            page_table_tensor=pt,
            scale=scale,
            sliding_window_size=window or None,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
                q_chunk_size=0,
                k_chunk_size=128,
                exp_approx_mode=False,
            ),
            compute_kernel_config=ck,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        sdh = ttnn.to_torch(ttnn.get_device_tensors(sd)[0]).float()[0][:, :nq]
        ts = {}
        for name, f in (("row", lambda: ra(qt, k, v, pos, pt, sinks_t, scale, window)),):
            for _ in range(3):
                f()
            ttnn.synchronize_device(mesh_device)
            t = time.perf_counter()
            for _ in range(20):
                f()
            ttnn.synchronize_device(mesh_device)
            ts[name] = (time.perf_counter() - t) / 20 * 1e6
        if os.environ.get("RA_DBG_STAGE") == "7":
            raw = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()[0]
            print(
                "RA   dbg m row0 heads 0..5 col0",
                raw[0, :6, 0].tolist(),
                "cols 0..3 of head 0",
                raw[0, 0, :4].tolist(),
                flush=True,
            )
            ks = torch.stack([kq[perm[t // bs], :, t % bs] for t in range(0, p0 + 1)])
            S = (qq[0, :6] * scale) @ ks[:, 0].T
            print("RA   ref m", torch.maximum(S.max(-1).values, sinks[:6] * 1.0).tolist(), flush=True)
            break
        if os.environ.get("RA_DBG_STAGE") == "5":
            raw = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()[0]  # [M, 32? heads, 64]
            ks = torch.stack([kq[perm[t // bs], :, t % bs] for t in range(0, min(32, p0 + 1))])
            S = (qq[0] * scale) @ ks[:, 0].T  # [nq, L]
            print("RA   dbg S0 row0 head0..2 cols 0..7 got", raw[0, :3, :8].tolist(), flush=True)
            print("RA   ref S0", S[:3, :8].tolist(), "valid L", ks.shape[0], flush=True)
            print(
                "RA   dbg sink tile (out tile1) rows0..2 cols 0..4",
                raw[0, :3, 32:37].tolist(),
                "ref sinks",
                (sinks[:3]).tolist(),
                flush=True,
            )
            break
        if os.environ.get("RA_DBG_STAGE"):
            print(
                f"RA   stage {os.environ['RA_DBG_STAGE']} out row0 rows 0..3 cols 0..5:",
                got[0, :4, :6].tolist(),
                flush=True,
            )
            # reference intermediates for row 0, head 0..3
            r = 0
            p = p0
            ks = torch.stack([kq[perm[t // bs], :, t % bs] for t in range(0, p + 1)])
            vs = torch.stack([vq[perm[t // bs], :, t % bs] for t in range(0, p + 1)])
            S = (qq[0, :4] * scale) @ ks[:, 0].T
            m = torch.maximum(S.max(-1).values, sinks[:4])
            P = torch.exp(S - m[:, None])
            l = P.sum(-1) + torch.exp(sinks[:4] - m)
            print(
                "RA   ref O unnorm",
                (P @ vs[:, 0])[:, :6].tolist(),
                "\nRA   ref l",
                l.tolist(),
                "m",
                m.tolist(),
                flush=True,
            )
            break
        if os.environ.get("RA_DBG"):
            for r in (0, M - 1):
                print(
                    f"RA   row {r}: per-head max|d| {[round(float((got[r, h] - ref[r, h]).abs().max()), 3) for h in range(nq)]}",
                    flush=True,
                )
            print("RA   got[0,0,:6]", got[0, 0, :6].tolist(), "ref", ref[0, 0, :6].tolist(), flush=True)
        print(
            f"RA p0={p0} window={window}: row PCC {pcc(got, ref):.5f} max|d| {float((got - ref).abs().max()):.4f}; "
            f"sdpa PCC {pcc(sdh, ref):.5f}; row eager {ts['row']:.0f} us",
            flush=True,
        )
