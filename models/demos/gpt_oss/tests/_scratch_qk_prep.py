# Scratch: dflash/qk_prep.py DraftQKPrep vs the drafter's ttnn path (rms_norm + to_memory_config +
# rotary_embedding_llama with RotarySetup.get_rot_mats) on random Q / K / V; timings.
import time

import pytest
import torch

import ttnn
from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric


def pcc(a, b):
    return float(torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1])


@pytest.mark.timeout(1200)
@parametrize_mesh_with_fabric([(1, 4)])
def test_qk_prep(mesh_device, device_params):
    from models.demos.gpt_oss.tt.dflash.qk_prep import DraftQKPrep
    from models.tt_transformers.tt.rope import RotarySetup

    M, nq, nkv, hd, eps = 8, 8, 2, 64, 1e-5
    torch.manual_seed(0)
    rep = ttnn.ReplicateTensorToMesh(mesh_device)

    def dev(t, dt=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            t, dtype=dt, layout=layout, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep
        )

    rope = RotarySetup(mesh_device, 32, hd, 4096, 150000.0, None, datatype=ttnn.bfloat16)
    trans = rope.get_both_trans_mats()["decode"]
    head_mc = ttnn.create_sharded_memory_config(
        shape=(32, hd),
        core_grid=ttnn.CoreGrid(y=4, x=8),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )
    prep = DraftQKPrep(mesh_device, M, nq, nkv, hd, eps, head_mc, rope.cos_matrix, rope.sin_matrix)
    qn_w = 1 + 0.3 * torch.randn(hd)
    kn_w = 1 + 0.3 * torch.randn(hd)
    gamma = prep.gamma(qn_w, kn_w)
    norm_w = lambda w: dev(w.reshape(1, 1, -1, 32), layout=ttnn.ROW_MAJOR_LAYOUT)
    qn_t, kn_t = norm_w(qn_w), norm_w(kn_w)

    def ref(q, k, v, rot_pos):
        rot = rope.get_rot_mats(rot_pos)
        out = []
        for t, w in ((q, qn_t), (k, kn_t)):
            n = ttnn.rms_norm(t, epsilon=eps, weight=w, compute_kernel_config=ck)
            s = ttnn.to_memory_config(n, head_mc)
            out.append(ttnn.experimental.rotary_embedding_llama(s, rot[0], rot[1], trans, is_decode_mode=True))
        out.append(ttnn.to_memory_config(v, head_mc))
        return out

    for p0 in (0, 5, 100, 1000, 3000):
        q = torch.randn(1, 32, nq, hd) * 3
        k = torch.randn(1, 32, nkv, hd) * 3
        v = torch.randn(1, 32, nkv, hd)
        qt, kt, vt = dev(q), dev(k), dev(v)
        pos = torch.zeros(1, 32, dtype=torch.int32)
        pos[0, :M] = torch.arange(p0, p0 + M)
        rp = dev(pos, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        got = prep(qt, kt, vt, gamma, rp)
        exp = ref(qt, kt, vt, rp)
        msg = []
        for name, g, e in zip("qkv", got, exp):
            gh = ttnn.to_torch(ttnn.get_device_tensors(g)[0]).float()[0, :M]
            eh = ttnn.to_torch(ttnn.get_device_tensors(e)[0]).float()[0, :M]
            msg.append(f"{name} PCC {pcc(gh, eh):.5f} max|d| {float((gh - eh).abs().max()):.4f}")
        for t in list(got) + exp:
            t.deallocate(True)
        print(f"QKP p0={p0}: " + "; ".join(msg), flush=True)
    # timings (eager, 8 layers' worth)
    ts = {}
    for name, f in (("prep", lambda: prep(qt, kt, vt, gamma, rp)), ("ttnn", lambda: ref(qt, kt, vt, rp))):
        for _ in range(3):
            for t in f():
                t.deallocate(True)
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(40):
            for t in f():
                t.deallocate(True)
        ttnn.synchronize_device(mesh_device)
        ts[name] = (time.perf_counter() - t0) / 40 * 1e6
    print(f"QKP eager us/call: prep {ts['prep']:.0f}, ttnn {ts['ttnn']:.0f}", flush=True)
