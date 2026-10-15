# Scratch: DraftQKPrep in joint mode with 16 per-row weighted heads (the drafter context K) vs ttnn rms_norm *
# per-head weight + rotary_embedding_llama.
import pytest
import torch

import ttnn
from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric


def pcc(a, b):
    return float(torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1])


@pytest.mark.timeout(1200)
@parametrize_mesh_with_fabric([(1, 4)])
def test_ctx_prep(mesh_device, device_params):
    from models.demos.gpt_oss.tt.dflash.qk_prep import DraftQKPrep
    from models.tt_transformers.tt.rope import RotarySetup

    M, H, hd, eps = 8, 16, 64, 1e-5
    torch.manual_seed(0)
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    dev = lambda t, dt=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT: ttnn.from_torch(
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
    prep = DraftQKPrep(mesh_device, M, H, 0, hd, eps, head_mc, rope.cos_matrix, rope.sin_matrix)
    assert prep.joint
    kn = 1 + 0.3 * torch.randn(H, hd)
    g = prep.gamma_rows(kn)
    kn_t = dev(kn.reshape(1, 1, H, hd))
    for p0 in (5, 300, 2000):
        k = torch.randn(1, 32, H, hd) * 3
        v = torch.randn(1, 32, H, hd)
        kt, vt = dev(k), dev(v)
        pos = torch.zeros(1, 32, dtype=torch.int32)
        pos[0, :M] = torch.arange(p0, p0 + M)
        rp = dev(pos, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        kr, ku, vs = prep(kt, kt, vt, g, rp)
        rot = rope.get_rot_mats(rp)
        n = ttnn.multiply(ttnn.rms_norm(kt, epsilon=eps, compute_kernel_config=ck), kn_t)
        ref = ttnn.experimental.rotary_embedding_llama(
            ttnn.to_memory_config(n, head_mc), rot[0], rot[1], trans, is_decode_mode=True
        )
        a = ttnn.to_torch(ttnn.get_device_tensors(kr)[0]).float()[0, :M]
        b = ttnn.to_torch(ttnn.get_device_tensors(ref)[0]).float()[0, :M]
        av = ttnn.to_torch(ttnn.get_device_tensors(vs)[0]).float()[0, :M]
        print(
            f"CTXP p0={p0}: k PCC {pcc(a, b):.6f} max|d| {float((a - b).abs().max()):.4f}; v max|d| {float((av - v[0, :M]).abs().max()):.4f}",
            flush=True,
        )
