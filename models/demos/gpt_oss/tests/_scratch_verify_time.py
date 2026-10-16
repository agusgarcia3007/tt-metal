# Scratch: traced device time of one FusedVerify pass (8 rows, all 24 layers) per block position and attention variant:
# base (row attention to 1024 positions, then per-row ttnn SDPA: every row reads the K / V cache), spec (shared-KV spec
# SDPA op), and block attention (dflash/block_attention.py) under the configurations in VT_BA ("name:ENV=v,ENV=v;...").
import os
import time

import pytest

import ttnn
from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric

POSITIONS = [int(p) for p in os.environ.get("VT_POS", "1000,65536").split(",")]
N = int(os.environ.get("VT_N", "10"))


@pytest.mark.timeout(7200)
@parametrize_mesh_with_fabric([(1, 4)])
def test_verify_time(mesh_device, device_params):
    import models.demos.gpt_oss.demo.text_demo as text_demo
    from models.demos.gpt_oss.config import MeshConfig, ModeConfig
    from models.demos.gpt_oss.tt.dflash.block_attention import BlockAttention, free_cores
    from models.demos.gpt_oss.tt.dflash.verify import FusedVerify

    ctx = max(4096, 1 << (max(POSITIONS) + 64).bit_length())
    ctx = min(ctx, 131072)
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
    ver.enable_device_argmax()
    toks = [1234, 200008, 976, 1825, 31064, 25, 392, 10930]
    ba0 = ver.block_attn
    cfgs = [("base", None), ("spec", None)]
    for spec in os.environ.get("VT_BA", "ba:X=1").split(";"):
        name, envs = spec.split(":")
        cfgs.append((name, dict(e.split("=") for e in envs.split(","))))

    def make_ba(envs):
        old = {k: os.environ.get(k) for k in envs}
        os.environ.update(envs)
        try:
            S = int(os.environ.get("DFLASH_BA_S", "24"))
            grid = mesh_device.compute_with_storage_grid_size()
            taken = {(c.x, c.y) for c in ttnn.corerange_to_cores(ver.qkv.cores, row_wise=True)}
            taken |= {(c.x, c.y) for c in ttnn.corerange_to_cores(ver.boundary.cores, row_wise=True)}
            taken |= {(h.x, h.y) for h in getattr(ver.boundary, "helpers", [])}
            per = BlockAttention.cores_needed(ver.nkv, ver.M, 1)
            S = min(S, (grid.x * grid.y - len(taken)) // per)
            return BlockAttention(
                mesh_device, ver.M, ver.nq, ver.nkv, ver.hd, ba0.block, free_cores(mesh_device, taken, per * S), S
            )
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    bas = {name: make_ba(envs) for name, envs in cfgs if envs is not None}
    for p in POSITIONS:
        row = {}
        for name, envs in cfgs:
            ver.set_inputs(toks, p)
            if name == "base":
                ver.use_shared = False
            else:
                ver.use_shared, ver.use_row_attn = True, False
                ver.block_attn = bas.get(name)
            ver.forward()  # compile
            ttnn.synchronize_device(mesh_device)
            tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            ver.forward()
            ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            t = time.perf_counter()
            for _ in range(N):
                ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            row[name] = (time.perf_counter() - t) / N * 1e3
            ttnn.release_trace(mesh_device, tid)
        print(f"VT pos={p}: " + ", ".join(f"{k} {v:.3f} ms" for k, v in row.items()), flush=True)
