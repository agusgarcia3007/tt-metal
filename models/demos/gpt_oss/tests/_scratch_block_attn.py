# Scratch: BlockAttention (dflash/block_attention.py: one K / V read per kv head for the 8 verify rows, sliced history
# + tree merge) standalone vs per-row ttnn SDPA on one full-attention layer with random K / V / Q: PCC per row, and
# traced device time per call vs per-row ttnn SDPA, the spec SDPA (SharedKVAttention.attend) and row attention.
import os
import time

import pytest
import torch

import ttnn
from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric

CTX = int(os.environ.get("SKV_CTX", "131072"))
POSITIONS = [int(p) for p in os.environ.get("SKV_POS", "5,40,100,300,1000,4096,16384,65536,130000").split(",")]
# BA_CFGS: "S:G:min_chunks[:tiles per core],..." (slices per kv head and tile group, chunks per group, fewest chunks per
# slice, packed Q tiles per core)
CFGS = [tuple(int(x) for x in c.split(":")) for c in os.environ.get("BA_CFGS", "16:8:4").split(",")]
TIME = os.environ.get("BA_TIME", "1") == "1"
UNI_W = int(os.environ.get("BA_UNI_W", "512"))  # the drafter window checked in uniform mode
N = 20


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


@pytest.mark.timeout(7200)
@parametrize_mesh_with_fabric([(1, 4)])
def test_block_attn(mesh_device, device_params):
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
    sinks, scale, ck = attn.weights.decode_sinks, attn.config.scaling, attn.program_config.get_compute_kernel_config()
    shared = SharedKVAttention(mesh_device, v.M, v.nq, v.nkv, v.hd, page_table, 64, None, ttnn.L1_MEMORY_CONFIG)
    shared.prepare([k_cache, v_cache], [sinks])
    shared.load_q(v.q)  # the packed Q (the verify QKV writes it with spec_tt)
    bas, unis, slides = {}, {}, {}
    for cfg in CFGS:
        S, G, mc = cfg[:3]
        tc = cfg[3] if len(cfg) > 3 else 2
        os.environ["BA_G"], os.environ["BA_MIN_CHUNKS"], os.environ["BA_TPC"] = str(G), str(mc), str(tc)
        n = BlockAttention.cores_needed(v.nkv, v.M, S)
        bas[f"S{S}G{G}m{mc}t{tc}"] = BlockAttention(
            mesh_device, v.M, v.nq, v.nkv, v.hd, 64, free_cores(mesh_device, set(), n), S
        )
        if os.environ.get("BA_SLIDE_W"):  # a sliding-window layer: row r attends its last W positions, sinks
            slides[f"S{S}G{G}m{mc}t{tc}"] = BlockAttention(
                mesh_device,
                v.M,
                v.nq,
                v.nkv,
                v.hd,
                64,
                free_cores(mesh_device, set(), n),
                S,
                window=int(os.environ["BA_SLIDE_W"]),
            )
        if os.environ.get("BA_UNI") == "1":  # the drafter's mode: every row attends [0, pos[M - 1]], no sinks
            for w in (0, UNI_W):
                unis[(f"S{S}G{G}m{mc}t{tc}", w)] = BlockAttention(
                    mesh_device,
                    v.M,
                    v.nq,
                    v.nkv,
                    v.hd,
                    64,
                    free_cores(mesh_device, set(), n),
                    S,
                    uniform=True,
                    window=w,
                    q_rows=True,  # (the drafter's case: Q in the rows layout)
                )
    ra = v.row_attn
    v.row_attn = None  # _sdpa: the ttnn per-row path

    def dev0(t):
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()

    def per_row():
        return ttnn.transformer.paged_scaled_dot_product_attention_decode(
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

    def traced_us(fn):
        fn()
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

    # host reference (device 0's shard): K / V cache, Q, sinks
    kh, vh = dev0(k_cache), dev0(v_cache)  # [blocks, nkv, 64, 64]
    qh = dev0(v.q)[0, :, : v.nq]  # [M, heads, 64]
    sk = dev0(sinks).reshape(-1, 32)[: v.nq, 0] * scale  # sink logits
    pt0 = page_table.reshape(-1).long()

    def host_ref(p, uniform=False, window=0):
        out = torch.zeros(v.M, v.nq, v.hd)
        for r in range(v.M):
            n = p + (v.M - 1 if uniform else r) + 1
            lo = max(0, n - window) if window else 0
            blocks = pt0[: (n + 63) // 64]
            for g in range(v.nkv):
                K = kh[blocks, g].reshape(-1, 64)[lo:n]
                V = vh[blocks, g].reshape(-1, 64)[lo:n]
                for hh in range(8):
                    h = 8 * g + hh
                    sc = K @ qh[r, h] * scale if uniform else torch.cat([K @ qh[r, h] * scale, sk[h : h + 1]])
                    w = torch.softmax(sc, 0)
                    out[r, h] = w[: n - lo] @ V
        return out

    for p in POSITIONS:
        v.set_positions(p)
        href = host_ref(p) if p < 20000 else None
        shared.begin(v.pos, v.page_table)
        a = per_row()
        ref = dev0(a).clone()
        a.deallocate(True)
        row = {}
        for S, ba in bas.items():
            ba(shared.q, k_cache, v_cache, v.pos, v.page_table, sinks, scale)
            shared._copy("SPEC_OUT", (ba.out, shared.out))
            got = dev0(shared.out).clone()
            pccs = [_pcc(ref[0, r, :16], got[0, r, :16]) for r in range(v.M)]
            err = float((ref[0, :, :16] - got[0, :, :16]).abs().max())
            print(f"BA pos={p} {S}: PCC per row min {min(pccs):.5f} max |diff| {err:.4f}", flush=True)
            if href is not None:
                hp = [_pcc(href[r], got[0, r, :16]) for r in range(v.M)]
                rp = [_pcc(href[r], ref[0, r, :16]) for r in range(v.M)]
                print(
                    f"BA host pos={p} {S}: BA vs host min {min(hp):.5f}, per-row ttnn vs host min {min(rp):.5f}; "
                    f"BA nan {int(got[0, :, :16].isnan().sum())}",
                    flush=True,
                )
            if min(pccs) < 0.999:
                print("BA rows", [f"{x:.4f}" for x in pccs], flush=True)
        if slides and href is not None:
            W = int(os.environ["BA_SLIDE_W"])
            sref = host_ref(p, window=W)
            for S, ba in slides.items():
                ba(shared.q, k_cache, v_cache, v.pos, v.page_table, sinks, scale)
                shared._copy("SPEC_OUT", (ba.out, shared.out))
                got = dev0(shared.out).clone()
                sp = [_pcc(sref[r], got[0, r, :16]) for r in range(v.M)]
                print(f"BA slide pos={p} {S} window={W}: vs host min {min(sp):.5f}", flush=True)
        if unis and href is not None:
            urefs = {w: host_ref(p, uniform=True, window=w) for w in (0, UNI_W)}
            for (S, w), ba in unis.items():
                ba(v.q, k_cache, v_cache, v.pos, v.page_table, None, scale)
                shared._copy("SPEC_OUT", (ba.out, shared.out))
                got = dev0(shared.out).clone()
                up = [_pcc(urefs[w][r], got[0, r, :16]) for r in range(v.M)]
                print(f"BA uniform pos={p} {S} window={w}: vs host min {min(up):.5f}", flush=True)
        if not TIME:
            continue

        def spec():
            o = shared.attend(k_cache, v_cache, sinks, scale, ck)
            o.deallocate(True)

        def pr():
            per_row().deallocate(True)

        row["per-row"] = traced_us(pr)
        row["spec"] = traced_us(spec)
        if ra is not None and p + v.M <= v.row_attn_max:
            row["row attention"] = traced_us(lambda: ra(v.q, k_cache, v_cache, v.pos, v.page_table, sinks, scale))
        for S, ba in bas.items():
            row[f"BA {S}"] = traced_us(lambda: ba(shared.q, k_cache, v_cache, v.pos, v.page_table, sinks, scale))
        print(f"BAT pos={p}: " + ", ".join(f"{k} {u:.1f} us" for k, u in row.items()), flush=True)
