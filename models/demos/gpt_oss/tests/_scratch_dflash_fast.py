# Scratch: FastDFlashRunner (8-row fused verify + traced drafter) vs plain greedy decode on a real prompt: identical
# tokens, acceptance, per-round time (eager compile run, then traced).
import os
import statistics
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric

NEW = int(os.environ.get("DFLASH_NEW", "64"))


@pytest.mark.timeout(7200)
@parametrize_mesh_with_fabric([(1, 4)])
def test_dflash_fast(mesh_device, device_params):
    import models.demos.gpt_oss.demo.text_demo as text_demo
    from models.demos.gpt_oss.config import MeshConfig, ModeConfig
    from models.demos.gpt_oss.tt.dflash.draft import DFlashDraft
    from models.demos.gpt_oss.tt.dflash.runner import DFlashRunner, FastDFlashRunner
    from models.tt_transformers.tt.generator import Generator

    ctx = 4096
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
    generator = Generator(model, model_args, mesh_device, processor=processor, tokenizer=tokenizer)
    m = model[0]

    def mem(tag):
        v = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1)
        print(
            f"DF L1 {tag}: allocated/bank {v.total_bytes_allocated_per_bank / 1024:.0f} KB, largest free "
            f"{v.largest_contiguous_bytes_free_per_bank / 1024:.0f} KB",
            flush=True,
        )

    mem("model loaded")
    wdt = ttnn.bfloat8_b if os.environ.get("DRAFT_BF8") else ttnn.bfloat16
    draft = DFlashDraft(mesh_device, m, max_positions=ctx, weight_dtype=wdt)
    mem("drafter")
    fast = FastDFlashRunner(generator, m, page_table, kv_cache, draft)
    mem("verify + drafter trace inputs")
    if os.environ.get("MEMDBG"):
        pass

        mem("end")
        return
    refs = torch.load(Path(__file__).parents[4] / "generated" / "dflash_reference.pt")
    which = [int(x) for x in os.environ.get("DFLASH_PROMPTS", "0,1,2").split(",")]
    plain_runner = DFlashRunner.__new__(DFlashRunner)
    plain_runner.generator, plain_runner.page_table, plain_runner.kv_cache, plain_runner.V = (
        generator,
        page_table,
        kv_cache,
        m.vocab_size,
    )
    plain = {}
    for w in which:
        ids = refs[w]["prompt_ids"]
        first = fast.prefill(ids)
        g = [first]
        for k in range(NEW - 1):
            g.append(plain_runner._decode(g[-1], len(ids) + k))
        plain[w] = g
        if os.environ.get("DETCHK"):
            first2 = fast.prefill(ids)
            g2 = [first2]
            for k in range(NEW - 1):
                g2.append(plain_runner._decode(g2[-1], len(ids) + k))
            print(f"DF plain greedy deterministic (prompt {w}): {g2 == g}", flush=True)
    for mode in ("eager", "traced"):
        if mode == "traced":
            fast.capture_traces()
        for w in which:
            ids = refs[w]["prompt_ids"]
            toks, acc, tm = fast.generate(ids, NEW)
            n = min(len(toks), NEW)
            same = toks[:n] == plain[w][:n]
            r = tm["rounds"][1:]
            dt, vt, ct = (statistics.median(x[i] for x in r) * 1e3 for i in range(3))
            per_round = dt + vt + ct
            a = sum(acc) / len(acc)
            print(
                f"DF {mode} prompt {w}: {n} tokens == plain greedy: {same}; accept {a:.2f} over {len(acc)} rounds; "
                f"round ms issue {dt:.2f} wait+read {vt:.2f} ctx {ct:.2f} = {per_round:.2f} -> "
                f"{a / per_round * 1e3:.1f} tok/s/user",
                flush=True,
            )
            if not same:
                bad = next(i for i in range(n) if toks[i] != plain[w][i])
                print(f"DF   first difference at token {bad}: {toks[bad]} vs {plain[w][bad]}", flush=True)
    if os.environ.get("BREAKDOWN"):
        # Per-stage wall time of traced rounds with a device sync after each stage (host work vs device work).
        ids = refs[which[0]]["prompt_ids"]
        first = fast.prefill(ids)
        seq = list(ids) + [first]
        mesh = mesh_device
        st = {k: [] for k in ("set", "exec_d", "exec_v", "read", "set_c", "exec_c")}

        def tm(key, f):
            t = time.perf_counter()
            r = f()
            ttnn.synchronize_device(mesh)
            st[key].append(time.perf_counter() - t)
            return r

        for rnd in range(12):
            start = len(seq) - 1
            if not (fast.device_rounds and rnd):
                tm("set", lambda: (fast.draft.set_draft_inputs(seq[-1], start), fast.verify.set_positions(start)))
            else:
                st["set"].append(0.0)
            tm("exec_d", lambda: ttnn.execute_trace(mesh, fast.draft.trace_draft, cq_id=0, blocking=False))
            tm("exec_v", lambda: ttnn.execute_trace(mesh, fast.verify_trace, cq_id=0, blocking=False))
            block, post = tm("read", fast.verify.read_result)
            d = block[1:]
            k = 0
            while k < 7 and d[k] == post[k]:
                k += 1
            seq += d[:k] + [post[k]]
            if fast.device_rounds:
                st["set_c"].append(0.0)
            else:
                tm("set_c", lambda: fast.draft.set_context_positions(range(start, start + k + 1)))
            tm("exec_c", lambda: ttnn.execute_trace(mesh, fast.draft.trace_ctx, cq_id=0, blocking=False))
        print(
            "DF breakdown (median ms): "
            + ", ".join(f"{k} {statistics.median(v[2:]) * 1e3:.2f}" for k, v in st.items()),
            flush=True,
        )
