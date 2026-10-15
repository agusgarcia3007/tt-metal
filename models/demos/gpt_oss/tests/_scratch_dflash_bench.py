# Scratch: DFlash vs plain batch-1 decode on a prompt set (math / code / chat), both traced, NEW tokens per prompt:
# DFlash tok/s/user (FastDFlashRunner.generate round times), plain tok/s/user (traced fused decode, on-device greedy),
# per prompt and per category, and whether the DFlash tokens equal plain greedy.
import os
import statistics
import time

import pytest
import torch

from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric

NEW = int(os.environ.get("BENCH_NEW", "256"))
PROMPTS = [
    ("math", "Solve step by step: a train travels 120 km in 1.5 hours. What is its average speed in km/h?"),
    ("math", "If 3x + 5 = 20, what is x? Show your work."),
    ("math", "A rectangle has a perimeter of 30 cm and a length of 9 cm. What is its area?"),
    ("math", "What is the sum of the first 50 positive integers? Explain."),
    ("code", "Write a quicksort in Python."),
    ("code", "Write a Python function that checks whether a string is a palindrome, with tests."),
    ("code", "Implement binary search in C++."),
    ("code", "Write a SQL query that returns the second highest salary from an employees table."),
    ("chat", "How many moons does Earth have?"),
    ("chat", "Give me three tips for better sleep."),
    ("chat", "Explain what a black hole is to a 10-year-old."),
    ("chat", "Write a haiku about autumn."),
]


@pytest.mark.timeout(7200)
@parametrize_mesh_with_fabric([(1, 4)])
def test_dflash_bench(mesh_device, device_params):
    import models.demos.gpt_oss.demo.text_demo as text_demo
    from models.common.sampling import SamplingParams
    from models.demos.gpt_oss.config import MeshConfig, ModeConfig
    from models.demos.gpt_oss.tt.dflash.draft import DFlashDraft
    from models.demos.gpt_oss.tt.dflash.runner import FastDFlashRunner
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
    draft = DFlashDraft(mesh_device, m, max_positions=ctx)
    fast = FastDFlashRunner(generator, m, page_table, kv_cache, draft)
    sampling = SamplingParams(temperature=[0], top_k=[1], top_p=[1.0], enable_log_probs=[False], num_logprobs=[0])
    which = [int(i) for i in os.environ["BENCH_PROMPTS"].split(",")] if os.environ.get("BENCH_PROMPTS") else None
    prompts = [p for i, p in enumerate(PROMPTS) if which is None or i in which]

    def encode(text):
        e = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True
        )
        if isinstance(e, dict) or hasattr(e, "input_ids"):
            e = e["input_ids"]
        return [int(t) for t in e]

    def plain_decode(ids, first):
        # the DFlash feature capture hooks read to the host: off for the traced plain decode
        hooks = [(L, L.capture_input) for L in m.layers]
        for L, _ in hooks:
            L.capture_input = None
        try:
            return _plain_decode(ids, first)
        finally:
            for L, h in hooks:
                L.capture_input = h

    def _plain_decode(ids, first):
        tok = torch.tensor([first])
        pos = torch.tensor([len(ids)])
        out, ms = [first], []
        for i in range(NEW - 1):
            t = time.perf_counter()
            tok, _ = generator.decode_forward(
                tok.reshape(-1)[:1],
                pos,
                enable_trace=True,
                page_table=page_table,
                kv_cache=kv_cache,
                sampling_params=sampling,
                reload_inputs=i == 0,
                reload_page_table=False,
                reload_sampling_params=True,
                reset_sampling_state=i == 0,
            )
            out.append(int(tok.reshape(-1)[0]))
            if i:
                ms.append((time.perf_counter() - t) * 1e3)
            pos += 1
        return out, statistics.fmean(ms)

    # warm up both paths (compile + trace capture) on the first prompt
    ids0 = encode(prompts[0][1])
    first = fast.prefill(ids0)
    fast.generate(ids0, 16, first=first)
    fast.capture_traces()
    first = fast.prefill(ids0)
    plain_decode(ids0, first)

    rows = []
    for cat, text in prompts:
        ids = encode(text)
        first = fast.prefill(ids)
        toks, acc, tm = fast.generate(ids, NEW, first=first)
        n = min(len(toks), NEW)
        r = tm["rounds"]
        d_ms = sum(sum(x) for x in r) * 1e3  # all rounds (issue + wait / read + context)
        d_tps = n / d_ms * 1e3
        first = fast.prefill(ids)
        plain, p_ms = plain_decode(ids, first)
        same = toks[:n] == plain[:n]
        bad = next((i for i in range(n) if toks[i] != plain[i]), None)
        rows.append((cat, sum(acc) / len(acc), d_tps, 1e3 / p_ms))
        print(
            f"BENCH {cat} '{text[:40]}': {n} tokens, accept {sum(acc) / len(acc):.2f}, DFlash {d_tps:.1f} tok/s/u, "
            f"plain {1e3 / p_ms:.1f} tok/s/u ({p_ms:.3f} ms/token) -> {d_tps * p_ms / 1e3:.2f}x; == plain greedy: "
            f"{same}{'' if same else f' (first difference at token {bad})'}",
            flush=True,
        )
    for cat in ("math", "code", "chat", None):
        sel = [x for x in rows if cat is None or x[0] == cat]
        if not sel:
            continue
        a = statistics.fmean(x[1] for x in sel)
        d = statistics.fmean(x[2] for x in sel)
        p = statistics.fmean(x[3] for x in sel)
        print(
            f"BENCH {cat or 'all'}: mean accept {a:.2f}, DFlash {d:.1f} tok/s/u, plain {p:.1f} tok/s/u -> {d / p:.2f}x",
            flush=True,
        )
