# Scratch: DFlash vs plain batch-1 decode at input length ISL_TOKENS on the tt_transformers long-context demo text
# (Project Gutenberg Frankenstein, pg84.txt, raw text as the gpt-oss demo feeds it: instruct=False, no chat template):
# the prompt is the book's first ISL_TOKENS tokens (past the book's 101,749 tokens: Pride and Prejudice, pg1342.txt,
# follows). Both paths traced, NEW tokens each from the same prefill; tok/s/user = tokens / decode time.
import os
import statistics
import time
from pathlib import Path

import pytest
import torch

from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric

NEW = int(os.environ.get("BENCH_NEW", "256"))
CACHE = Path(__file__).parents[3] / "tt_transformers" / "demo" / "context_cache"
BOOKS = [CACHE / "4c1a705addfd44fe41b8ce06d83e3fd5", CACHE / "pg1342.txt"]  # md5 of the pg84 URL: the demo's cache name


@pytest.mark.timeout(7200)
@parametrize_mesh_with_fabric([(1, 4)])
def test_dflash_isl_gutenberg(mesh_device, device_params):
    import models.demos.gpt_oss.demo.text_demo as text_demo
    from models.common.sampling import SamplingParams
    from models.demos.gpt_oss.config import MeshConfig, ModeConfig
    from models.demos.gpt_oss.tt.dflash.draft import DFlashDraft
    from models.demos.gpt_oss.tt.dflash.runner import FastDFlashRunner
    from models.tt_transformers.tt.generator import Generator

    isl = int(os.environ["ISL_TOKENS"])
    ctx = max(4096, 1 << (isl + NEW + 15).bit_length())  # positions up to isl + NEW + 8 (the last verify block)
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

    book_ids = []
    for book in BOOKS:
        book_ids += tokenizer.encode(book.read_text())
        if len(book_ids) >= isl:
            break
    if os.environ.get("GUT_INSTRUCT") == "1":
        # as the tt_transformers demo with instruct=True: "```" + book excerpt + "```\n\n" + the file's question, in
        # the chat template; the excerpt is cut so that the whole prompt is isl tokens
        import json

        q = json.load(open(CACHE.parent / "sample_prompts" / "input_data_long_1k.json"))[0]["prompt"]
        mark = "XQXSENTINELXQX"
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": "```" + mark + "```\n\n" + q}], add_generation_prompt=True, tokenize=False
        )
        pre, post = text.split(mark)
        pre_ids = tokenizer.encode(pre, add_special_tokens=False)
        post_ids = tokenizer.encode(post, add_special_tokens=False)
        k = isl - len(pre_ids) - len(post_ids)
        assert 0 < k <= len(book_ids), (k, len(book_ids))
        ids = [int(t) for t in pre_ids + book_ids[:k] + post_ids]
    else:
        assert len(book_ids) >= isl, (len(book_ids), isl)
        ids = [int(t) for t in book_ids[:isl]]

    def plain_decode(first):
        # the DFlash feature capture hooks read to the host: off for the traced plain decode
        hooks = [(L, L.capture_input) for L in m.layers]
        for L, _ in hooks:
            L.capture_input = None
        try:
            tok, pos = torch.tensor([first]), torch.tensor([isl])
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
        finally:
            for L, h in hooks:
                L.capture_input = h

    # compile everything before the first trace capture, then the plain decode trace, then the DFlash traces (a buffer
    # allocated after a capture can be overwritten by its replays); one prompt length: no later prefill compiles
    first = fast.prefill(ids)
    fast.generate(ids, 16, first=first)
    fast.compile_variants()
    plain_decode(first)
    fast.capture_traces()
    # the plain decode trace's compile pass writes mock K / V at position 0: prefill again. Decode and DFlash write
    # positions >= isl only (each written before it is read), so both timed runs share this prefill.
    t = time.perf_counter()
    first = fast.prefill(ids)
    ttft = time.perf_counter() - t
    toks, acc, tm = fast.generate(ids, NEW, first=first)
    n = min(len(toks), NEW)
    d_ms = sum(sum(x) for x in tm["rounds"]) * 1e3
    d_tps = n / d_ms * 1e3
    plain, p_ms = plain_decode(first)
    p_tps = 1e3 / p_ms
    same = next((i for i in range(n) if toks[i] != plain[i]), None)
    if os.environ.get("BENCH_SHOW"):
        print(f"DFISLG_TXT prompt tail {tokenizer.decode(ids[-60:])!r} -> {tokenizer.decode(toks[:n])!r}"[:2000])
        print(f"DFISLG_PLAIN {tokenizer.decode(plain[:n])!r}"[:1500])
    print(
        f"DFISLG isl={isl} ctx={ctx} prefill {ttft:.1f} s: {n} tokens, accept {sum(acc) / len(acc):.2f} "
        f"({len(acc)} rounds), DFlash {d_tps:.1f} tok/s/u, plain {p_tps:.1f} tok/s/u ({p_ms:.3f} ms/token) -> "
        f"{d_tps / p_tps:.2f}x; first difference vs plain greedy at token {same}",
        flush=True,
    )
