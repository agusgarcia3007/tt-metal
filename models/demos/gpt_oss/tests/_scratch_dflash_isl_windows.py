# Scratch: DFlash on the chat-template Gutenberg prompt (as _scratch_dflash_isl_gutenberg.py with GUT_INSTRUCT=1) at
# ISL_TOKENS, for several drafter attention windows in one process: one prefill, then per window (WINDOWS, 0 = none)
# the traces re-captured and BENCH_NEW tokens generated (the prompt's K / V and drafter context are untouched: decode
# writes positions >= isl only). Prints one DFISLG line per window (no plain decode: measured by _scratch_isl_sweep.py).
import json
import os
from pathlib import Path

import pytest

from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric

NEW = int(os.environ.get("BENCH_NEW", "256"))
CACHE = Path(__file__).parents[3] / "tt_transformers" / "demo" / "context_cache"


@pytest.mark.timeout(7200)
@parametrize_mesh_with_fabric([(1, 4)])
def test_dflash_isl_windows(mesh_device, device_params):
    import models.demos.gpt_oss.demo.text_demo as text_demo
    from models.demos.gpt_oss.config import MeshConfig, ModeConfig
    from models.demos.gpt_oss.tt.dflash.draft import DFlashDraft
    from models.demos.gpt_oss.tt.dflash.runner import FastDFlashRunner
    from models.tt_transformers.tt.generator import Generator

    isl = int(os.environ["ISL_TOKENS"])
    windows = [int(w) for w in os.environ.get("WINDOWS", "0,256,512,1024,2048,4096").split(",")]
    ctx = max(4096, 1 << (isl + NEW + 15).bit_length())
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

    book_ids = tokenizer.encode((CACHE / "4c1a705addfd44fe41b8ce06d83e3fd5").read_text())
    if len(book_ids) < isl:
        book_ids += tokenizer.encode((CACHE / "pg1342.txt").read_text())
    q = json.load(open(CACHE.parent / "sample_prompts" / "input_data_long_1k.json"))[0]["prompt"]
    mark = "XQXSENTINELXQX"
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "```" + mark + "```\n\n" + q}], add_generation_prompt=True, tokenize=False
    )
    pre, post = text.split(mark)
    pre_ids = tokenizer.encode(pre, add_special_tokens=False)
    post_ids = tokenizer.encode(post, add_special_tokens=False)
    ids = [int(t) for t in pre_ids + book_ids[: isl - len(pre_ids) - len(post_ids)] + post_ids]

    first = fast.prefill(ids)
    for w in windows:
        if fast.verify_trace is not None:
            fast.release_traces()
        fast.set_draft_window(w)
        fast.generate(ids, 16, first=first)  # eager: compiles this window's programs
        fast.compile_variants()
        fast.capture_traces()
        toks, acc, tm = fast.generate(ids, NEW, first=first)
        n = min(len(toks), NEW)
        d_ms = sum(sum(x) for x in tm["rounds"]) * 1e3
        print(
            f"DFISLG isl={isl} ctx={ctx} window={w}: {n} tokens, accept {sum(acc) / len(acc):.2f} ({len(acc)} rounds), "
            f"DFlash {n / d_ms * 1e3:.1f} tok/s/u, round {d_ms / len(acc):.3f} ms",
            flush=True,
        )
        if os.environ.get("SHOW_ACC") == "1":  # accepted tokens per round, in order (position = isl + running sum)
            print(f"DFACC window={w}: {''.join(str(a - 1) for a in acc)}", flush=True)
            print(f"DFTOK window={w}: {list(toks[:n])}", flush=True)  # the generated ids (compare paths exactly)
            if w == windows[0]:
                print(f"DFTXT {tokenizer.decode(toks[:n])!r}"[:1200], flush=True)
        if os.environ.get("PLAIN_REF") == "1":  # the target's own greedy decode (eager) on the same prompt
            from models.demos.gpt_oss.tt.dflash.runner import DFlashRunner

            fast.release_traces()  # (eager programs compiled after a capture could sit in the traces' space)
            plain = [first]
            for i in range(n - 1):
                plain.append(DFlashRunner._decode(fast, plain[-1], len(ids) + i))
            k = next((i for i in range(n) if toks[i] != plain[i]), None)
            print(f"DFPLAIN window={w}: == plain greedy over {n} tokens: {k is None}; first diff at {k}", flush=True)
