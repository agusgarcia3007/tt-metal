# Scratch: a token-frequency table of gpt-oss-20b's own greedy output (DFlash traced generation, == target greedy up to
# near-ties) on ~40 varied prompts, for a frequency-ranked draft vocabulary; saved to generated/dflash_vocab_freq.pt.
import os
from pathlib import Path

import pytest
import torch

from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric

NEW = int(os.environ.get("CORPUS_NEW", "512"))
TOPICS = [
    "photosynthesis",
    "the French Revolution",
    "black holes",
    "compound interest",
    "the immune system",
    "machine learning",
    "climate change",
    "the Roman Empire",
    "plate tectonics",
    "jazz music",
]
PROMPTS = (
    [f"Explain {t} in a few paragraphs." for t in TOPICS]
    + [f"Write a short story about {t}." for t in TOPICS[:5]]
    + [
        "Write a Python class for a linked list with insert, delete and search.",
        "Implement Dijkstra's algorithm in C++.",
        "Write a bash script that backs up a directory with a timestamp.",
        "Write a JavaScript debounce function.",
        "Write a Rust function that parses a CSV line.",
        "Write unit tests for a function that reverses a string.",
        "Write a SQL schema for a library with books, members and loans.",
        "Explain big-O of merge sort with code.",
    ]
    + [
        "A car travels 300 km at 60 km/h and returns at 40 km/h. What is the average speed?",
        "Solve for x: 2x^2 - 8x + 6 = 0.",
        "What is the probability of rolling two sixes with two dice?",
        "A shop discounts a $80 item by 15% then adds 8% tax. What is the final price?",
        "How many ways can 5 people sit in a row?",
        "Find the derivative of x^3 * sin(x).",
        "If a train leaves at 3pm going 80 km/h, when has it gone 200 km?",
        "What is 17% of 340? Explain.",
    ]
    + [
        "Give me tips to learn a new language.",
        "What should I pack for a week of hiking?",
        "Write a polite email declining a meeting.",
        "Summarize the plot of Romeo and Juliet.",
        "What are the pros and cons of remote work?",
        "Suggest a weekly workout plan for a beginner.",
        "How do I make a sourdough starter?",
        "Write a limerick about a cat.",
    ]
)


@pytest.mark.timeout(7200)
@parametrize_mesh_with_fabric([(1, 4)])
def test_dflash_corpus(mesh_device, device_params):
    import models.demos.gpt_oss.demo.text_demo as text_demo
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

    def encode(text):
        e = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True
        )
        if isinstance(e, dict) or hasattr(e, "input_ids"):
            e = e["input_ids"]
        return [int(t) for t in e]

    ids0 = encode(PROMPTS[0])
    fast.generate(ids0, 16)
    fast.capture_traces()
    counts = torch.zeros(m.vocab_size, dtype=torch.long)
    n = 0
    for p in PROMPTS:
        toks, _, _ = fast.generate(encode(p), NEW)
        t = torch.tensor(toks, dtype=torch.long)
        counts += torch.bincount(t, minlength=m.vocab_size)
        n += len(toks)
    out = Path(__file__).parents[4] / "generated" / "dflash_vocab_freq.pt"
    torch.save(counts, out)
    print(f"CORPUS {n} tokens, {int((counts > 0).sum())} distinct -> {out}", flush=True)
