# Scratch: the device drafter vs the torch reference drafter (dflash/reference.py) on a long prompt (the Gutenberg
# instruct prompt of _scratch_dflash_isl_gutenberg.py, ISL_TOKENS tokens). Both get the same context: the device
# target's prefill features of positions < s. At cut points s (last first: a device draft writes its block's K / V at
# s .. s + 7 into the draft cache, which only cut points past s read), block [prompt[s], mask x 7] at positions
# s .. s + 7 -> 7 drafts each; prints how many device drafts equal the reference's.
import os
from pathlib import Path

import pytest
import torch

from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric

CACHE = Path(__file__).parents[3] / "tt_transformers" / "demo" / "context_cache"
CUTS = int(os.environ.get("REF_CUTS", "12"))


def _hf_tensor(name):
    import json

    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    p = Path(snapshot_download("openai/gpt-oss-20b", allow_patterns=["*.json"]))
    idx = json.loads((p / "model.safetensors.index.json").read_text())["weight_map"]
    p = Path(snapshot_download("openai/gpt-oss-20b", allow_patterns=[idx[name]]))
    with safe_open(str(p / idx[name]), "pt") as f:
        return f.get_tensor(name)


@pytest.mark.timeout(7200)
@parametrize_mesh_with_fabric([(1, 4)])
def test_dflash_ref_long(mesh_device, device_params):
    import json

    import models.demos.gpt_oss.demo.text_demo as text_demo
    from models.demos.gpt_oss.config import MeshConfig, ModeConfig
    from models.demos.gpt_oss.tt.dflash.draft import DFlashDraft
    from models.demos.gpt_oss.tt.dflash.reference import DraftReference
    from models.demos.gpt_oss.tt.dflash.runner import FastDFlashRunner
    from models.tt_transformers.tt.generator import Generator

    isl = int(os.environ["ISL_TOKENS"])
    ctx = max(4096, 1 << (isl + 64).bit_length())
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
    q = json.load(open(CACHE.parent / "sample_prompts" / "input_data_long_1k.json"))[0]["prompt"]
    mark = "XQXSENTINELXQX"
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "```" + mark + "```\n\n" + q}], add_generation_prompt=True, tokenize=False
    )
    pre, post = text.split(mark)
    pre_ids, post_ids = tokenizer.encode(pre, add_special_tokens=False), tokenizer.encode(
        post, add_special_tokens=False
    )
    ids = [int(t) for t in pre_ids + book_ids[: isl - len(pre_ids) - len(post_ids)] + post_ids]

    feats = {}
    add = draft.add_prompt_context

    def keep(f, **kw):
        feats["f"] = f.clone()
        return add(f, **kw)

    draft.add_prompt_context = keep
    fast.prefill(ids)
    F = feats["f"]  # [isl, 5 * hidden] the device target's features
    ref = DraftReference()
    emb = _hf_tensor("model.embed_tokens.weight").float()
    lm = _hf_tensor("lm_head.weight").float()
    B = ref.block_size
    # cut points spread over the book part of the prompt, last first
    cuts = sorted({int(s) for s in torch.linspace(len(pre_ids) + 16, isl - B - 1, CUTS).tolist()}, reverse=True)
    same = tot = ref_ok = dev_ok = 0
    for s in cuts:
        draft.set_draft_inputs(ids[s], s)
        draft._draft_device()
        # [last token, 7 drafts] (the fused draft pass writes them into the verify pass's token input)
        import ttnn

        dev = ttnn.to_torch(ttnn.get_device_tensors(draft.verify_tokens)[0]).reshape(-1)[1:B].long().tolist()
        noise = emb[torch.tensor([ids[s]] + [ref.mask_token_id] * (B - 1))]
        hid = ref.forward(F[:s].float(), noise, range(s), range(s, s + B))
        rd = (hid[1:] @ lm.T).argmax(-1).tolist()
        truth = ids[s + 1 : s + B]
        n = sum(a == b for a, b in zip(dev, rd))
        same += n
        tot += B - 1
        ref_ok += sum(a == b for a, b in zip(rd, truth))
        dev_ok += sum(a == b for a, b in zip(dev, truth))
        print(f"REFL s={s}: device {dev} ref {rd} text {truth}: {n}/7 equal", flush=True)
    print(
        f"REFL isl={isl}: device == reference {same}/{tot} drafts; matching the prompt text: device {dev_ok}, "
        f"reference {ref_ok} of {tot}",
        flush=True,
    )
