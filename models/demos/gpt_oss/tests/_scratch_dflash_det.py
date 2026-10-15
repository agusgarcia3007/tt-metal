# Scratch: determinism of the teacher-forced DFlash loop (same prompt run twice in one process); was: teacher-forced DFlash acceptance: rounds follow a fixed reference continuation (plain greedy, computed once),
# so the drafter variants (ttnn / fused) are compared on the same text: accepted = matching drafts + 1 per round.
import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric

NEW = int(os.environ.get("TF_NEW", "96"))


@pytest.mark.timeout(7200)
@parametrize_mesh_with_fabric([(1, 4)])
def test_dflash_det(mesh_device, device_params):
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
    if os.environ.get("DET_PLAIN"):  # no DFlash objects at all: the plain model's prefill only

        class _F:
            pass

        fast = _F()
        fast.generator, fast.page_table, fast.kv_cache = generator, page_table, kv_cache
        draft = None
        alloc = os.environ.get("DET_ALLOC", "")
        keep = []
        if "ar" in alloc.split(","):
            from models.demos.gpt_oss.tt.dflash.draft import _AllReduce

            keep.append(_AllReduce(mesh_device, m.hf_config.hidden_size))
        if "sem" in alloc.split(","):
            grid = mesh_device.compute_with_storage_grid_size()
            cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
            keep += [ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(2)]
        if "rope" in alloc.split(","):
            from models.demos.gpt_oss.tt.model import create_rope_setup

            keep.append(create_rope_setup(mesh_device, m.hf_config, max_local_batch_size=32, datatype=ttnn.bfloat16))
    else:
        draft = DFlashDraft(mesh_device, m, max_positions=ctx)
    if draft is not None and os.environ.get("DET_STAGE") == "draft":

        class _F:
            pass

        fast = _F()
        fast.generator, fast.page_table, fast.kv_cache = generator, page_table, kv_cache
    elif draft is not None:
        fast = FastDFlashRunner(generator, m, page_table, kv_cache, draft)
        fused = draft.fused
    plain = DFlashRunner.__new__(DFlashRunner)
    plain.generator, plain.page_table, plain.kv_cache, plain.V = generator, page_table, kv_cache, m.vocab_size
    refs = torch.load(Path(__file__).parents[4] / "generated" / "dflash_reference.pt")
    PROMPT = int(os.environ.get("DET_PROMPT", "0"))
    REPS = int(os.environ.get("DET_REPS", "2"))
    r = refs[PROMPT]
    ids = r["prompt_ids"]
    L = len(ids)
    ref = torch.load(Path(__file__).parents[4] / "generated" / f"dflash_tf_ref_{PROMPT}.pt")
    if os.environ.get("DET_LEN"):  # truncate / repeat the prompt to this length
        n = int(os.environ["DET_LEN"])
        ids = (ids * (n // len(ids) + 1))[:n]
        L = n
    if os.environ.get("DET_LAYER0"):
        # record layer DET_LAYER's attention and MLP outputs (prefill) over repeated prefills
        li = int(os.environ["DET_LAYER0"])
        lay = m.layers[li]
        rec = {}

        def wrap(obj, attr, tag):
            f = getattr(obj, attr)

            def g(*a, **k):
                pre = not k.get("is_decode", True)
                if pre:
                    rec.setdefault(tag + "_in", []).append(
                        torch.cat([ttnn.to_torch(x).float() for x in ttnn.get_device_tensors(a[0])], dim=1).clone()
                    )
                out = f(*a, **k)
                if pre:
                    rec.setdefault(tag, []).append(
                        torch.cat([ttnn.to_torch(x).float() for x in ttnn.get_device_tensors(out)], dim=1).clone()
                    )
                return out

            setattr(obj, attr, g)

        wrap(lay, "self_attn", "attn")
        wrap(lay, "mlp", "mlp")
        if os.environ.get("DET_OPS"):
            # every ttnn op the layer's MLP runs in prefill, by call index
            pass

            calls = {"n": 0}
            names = os.environ["DET_OPS"].split(",")
            in_mlp = {"on": False}
            scope = os.environ.get("DET_SCOPE", "mlp")
            fmlp = getattr(lay, scope)

            def mlp_g(*a, **k):
                in_mlp["on"] = not k.get("is_decode", True)
                calls["n"] = 0
                try:
                    return fmlp(*a, **k)
                finally:
                    in_mlp["on"] = False

            setattr(lay, scope, mlp_g)
            for nm in names:
                mod = ttnn
                for part in nm.split(".")[:-1]:
                    mod = getattr(mod, part)
                leaf = nm.split(".")[-1]
                orig = getattr(mod, leaf)

                def mk(orig, nm):
                    def h(*a, **k):
                        out = orig(*a, **k)
                        if in_mlp["on"] and calls["n"] == 0 and os.environ.get("DET_REPEAT_OP"):
                            # rerun the first op on the same inputs: which (device, dim-1 block) outputs vary
                            base = [ttnn.to_torch(x).float() for x in ttnn.get_device_tensors(out)]
                            for rr in range(int(os.environ["DET_REPEAT_OP"])):
                                o2 = orig(*a, **k)
                                cur = [ttnn.to_torch(x).float() for x in ttnn.get_device_tensors(o2)]
                                bad = []
                                for dv, (x0, x1) in enumerate(zip(base, cur)):
                                    dd = (x1 - x0).abs()
                                    if float(dd.max()) > 0:
                                        blocks = sorted(
                                            {
                                                tuple(ix[:2])
                                                for ix in (dd.reshape(dd.shape[0], dd.shape[1], -1).amax(-1) > 0)
                                                .nonzero()
                                                .tolist()
                                            }
                                        )
                                        bad.append((dv, blocks))
                                print(f"DET repeat {rr}: {bad}", flush=True)
                                o2.deallocate(True)
                        if in_mlp["on"]:
                            calls["n"] += 1
                            t = out[0] if isinstance(out, (list, tuple)) else out
                            rec.setdefault(f"{calls['n']:02d}:{nm}", []).append(
                                torch.cat([ttnn.to_torch(x).float() for x in ttnn.get_device_tensors(t)], dim=1).clone()
                            )
                        return out

                    return h

                setattr(mod, leaf, mk(orig, nm))
        for i in range(3):
            fast.generator.prefill_forward_text(
                torch.tensor([ids]),
                page_table=fast.page_table,
                kv_cache=fast.kv_cache,
                prompt_lens=[L],
                enable_trace=False,
                warmup_prefill=False,
            )
        if os.environ.get("DET_SAVE"):
            torch.save({k: v[0] for k, v in rec.items()}, os.environ["DET_SAVE"])
        if os.environ.get("DET_CMP"):
            other = torch.load(os.environ["DET_CMP"])
            for k, v in rec.items():
                if k in other:
                    d = (v[0] - other[k]).float()
                    blk = (
                        [
                            [
                                round(float(d[0, dv, r * 32 : (r + 1) * 32].abs().max()), 4)
                                for r in range(d.shape[2] // 32)
                            ]
                            for dv in range(d.shape[1])
                        ]
                        if d.dim() == 4
                        else None
                    )
                    print(
                        f"DET cmp {k}: max|diff| {float(d.abs().max()):.4g} rel {float(d.norm() / other[k].norm()):.3e} per device x 32-row block max {blk}",
                        flush=True,
                    )
        for tag, v in rec.items():
            for i in range(1, len(v)):
                d = (v[i] - v[0]).reshape(-1, v[0].shape[-1])
                rows = (d.abs().amax(-1) > 0).nonzero().reshape(-1).tolist()
                if v[0].dim() == 4:
                    dd = (v[i] - v[0]).abs()
                    print(
                        f"DET   {tag} differing [d1, d2] blocks:",
                        sorted({(a, b // 32) for a, b, *_ in (dd > 0).nonzero().tolist()[:100000] for _ in [0]})[:12],
                        flush=True,
                    )
                print(
                    f"DET layer {li} {tag}: shape {tuple(v[0].shape)} rel {float(d.norm() / v[0].norm()):.2e} rows differing "
                    f"{len(rows)} first {rows[:4]} last {rows[-2:]}",
                    flush=True,
                )
        return
    if os.environ.get("DET_PREFILL"):
        # prefill repeated: compare the captured prefill features (host) and the following draft
        feats, outs = [], []
        for i in range(int(os.environ.get("DET_N", "4"))):
            generator_logits = fast.generator.prefill_forward_text(
                torch.tensor([ids]),
                page_table=fast.page_table,
                kv_cache=fast.kv_cache,
                prompt_lens=[L],
                enable_trace=False,
                warmup_prefill=False,
            )
            f = fast.capture.prefill_features(L)
            fh = f if isinstance(f, torch.Tensor) else ttnn.to_torch(ttnn.get_device_tensors(f)[0])
            feats.append(fh.float().clone())
            draft.add_prompt_context(f, chunk=256)
            draft.set_draft_inputs(ref[0], L)
            draft._draft_device()
            outs.append(ttnn.to_torch(ttnn.get_device_tensors(draft.verify_tokens)[0]).reshape(-1)[:8].tolist())
            if i > 0:
                dd = (feats[-1] - feats[0]).reshape(L, 5, -1)
                ref0 = feats[0].reshape(L, 5, -1)
                rel = [float(dd[:, j].norm() / ref0[:, j].norm()) for j in range(5)]
                bad_pos = (dd.abs().amax(dim=(1, 2)) > 0).nonzero().reshape(-1).tolist()
                print(
                    f"DET   rel diff per captured layer {[f'{x:.2e}' for x in rel]}; positions differing "
                    f"{len(bad_pos)}/{L} first {bad_pos[:8]}",
                    flush=True,
                )
            print(
                f"DET prefill {i}: logits argmax {int(generator_logits.reshape(-1)[: m.vocab_size].float().argmax())}"
                f" feat max|diff| vs 0 {float((feats[-1] - feats[0]).abs().max()):.4g} draft {outs[-1]}",
                flush=True,
            )
        return
    if os.environ.get("DET_DRAFT_ONLY"):
        # the drafter alone, repeated on identical inputs after one prefill (its KV rows are rewritten identically)
        fast.prefill(ids)
        outs = []
        for i in range(int(os.environ.get("DET_N", "20"))):
            draft.set_draft_inputs(ref[0], L)
            draft._draft_device()
            t = ttnn.to_torch(ttnn.get_device_tensors(draft.verify_tokens)[0]).reshape(-1)[:8].tolist()
            outs.append(t)
        print("DET draft-only distinct outputs:", len({tuple(o) for o in outs}), outs[:3], flush=True)
        for v in sorted({tuple(o) for o in outs}):
            print("DET   ", outs.count(list(v)), v, flush=True)
        return
    runs = []
    for rep in range(REPS):
        fast.prefill(ids)
        pos, log = 0, []
        while pos < NEW:
            start = L + pos
            draft.set_draft_inputs(ref[pos], start)
            fast.verify.set_positions(start)
            draft._draft_device()
            fast.verify.forward()
            block, post = fast.verify.read_result()
            d = block[1:]
            k = 0
            while k < 7 and d[k] == ref[pos + 1 + k]:
                k += 1
            log.append((pos, list(block), list(post), k))
            draft.set_context_positions(range(start, start + k + 1))
            draft._context_device()
            pos += k + 1
        runs.append(log)
        print(f"DET rep {rep}: rounds {len(log)}, accept {sum(x[3] + 1 for x in log) / len(log):.3f}", flush=True)
    a = runs[0]
    for rep, b in enumerate(runs[1:], 1):
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                print(
                    f"DET rep {rep} first diff at round {i} pos {x[0]}/{y[0]}:\n  drafts {x[1]}\n      vs {y[1]}\n  "
                    f"verify {x[2]}\n      vs {y[2]}",
                    flush=True,
                )
                break
        else:
            print(f"DET rep {rep}: identical", flush=True)
