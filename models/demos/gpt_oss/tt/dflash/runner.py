# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Greedy DFlash generation with the gpt-oss target (tt_transformers Generator) and the TT drafter.

Bring-up verify: the target scores the block [last token, d1 .. d7] one decode step at a time and stops at the first
draft it disagrees with, so the target's KV cache and outputs are exactly those of plain greedy decoding (the output
equals the target's greedy output); each step's captured hidden states become the drafter's context.
"""

import os
import time

import torch

import ttnn

from .capture import TargetCapture


class DFlashRunner:
    def __init__(self, generator, model, page_table, kv_cache, draft, eos_ids=(200002, 200012)):
        self.generator, self.model, self.page_table, self.kv_cache = generator, model, page_table, kv_cache
        self.draft = draft
        self.capture = TargetCapture(model, draft.target_layer_ids)
        self.eos_ids = set(eos_ids)
        self.V = model.vocab_size

    def _decode(self, token, pos):
        out = self.generator.decode_forward(
            torch.tensor([token]),
            torch.tensor([pos]),
            enable_trace=False,
            page_table=self.page_table,
            kv_cache=self.kv_cache,
            sampling_params=None,
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=False,
            reset_sampling_state=False,
        )
        logits = out[0] if isinstance(out, tuple) else out
        return int(logits.reshape(-1)[: self.V].float().argmax())

    def generate(self, prompt_ids, max_new_tokens):
        """Returns (new tokens, per-round accepted counts, per-round (start, drafts), timings)."""
        L = len(prompt_ids)
        t0 = time.perf_counter()
        logits = self.generator.prefill_forward_text(
            torch.tensor([prompt_ids]),
            page_table=self.page_table,
            kv_cache=self.kv_cache,
            prompt_lens=[L],
            enable_trace=False,
            warmup_prefill=False,
        )
        ids = list(prompt_ids) + [int(logits.reshape(-1)[: self.V].float().argmax())]
        self.draft.add_prompt_context(self.capture.prefill_features(L))
        t_prefill = time.perf_counter() - t0
        accept, drafts, t_draft, t_verify = [], [], 0.0, 0.0
        B = self.draft.block
        while len(ids) - L < max_new_tokens and not (set(ids[L:]) & self.eos_ids):
            start = len(ids) - 1
            t = time.perf_counter()
            d = self.draft.draft(ids[-1], start)
            t_draft += time.perf_counter() - t
            drafts.append((start, d))
            t = time.perf_counter()
            block = [ids[-1]] + d
            feats = []
            for k in range(B):
                post = self._decode(block[k], start + k)
                feats.append(self.capture.decode_features())
                if k == B - 1 or d[k] != post:
                    break
            t_verify += time.perf_counter() - t
            ids += d[:k] + [post]
            self.draft.add_context(torch.cat(feats), range(start, start + k + 1))
            accept.append(k + 1)
        return ids[L:], accept, drafts, dict(prefill=t_prefill, draft=t_draft, verify=t_verify)


class BatchedDFlashRunner(DFlashRunner):
    """Greedy DFlash for every user of the batched (32-user) decode path. Each user is prefilled into its own slot; per
    round the drafter drafts 4 users per call, and the target verifies all users together one decode step at a time
    (a user that hit a mismatch, or is done, sits out with position -1), so each user's output equals its plain greedy
    output."""

    def __init__(self, generator, model, page_table, kv_cache, draft, batch=32, **kw):
        super().__init__(generator, model, page_table, kv_cache, draft, **kw)
        self.batch = batch

    def _decode_batch(self, tokens, positions):
        out = self.generator.decode_forward(
            torch.tensor(tokens),
            torch.tensor(positions),
            enable_trace=False,
            page_table=self.page_table,
            kv_cache=self.kv_cache,
            sampling_params=None,
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=False,
            reset_sampling_state=False,
        )
        logits = out[0] if isinstance(out, tuple) else out
        return logits.reshape(self.batch, -1)[:, : self.V].float().argmax(-1).tolist()

    def generate_batch(self, prompts, max_new_tokens):
        """prompts: one token list per user (len == batch). Returns (per-user new tokens, per-user accepted counts)."""
        Bu, Bk = self.batch, self.draft.block
        ids, lens = [], []
        for u, p in enumerate(prompts):
            logits = self.generator.prefill_forward_text(
                torch.tensor([p]),
                page_table=self.page_table,
                kv_cache=self.kv_cache,
                prompt_lens=[len(p)],
                empty_slots=[u],
                enable_trace=False,
                warmup_prefill=False,
            )
            ids.append(list(p) + [int(logits.reshape(-1)[: self.V].float().argmax())])
            lens.append(len(p))
            self.draft.add_prompt_context(self.capture.prefill_features(len(p)), user=u)
        accept = [[] for _ in range(Bu)]

        def done(u):
            return len(ids[u]) - lens[u] >= max_new_tokens or bool(set(ids[u][lens[u] :]) & self.eos_ids)

        while not all(done(u) for u in range(Bu)):
            active = [u for u in range(Bu) if not done(u)]
            starts = {u: len(ids[u]) - 1 for u in active}
            drafts = {}
            for g in range(0, len(active), 4):
                grp = active[g : g + 4]
                for u, d in zip(grp, self.draft.draft_users([ids[u][-1] for u in grp], [starts[u] for u in grp], grp)):
                    drafts[u] = d
            verifying = set(active)
            result, feats = {}, {u: [] for u in active}
            for k in range(Bk):
                toks = [0] * Bu
                pos = [-1] * Bu
                for u in verifying:
                    toks[u] = ([ids[u][-1]] + drafts[u])[k]
                    pos[u] = starts[u] + k
                post = self._decode_batch(toks, pos)
                rows = self.capture.decode_features()
                for u in list(verifying):
                    feats[u].append(rows[u : u + 1])
                    if k == Bk - 1 or drafts[u][k] != post[u]:
                        result[u] = (k, post[u])
                        verifying.discard(u)
                if not verifying:
                    break
            for u in active:
                k, p = result[u]
                ids[u] += drafts[u][:k] + [p]
                self.draft.add_context(torch.cat(feats[u]), range(starts[u], starts[u] + k + 1), user=u)
                accept[u].append(k + 1)
        return [ids[u][lens[u] :] for u in range(Bu)], accept


class FastDFlashRunner:
    """Greedy batch-1 DFlash with the 8-row fused verify pass (dflash/verify.py) and the traced drafter
    (DFlashDraft.enable_trace). Per round: draft (trace) -> the 7 drafts to the host -> verify (trace) -> the 8 target
    argmaxes and the accepted rows' layer features to the host -> drafter context update (trace)."""

    def __init__(self, generator, model, page_table, kv_cache, draft, eos_ids=(200002, 200012)):
        from .verify import FusedVerify

        self.generator, self.model, self.page_table, self.kv_cache = generator, model, page_table, kv_cache
        self.draft = draft
        self.capture = TargetCapture(model, draft.target_layer_ids)
        self.eos_ids = set(eos_ids)
        self.V = model.vocab_size
        kv = kv_cache[0] if isinstance(kv_cache[0][0], list) else kv_cache
        self.verify = FusedVerify(model, page_table, kv, rows=draft.block, capture_layers=draft.target_layer_ids)
        draft._shared_lm = (self.verify.lm_weights, self.verify.lm_n, len(self.verify.lm_head.assignments) // 8)
        draft.enable_trace()
        self.verify.set_context_target(draft.t_ctx)
        self.verify.enable_device_argmax()
        draft.feed_verify(self.verify.tokens)
        if os.environ.get("DFLASH_FUSED_DRAFT", "1") == "1":
            from .fused_draft import FusedDraft

            draft.fused = FusedDraft(draft, self.verify)
            self.verify.set_context_target_rowtile(draft.fused.ctx_buf, draft.fused.ctx_split)
        self.verify_trace = None

    def _verify(self, block, start):
        self.verify.set_inputs(block, start)
        if self.verify_trace is None:
            self.verify.forward()
        else:
            ttnn.execute_trace(self.model.mesh_device, self.verify_trace, cq_id=0, blocking=False)
        return self.verify.argmax()

    def capture_traces(self):
        mesh = self.model.mesh_device
        self.verify_trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        self.verify.forward()
        ttnn.end_trace_capture(mesh, self.verify_trace, cq_id=0)
        self.draft.capture_traces()

    def prefill(self, prompt_ids):
        L = len(prompt_ids)
        logits = self.generator.prefill_forward_text(
            torch.tensor([prompt_ids]),
            page_table=self.page_table,
            kv_cache=self.kv_cache,
            prompt_lens=[L],
            enable_trace=False,
            warmup_prefill=False,
        )
        # 256-row chunks: the verify pass's persistent L1 buffers leave less room for the matmul's circular buffers.
        self.draft.add_prompt_context(self.capture.prefill_features(L), chunk=256)
        return int(logits.reshape(-1)[: self.V].float().argmax())

    def generate(self, prompt_ids, max_new_tokens, first=None):
        """Returns (new tokens, per-round accepted counts, timings). first: the prefill's next token (prefill done)."""
        L = len(prompt_ids)
        t0 = time.perf_counter()
        if first is None:
            first = self.prefill(prompt_ids)
        ids = list(prompt_ids) + [first]
        t_prefill = time.perf_counter() - t0
        accept, rounds = [], []
        B = self.draft.block
        mesh = self.model.mesh_device
        while len(ids) - L < max_new_tokens and not (set(ids[L:]) & self.eos_ids):
            t = time.perf_counter()
            start = len(ids) - 1
            # draft -> (on device) verify block -> verify -> one host read of [block, target argmaxes]
            self.draft.set_draft_inputs(ids[-1], start)
            self.verify.set_positions(start)
            if self.verify_trace is None:
                self.draft._draft_device()
                self.verify.forward()
            else:
                ttnn.execute_trace(mesh, self.draft.trace_draft, cq_id=0, blocking=False)
                ttnn.execute_trace(mesh, self.verify_trace, cq_id=0, blocking=False)
            t1 = time.perf_counter()
            block, post = self.verify.read_result()
            d = block[1:]
            t2 = time.perf_counter()
            k = 0
            while k < B - 1 and d[k] == post[k]:
                k += 1
            ids += d[:k] + [post[k]]
            self.draft.add_context_traced(None, range(start, start + k + 1))
            t3 = time.perf_counter()
            accept.append(k + 1)
            rounds.append((t1 - t, t2 - t1, t3 - t2))
        return ids[L:], accept, dict(prefill=t_prefill, rounds=rounds)
