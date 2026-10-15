# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The DFlash drafter's draft pass on the multi-row fused-decode ops (the same ops as the 8-row verify pass,
dflash/verify.py), replacing its ttnn matmuls / all-reduces / norms:

  embeddings [last token, mask x 7] -> entry boundary (row-tile residual + input norm)
  per layer: QKV stream (+ bias) into [32 users, heads, 64] head tensors -> q / k RMSNorm + RoPE (ttnn, one user per
             row) -> K / V rows into the draft cache (dflash/kv_write.py) -> paged SDPA (every row bounded by the
             block's last position: not causal) -> o_proj stream (+ bias on device 0) with the boundary all-reduce
             send -> boundary (all-reduce + residual + post-attention norm) -> dense gate|up stream (silu(g) * u, the
             union-expert kernels with one expert) -> down stream with the boundary send
  final norm -> the 8-row argmax LM head (the verify pass's) -> on-device merge into the verify block.

Weights are streamed in BF8 (the drafter's checkpoint is BF16), the residual is the boundary's BF16 row-tile vector
(the ttnn drafter keeps it FP32).
"""

import os
from pathlib import Path

import torch

import ttnn
from models.tt_transformers.tt.load_checkpoints import reverse_permute

from ..fused_decode.boundary import PendingBoundary
from ..fused_decode.stream import (
    LinearStream,
    as_stream_tensor,
    stream_down_layout,
    stream_gate_up_layout,
    stream_linear_layout,
)
from ..fused_decode.union import ExpertDownUnion, ExpertGateUpUnion, act_buffer

TILE = ttnn.TILE_SIZE
ROWS = 32


class _Norm:
    def __init__(self, tt_weight):
        self.tt_weight = tt_weight


class FusedDraft:
    def __init__(self, draft, verify, dtype=ttnn.bfloat8_b):
        """draft: a DFlashDraft with enable_trace() and feed_verify() done; verify: the FusedVerify it feeds."""
        self.d, self.v = draft, verify
        mesh = self.mesh = draft.mesh_device
        b = self.b = verify.boundary
        D, H, hd, nq, nkv = draft.D, draft.hidden, draft.hd, draft.nq, draft.nkv
        self.M = M = draft.block
        banks = mesh.dram_grid_size().x
        I = draft.inter
        self.I_pad = I_pad = -(-I // (TILE * banks)) * TILE * banks
        self.g_ln1 = [b.flat_gamma(_Norm(L["ln1"])) for L in draft.L]
        self.g_ln2 = [b.flat_gamma(_Norm(L["ln2"])) for L in draft.L]
        self.g_final = b.flat_gamma(_Norm(draft.norm))
        self.qkv = LinearStream(
            mesh,
            H // TILE,
            (nq + 2 * nkv) * hd,
            dtype,
            readers=1,
            out_mode=3,
            heads=(nq * hd // TILE, nkv * hd // TILE),
            rows=M,
            prefetch_cols=2,
        )
        self.o = LinearStream(mesh, nq * hd // TILE, H, dtype, readers=2, x_pages=hd // TILE, out_mode=5, rows=M)
        self.gu = ExpertGateUpUnion(
            mesh, H, I_pad, 0.0, 1.0, readers=4, rows=M, dense_silu=True, max_experts=1, weight_dtype=dtype
        )
        self.down = ExpertDownUnion(
            mesh, H, I_pad, readers=4, rows=M, expert_outer=True, max_experts=1, weight_dtype=dtype, num_buffers=2
        )
        self._load_weights(dtype, banks)
        rep = ttnn.ReplicateTensorToMesh(mesh)

        def dram(t, dt=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mc=ttnn.DRAM_MEMORY_CONFIG):
            return ttnn.from_torch(t, dtype=dt, layout=layout, device=mesh, memory_config=mc, mesh_mapper=rep)

        self.q = dram(torch.zeros(1, ROWS, nq, hd))
        self.k = dram(torch.zeros(1, ROWS, nkv, hd))
        self.vv = dram(torch.zeros(1, ROWS, nkv, hd))
        self.act = act_buffer(mesh, I_pad, M, slots=1)
        u = torch.zeros(1, 64, dtype=torch.int32)
        u[0, 0] = 1  # one "expert": the dense MLP, id 0
        self.union = dram(u, ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT, ttnn.L1_MEMORY_CONFIG)
        w = torch.zeros(1, 32 * M)
        w[0, ::32] = 1.0  # every row's weight of expert 0
        self.weights = dram(w, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, ttnn.L1_MEMORY_CONFIG)

    def _load_weights(self, dtype, banks):
        d = self.d
        D, H, hd, nq, nkv, I, I_pad = d.D, d.hidden, d.hd, d.nq, d.nkv, d.inter, self.I_pad
        dt = {ttnn.bfloat8_b: "bfp8", ttnn.bfloat4_b: "bfp4", ttnn.bfloat16: "bf16"}[dtype]
        cache_dir = os.environ.get("TT_CACHE_PATH")
        base = None if cache_dir is None else f"{cache_dir}/dflash_fused_draft_{dt}_b{banks}_i{I_pad}"
        marker = None if base is None else Path(base + ".built")
        w = None
        if marker is None or not marker.exists():
            from safetensors.torch import load_file

            w = load_file(str(d.path / "model.safetensors"))
        self.w_qkv, self.w_o, self.w_gu, self.w_down = [], [], [], []
        for i in range(d.layers):
            lay = {"qkv": None, "o": None, "gu": None, "down": None}
            if w is not None:
                p = f"layers.{i}."
                a = p + "self_attn."
                q = reverse_permute(w[a + "q_proj.weight"], D * nq, D * nq * hd, H).float()
                k = reverse_permute(w[a + "k_proj.weight"], D * nkv, D * nkv * hd, H).float()
                v = w[a + "v_proj.weight"].float()
                qb = reverse_permute(w[a + "q_proj.bias"][:, None], D * nq, D * nq * hd, 1)[:, 0].float()
                kb = reverse_permute(w[a + "k_proj.bias"][:, None], D * nkv, D * nkv * hd, 1)[:, 0].float()
                vb = w[a + "v_proj.bias"].float()
                o = w[a + "o_proj.weight"].float().T  # [nq_all * hd, H]
                ob = w[a + "o_proj.bias"].float()
                g = w[p + "mlp.gate_proj.weight"].float()  # [I_all, H]
                u = w[p + "mlp.up_proj.weight"].float()
                dn = w[p + "mlp.down_proj.weight"].float().T  # [I_all, H]
                lay = {k_: [] for k_ in lay}
                qh, kh = nq * hd, nkv * hd
                for dv in range(D):
                    wq = torch.cat([q[dv * qh : (dv + 1) * qh], k[dv * kh : (dv + 1) * kh], v[dv * kh : (dv + 1) * kh]])
                    bq = torch.cat(
                        [qb[dv * qh : (dv + 1) * qh], kb[dv * kh : (dv + 1) * kh], vb[dv * kh : (dv + 1) * kh]]
                    )
                    lay["qkv"].append(stream_linear_layout(wq.T.contiguous(), bq, banks))
                    lay["o"].append(
                        stream_linear_layout(o[dv * qh : (dv + 1) * qh], ob if dv == 0 else torch.zeros_like(ob), banks)
                    )
                    gd = torch.nn.functional.pad(g[dv * I : (dv + 1) * I].T, (0, I_pad - I))
                    ud = torch.nn.functional.pad(u[dv * I : (dv + 1) * I].T, (0, I_pad - I))
                    lay["gu"].append(
                        stream_gate_up_layout(torch.cat([gd, ud], -1)[None], torch.zeros(1, 2 * I_pad), banks)
                    )
                    dd = torch.nn.functional.pad(dn[dv * I : (dv + 1) * I], (0, 0, 0, I_pad - I))
                    lay["down"].append(stream_down_layout(dd[None], torch.zeros(1, H), banks))

            def put(name, rows):
                cache = None if base is None else f"{base}_l{i}_{name}"
                return as_stream_tensor(self.mesh, lay[name], rows, dtype, cache)

            self.w_qkv.append(put("qkv", self.qkv.weight_rows()))
            self.w_o.append(put("o", self.o.weight_rows()))
            self.w_gu.append(put("gu", self.gu.weight_rows(1)))
            self.w_down.append(put("down", self.down.weight_rows(1)))
        if marker is not None:
            marker.touch()

    def _norm_rope(self, t, w, rot):
        """[1, 32, heads, 64] DRAM -> RMSNorm over head_dim -> one user per core -> RoPE (decode)."""
        d = self.d
        n = ttnn.rms_norm(t, epsilon=d.eps, weight=w, compute_kernel_config=d.ck)
        s = ttnn.to_memory_config(n, d.head_mc)
        n.deallocate(True)
        r = ttnn.experimental.rotary_embedding_llama(s, rot[0], rot[1], d.trans_mat, is_decode_mode=True)
        s.deallocate(True)
        return r

    def forward(self):
        d, v, b, M = self.d, self.v, self.b, self.M
        toks = ttnn.slice(d.t_toks, (0, 0), (1, M))  # the block's M rows only (the L1 embedding is 8 rows)
        emb = ttnn.embedding(
            toks, d.target.embedding_weight, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        toks.deallocate(True)
        emb = ttnn.reshape(emb, (1, 1, M, d.hidden))
        pending = PendingBoundary(b, emb, self.g_ln1[0], entry=True)
        rot = d.rope.get_rot_mats(d.t_rot)
        for i in range(d.layers):
            L = d.L[i]
            self.qkv(pending, self.w_qkv[i], (self.q, self.k, self.vv))
            res = pending.residual_out
            pending.x.deallocate(True)
            pending.residual.deallocate(True)
            qr = self._norm_rope(self.q, L["q_norm"], rot)
            kr = self._norm_rope(self.k, L["k_norm"], rot)
            vs = ttnn.to_memory_config(self.vv, d.head_mc)
            kc, vc = d.caches[i]
            d.kv_write(kr, vs, kc, vc, d.t_pos, d.page_table)
            kr.deallocate(True)
            vs.deallocate(True)
            attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                qr,
                kc,
                vc,
                cur_pos_tensor=d.t_bound,
                page_table_tensor=d.page_table,
                scale=d.hd**-0.5,
                program_config=d.sdpa_pc,
                compute_kernel_config=d.ck,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            qr.deallocate(True)
            self.o(attn, self.w_o[i], v.partial, send=(b, "attn"))
            attn.deallocate(True)
            pending = PendingBoundary(b, res, self.g_ln2[i], v.partial, "attn", sent=True)
            res, x = pending.run()
            pending.residual.deallocate(True)
            self.gu(x, self.union, self.weights, self.w_gu[i], self.act)
            x.deallocate(True)
            self.down(self.act, self.union, self.weights, self.w_down[i], v.partial, send=(b, "moe"))
            gamma = self.g_ln1[i + 1] if i + 1 < d.layers else self.g_final
            pending = PendingBoundary(b, res, gamma, v.partial, "moe", sent=True)
        v.lm_head(pending, v.lm_weights, d.lists)
        pending.x.deallocate(True)
        pending.residual.deallocate(True)
        pending.residual_out.deallocate(True)
        d.argmax_gather(d.lists, d.t_toks, d.verify_tokens)
        for t in rot:
            t.deallocate(True)
