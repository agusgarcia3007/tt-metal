# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DFlash verify of the gpt-oss target in one pass: the M = 8 block rows [last token, d1 .. d7] of one user at
positions p .. p + 7 through all layers at once, with the fused-decode streamed ops in their multi-row form
(fused_decode/: the residual stream in the row-tile layout of boundary.py, every weight read once for the M rows):

  embeddings -> entry boundary (row-tile residual + input norm)
  per layer: QKV (+ RoPE + K / V cache write of the M rows) -> paged SDPA decode with the M rows as M users sharing the
             page table (row r at position p + r: causal inside the block) -> o_proj -> boundary (all-reduce + residual
             + post-attention norm) -> router (per-row top-4 + union) -> union-expert gate|up -> down -> boundary
  LM head: per-row argmax lists (LinearStream out_mode 6), merged on the host.

The target's argmax for row r is its prediction for position p + r + 1; the K / V rows of rejected drafts sit past the
accepted length and are overwritten by the next block (attention reads positions <= each row's own). Layer outputs
1, 6, 11, 16, 21 (the drafter's context) are kept in row-tile form per round.
"""

import torch

import ttnn

from ..fused_decode.boundary import DecodeBoundary, flat_memory_config, flat_shape, rowtile_pages, rowtile_to_rows
from ..fused_decode.config import (
    DOWN_STREAM_READERS,
    GATE_UP_STREAM_READERS,
    LM_HEAD_DECODE_WEIGHT_DTYPE,
    LM_HEAD_STREAM_PREFETCH,
    LM_HEAD_STREAM_READERS,
    OPROJ_DECODE_WEIGHT_DTYPE,
    OPROJ_STREAM_READERS,
    QKV_DECODE_WEIGHT_DTYPE,
    QKV_STREAM_READERS,
    ROUTER_DECODE_WEIGHT_DTYPE,
    sdpa_decode_program_config,
)
from ..fused_decode.stream import LinearStream, QKVRopeCache, merge_argmax, rope_rotation_tiles
from ..fused_decode.union import ExpertDownUnion, ExpertGateUpUnion, act_buffer, union_buffers

TILE = ttnn.TILE_SIZE


class FusedVerify:
    def __init__(self, model, page_table, kv_cache, rows=8, capture_layers=(1, 6, 11, 16, 21)):
        """model: the fused-decode gpt-oss Model (one user); page_table: its [1, blocks] INT32 page table (torch);
        kv_cache: per layer [k_cache, v_cache]."""
        self.model = model
        mesh = self.mesh = model.mesh_device
        cfg = model.hf_config
        self.M = M = rows
        self.H = H = cfg.hidden_size
        self.layers = model.layers
        self.kv_cache = kv_cache
        self.capture_layers = list(capture_layers)
        attn0 = self.layers[0].self_attn.config
        tp = mesh.shape[1]
        self.nq, self.nkv, self.hd = attn0.num_heads // tp, attn0.num_kv_heads // tp, attn0.head_dim
        q_cols, k_cols = self.nq * self.hd // TILE, self.nkv * self.hd // TILE
        self.boundary = DecodeBoundary(mesh, H, cfg.rms_norm_eps, cluster_axis=1, rows=M)
        b = self.boundary
        self.gamma_in = [b.flat_gamma(L.input_layernorm) for L in self.layers]
        self.gamma_post = [b.flat_gamma(L.post_attention_layernorm) for L in self.layers]
        self.gamma_final = b.flat_gamma(model.norm)

        self.qkv = LinearStream(
            mesh,
            H // TILE,
            (self.nq + 2 * self.nkv) * self.hd,
            QKV_DECODE_WEIGHT_DTYPE,
            readers=QKV_STREAM_READERS,
            out_mode=3,
            heads=(q_cols, k_cols),
            rows=M,
        )
        self.o_proj = LinearStream(
            mesh,
            self.nq * self.hd // TILE,
            H,
            OPROJ_DECODE_WEIGHT_DTYPE,
            readers=OPROJ_STREAM_READERS,
            x_pages=self.hd // TILE,
            out_mode=5,
            rows=M,
        )
        mlp0 = self.layers[0].mlp
        self.top_k, self.n_experts = mlp0.router.top_k, mlp0.router.num_experts
        self.router = LinearStream(
            mesh,
            H // TILE,
            self.n_experts,
            ROUTER_DECODE_WEIGHT_DTYPE,
            readers=1,
            out_mode=4,
            out_page_bytes=128,
            heads=(self.top_k, self.n_experts),
            rows=M,
        )
        ex = mlp0.experts
        inter_pad = ex.decode_weights.intermediate_padded
        self.gate_up = ExpertGateUpUnion(
            mesh, H, inter_pad, ex.config.swiglu_limit, ex.config.alpha, readers=GATE_UP_STREAM_READERS, rows=M
        )
        self.down = ExpertDownUnion(mesh, H, inter_pad, readers=DOWN_STREAM_READERS, rows=M)
        term = model.decode_terminal
        self.vd = term.vd
        self.lm_head = LinearStream(
            mesh,
            H // TILE,
            term.vd,
            LM_HEAD_DECODE_WEIGHT_DTYPE,
            readers=LM_HEAD_STREAM_READERS,
            out_mode=6,
            prefetch_cols=LM_HEAD_STREAM_PREFETCH,
            rows=M,
        )
        self.lm_weights = term.weights

        rep = ttnn.ReplicateTensorToMesh(mesh)
        dram, l1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
        # Logical [1, M, heads, head_dim] (tile-padded to 32 head rows): SDPA takes the head count from the shape.
        self.q = ttnn.from_torch(
            torch.zeros(1, M, self.nq, self.hd),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=dram,
            mesh_mapper=rep,
        )
        self.pages = rowtile_pages(H, M)
        self.partial = ttnn.from_torch(
            torch.zeros(flat_shape(H, self.pages)),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=flat_memory_config(mesh, H, self.pages),
            mesh_mapper=rep,
        )
        self.union, self.weights = union_buffers(mesh, M)
        self.act = act_buffer(mesh, inter_pad, M)
        n_lists = len(self.lm_head.assignments)
        self.lists = ttnn.from_torch(
            torch.zeros(1, n_lists * M * 2, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=l1,
            mesh_mapper=rep,
        )
        # Inputs (host-written each round): token ids, row positions, the rows' page table (user 0's row repeated).
        self.tokens = ttnn.from_torch(
            torch.zeros(1, M, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=dram,
            mesh_mapper=rep,
        )
        self.pos = ttnn.from_torch(
            torch.zeros(M, dtype=torch.int32), dtype=ttnn.int32, device=mesh, memory_config=dram, mesh_mapper=rep
        )
        pt = page_table.reshape(1, -1).int()
        self.page_table = ttnn.from_torch(
            pt.repeat(M, 1), dtype=ttnn.int32, device=mesh, memory_config=dram, mesh_mapper=rep
        )
        self.rot = rope_rotation_tiles(mesh)
        self.cos_table, self.sin_table = model.rope_setup.cos_matrix, model.rope_setup.sin_matrix
        self.features = {}
        self.debug = None  # list: (tag, host [M, hidden]) after every boundary (debugging)

    def set_inputs(self, tokens, start):
        """Host -> device: the M block tokens and positions start .. start + M - 1."""
        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        t = ttnn.from_torch(
            torch.tensor([tokens], dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep
        )
        p = ttnn.from_torch(torch.arange(start, start + self.M, dtype=torch.int32), dtype=ttnn.int32, mesh_mapper=rep)
        ttnn.copy_host_to_device_tensor(t, self.tokens)
        ttnn.copy_host_to_device_tensor(p, self.pos)

    def forward(self):
        """The verify pass on the current inputs; leaves the argmax lists in self.lists and the captured layer outputs
        (row-tile residuals) in self.features."""
        M, b = self.M, self.boundary
        emb = ttnn.embedding(
            self.tokens, self.model.embedding_weight, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        emb = ttnn.reshape(emb, (1, 1, M, self.H))
        res, x = b(emb, self.gamma_in[0], entry=True)
        emb.deallocate(True)
        self._dbg("entry", res, x)
        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            k_cache, v_cache = self.kv_cache[i]
            rk = QKVRopeCache.rows(
                self.rot, self.cos_table, self.sin_table, self.pos, self.page_table, k_cache, v_cache
            )
            self.qkv(x, attn.weights.wqkv_stream, (self.q, self.q, self.q), rope_kv=rk)
            x.deallocate(True)
            sdpa = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                self.q,
                k_cache,
                v_cache,
                cur_pos_tensor=self.pos,
                sliding_window_size=attn.config.sliding_window,
                attention_sink=attn.weights.decode_sinks,
                page_table_tensor=self.page_table,
                scale=attn.config.scaling,
                program_config=sdpa_decode_program_config(attn.config.sliding_window),
                compute_kernel_config=attn.program_config.get_compute_kernel_config(),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            if self.debug is not None and i == 0:
                d0 = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
                self.debug.append(("sdpa0", d0(self.q), d0(sdpa)))
            self.o_proj(sdpa, attn.weights.o_proj_stream, self.partial)
            sdpa.deallocate(True)
            res2, x = b(res, self.gamma_post[i], partial=self.partial, site="attn")
            res.deallocate(True)
            res = res2
            self._dbg(f"attn{i}", res, x)
            mlp = layer.mlp
            self.router(x, mlp.router.weight_stream, (self.union, self.weights))
            dw = mlp.experts.decode_weights
            self.gate_up(x, self.union, self.weights, dw.gate_up_stream, self.act)
            x.deallocate(True)
            self.down(self.act, self.union, self.weights, dw.down_stream, self.partial)
            last = i + 1 == len(self.layers)
            res2, x = b(res, self.gamma_final if last else self.gamma_in[i + 1], partial=self.partial, site="moe")
            res.deallocate(True)
            res = res2
            self._dbg(f"moe{i}", res, x)
            if i in self.capture_layers:
                old = self.features.get(i)
                if old is not None:
                    old.deallocate(True)
                self.features[i] = ttnn.sharded_to_interleaved(res, ttnn.DRAM_MEMORY_CONFIG)
        res.deallocate(True)
        self.lm_head(x, self.lm_weights, self.lists)
        x.deallocate(True)

    def _dbg(self, tag, res, x):
        if self.debug is None:
            return
        rows = lambda t: rowtile_to_rows(ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()[0, 0], self.M, self.H)
        self.debug.append((tag, rows(res), rows(x)))

    def argmax(self):
        """Host: the target's argmax token for each of the M rows."""
        lists = [ttnn.to_torch(t).reshape(-1).long() & 0xFFFFFFFF for t in ttnn.get_device_tensors(self.lists)]
        return merge_argmax(lists, self.M, self.vd)

    def host_features(self, n):
        """Host: [n, 5 * hidden] drafter context features of the first n rows (layer order of capture_layers)."""
        out = []
        for L in self.capture_layers:
            t = ttnn.to_torch(ttnn.get_device_tensors(self.features[L])[0]).float()[0, 0]
            out.append(rowtile_to_rows(t, self.M, self.H)[:n])
        return torch.cat(out, dim=-1)
