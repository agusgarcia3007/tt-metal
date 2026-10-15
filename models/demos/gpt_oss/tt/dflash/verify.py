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

import os

import torch

import ttnn

from ..fused_decode.boundary import (
    DecodeBoundary,
    PendingBoundary,
    flat_memory_config,
    flat_shape,
    rowtile_pages,
    rowtile_to_rows,
)
from ..fused_decode.config import (
    GATE_UP_STREAM_READERS,
    LM_HEAD_DECODE_WEIGHT_DTYPE,
    LM_HEAD_STREAM_PREFETCH,
    OPROJ_DECODE_WEIGHT_DTYPE,
    QKV_DECODE_WEIGHT_DTYPE,
    QKV_STREAM_READERS,
    ROUTER_DECODE_WEIGHT_DTYPE,
    sdpa_decode_program_config,
)
from ..fused_decode.stream import LinearStream, QKVRopeCache, merge_argmax, read_lists, rope_rotation_tiles
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
        self.boundary = DecodeBoundary(
            mesh,
            H,
            cfg.rms_norm_eps,
            cluster_axis=1,
            rows=M,
            links=int(os.environ.get("DFLASH_BOUNDARY_LINKS", "1")),
            ccl=os.environ.get("DFLASH_CCL", "fabric"),
            # DFLASH_BOUNDARY_HELPERS=n: the multi-core boundary on n helper cores of grid row 2 (no streamed-op worker
            # sits there)
            helpers=[(x, 2) for x in range(int(os.environ.get("DFLASH_BOUNDARY_HELPERS", "4")))] or None,
        )
        b = self.boundary
        self.gamma_in = [b.flat_gamma(L.input_layernorm) for L in self.layers]
        self.gamma_post = [b.flat_gamma(L.post_attention_layernorm) for L in self.layers]
        self.gamma_final = b.flat_gamma(model.norm)

        self.qkv = LinearStream(
            mesh,
            H // TILE,
            (self.nq + 2 * self.nkv) * self.hd,
            QKV_DECODE_WEIGHT_DTYPE,
            readers=int(os.environ.get("DFLASH_QKV_READERS", QKV_STREAM_READERS)),
            out_mode=3,
            heads=(q_cols, k_cols),
            rows=M,
        )
        self.o_proj = LinearStream(
            mesh,
            self.nq * self.hd // TILE,
            H,
            OPROJ_DECODE_WEIGHT_DTYPE,
            readers=int(os.environ.get("DFLASH_OPROJ_READERS", "2")),  # 2: 4.81 vs 4.88 ms (1), 4.84 (3)
            x_pages=self.hd // TILE,
            out_mode=5,
            rows=M,
            gather_split=os.environ.get("DFLASH_OPROJ_SPLIT", "1") == "1",
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
        self.down = ExpertDownUnion(
            mesh,
            H,
            inter_pad,
            readers=int(os.environ.get("DFLASH_DOWN_READERS", "3")),
            rows=M,
            expert_outer=os.environ.get("DFLASH_DOWN_EO", "1") == "1",
        )
        # DFLASH_ROW_ATTN_MAX=n (> 0): row attention (dflash/row_attention.py) for positions < n (32-chunk cores)
        self.row_attn = None
        self.use_row_attn = False
        ra_max = int(os.environ.get("DFLASH_ROW_ATTN_MAX", "1024"))
        if ra_max:
            from .row_attention import RowAttention

            self.row_attn = RowAttention(
                mesh, M, self.nq, self.nkv, self.hd, 64, max_chunks=ra_max // TILE, cores_start=(0, 5)
            )
            self.row_attn_max = ra_max
            self.use_row_attn = True
        self.moe_fused = None
        if os.environ.get("DFLASH_MOE_FUSED", "1") == "1" and self.boundary.ccl == "fabric":
            from .moe_fused import MoEFused

            self.moe_fused = MoEFused(
                mesh,
                self.boundary,
                self.router,
                H,
                inter_pad,
                ex.config.swiglu_limit,
                ex.config.alpha,
                M,
                gate_up_readers=GATE_UP_STREAM_READERS,
                down_readers=int(os.environ.get("DFLASH_DOWN_READERS", "3")),
                real_inter=cfg.intermediate_size // tp if os.environ.get("DFLASH_SKIP_PAD", "0") == "1" else None,
            )
        term = model.decode_terminal
        self.vd = term.vd
        lm_readers = int(os.environ.get("DFLASH_LM_READERS", "2"))
        if lm_readers > 1:
            from .lm_head import padded_lm_head

            self.lm_weights, lm_n = padded_lm_head(mesh, term, lm_readers, os.environ.get("TT_CACHE_PATH"))
        else:
            self.lm_weights, lm_n = term.weights, term.vd
        self.lm_n = lm_n
        self.lm_head = LinearStream(
            mesh,
            H // TILE,
            lm_n,
            LM_HEAD_DECODE_WEIGHT_DTYPE,
            readers=lm_readers,
            out_mode=6,
            prefetch_cols=LM_HEAD_STREAM_PREFETCH,
            rows=M,
        )
        # DFLASH_LM_PRESCREEN=K (> 0): two-stage LM head (dflash/lm_rescore.py): the BFP4 head keeps each row's K
        # best columns per core, then the device's K best per row are rescored with the BF8 weights
        self.prescreen = int(os.environ.get("DFLASH_LM_PRESCREEN", "8"))
        if self.prescreen:
            from .lm_head import padded_lm_head
            from .lm_rescore import LMRescore

            r4 = int(os.environ.get("DFLASH_LM_PRESCREEN_READERS", "3"))
            self.lm_w8 = self.lm_weights
            self.lm_weights, n4 = padded_lm_head(mesh, term, r4, os.environ.get("TT_CACHE_PATH"), dtype=ttnn.bfloat4_b)
            # both streams keep the vocab's column order (only the padding past vd differs): same column indices
            assert n4 >= self.vd and lm_n >= self.vd
            self.lm_n4 = n4
            self.lm_head = LinearStream(
                mesh,
                H // TILE,
                n4,
                ttnn.bfloat4_b,
                readers=r4,
                out_mode=6,
                prefetch_cols=LM_HEAD_STREAM_PREFETCH,
                rows=M,
                out_page_bytes=8 * self.prescreen,  # the top-K lists: one page per (core, row)
            )
            self.rescore = LMRescore(mesh, M, self.prescreen, len(self.lm_head.assignments), H, lm_n)

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
        n_lists = self.prescreen or len(self.lm_head.assignments)
        if self.prescreen:  # the BFP4 stage's per-core top-K lists
            self.lists1 = ttnn.from_torch(
                torch.zeros(len(self.lm_head.assignments) * M, self.prescreen * 2, dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                memory_config=l1,
                mesh_mapper=rep,
            )
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
            memory_config=l1,
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
        self.fused = self.boundary.ccl == "fabric"  # producer-sent / consumer-fused boundaries need the fabric CCL
        self.debug = None  # list: (tag, host [M, hidden]) after every boundary (debugging)
        # (mask id, next-round input tensors): the device computes the round's acceptance and writes the next round's
        # inputs (dflash/argmax.py ACCEPT; FastDFlashRunner device rounds)
        self.accept = None
        # verified rows (dflash/moe_fused.py VERIFY_ROWS; 0: all): set per captured trace by FastDFlashRunner
        from .moe_fused import VERIFY_ROWS

        self.active_rows = VERIFY_ROWS

    def enable_device_argmax(self):
        """Merge the LM head's argmax lists on device (dflash/argmax.py): result = [block tokens, 8 argmaxes] (UINT32
        [1, 2 M], read by read_result)."""
        from .argmax import ArgmaxGather

        n_lists = self.prescreen or len(self.lm_head.assignments)
        self.argmax_gather = ArgmaxGather(self.mesh, self.M, n_lists, self.vd, kind=1)
        self.result = ttnn.from_torch(
            torch.zeros(1, 2 * self.M, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def read_result(self):
        """Host: (block tokens, target argmaxes) from device 0's result buffer."""
        r = ttnn.to_torch(ttnn.get_device_tensors(self.result)[0]).reshape(-1).long().tolist()
        return r[: self.M], r[self.M :]

    def _row_attn_for(self, start):
        # full-attention layers use row attention only while every row's positions fit its max_chunks
        if self.row_attn is not None:
            self.use_row_attn = start + self.M <= self.row_attn_max

    def set_positions(self, start):
        self._row_attn_for(start)
        p = ttnn.from_torch(
            torch.arange(start, start + self.M, dtype=torch.int32),
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        ttnn.copy_host_to_device_tensor(p, self.pos)

    def set_inputs(self, tokens, start):
        """Host -> device: the M block tokens and positions start .. start + M - 1."""
        self._row_attn_for(start)
        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        t = ttnn.from_torch(
            torch.tensor([tokens], dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep
        )
        p = ttnn.from_torch(torch.arange(start, start + self.M, dtype=torch.int32), dtype=ttnn.int32, mesh_mapper=rep)
        ttnn.copy_host_to_device_tensor(t, self.tokens)
        ttnn.copy_host_to_device_tensor(p, self.pos)

    def forward(self):
        """The verify pass on the current inputs; leaves the argmax lists in self.lists and the captured layer outputs
        (row-tile residuals) in self.features. fused (default): every boundary's all-reduce is sent by its producer
        (o_proj / down) and the boundary runs inside the op consuming its normed output (QKV, router, LM head), as in
        the one-token fused decode layer (fused_decode/boundary.py); else each boundary is an op of its own."""
        if not self.fused:
            return self._forward_unfused()
        M, b = self.M, self.boundary
        emb = ttnn.embedding(
            self.tokens, self.model.embedding_weight, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        emb = ttnn.reshape(emb, (1, 1, M, self.H))
        pending = PendingBoundary(b, emb, self.gamma_in[0], entry=True)
        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            k_cache, v_cache = self.kv_cache[i]
            rk = QKVRopeCache.rows(
                self.rot, self.cos_table, self.sin_table, self.pos, self.page_table, k_cache, v_cache
            )
            self.qkv(pending, attn.weights.wqkv_stream, (self.q, self.q, self.q), rope_kv=rk)
            res = pending.residual_out
            pending.x.deallocate(True)
            pending.residual.deallocate(True)
            if i - 1 in self.capture_layers:  # layer i - 1's output = this layer's input residual
                self._capture(i - 1, res)
            sdpa = self._sdpa(i)
            self.o_proj(sdpa, attn.weights.o_proj_stream, self.partial, send=(b, "attn"))
            if self.row_attn is None or sdpa is not self.row_attn.out:
                sdpa.deallocate(True)
            pending = PendingBoundary(b, res, self.gamma_post[i], self.partial, "attn", sent=True)
            mlp = layer.mlp
            dw = mlp.experts.decode_weights
            if self.moe_fused is not None:
                self.moe_fused(
                    pending,
                    mlp.router.weight_stream,
                    self.union,
                    self.weights,
                    dw.gate_up_stream,
                    self.act,
                    dw.down_stream,
                    self.partial,
                    layer=i,
                    active_rows=self.active_rows,
                )
                res = pending.residual_out
                pending.residual.deallocate(True)
                pending.x.deallocate(True)
                if getattr(self, "union_sets", None) is not None:  # debugging: the union's expert ids (eager)
                    u = ttnn.to_torch(ttnn.get_device_tensors(self.union)[0]).reshape(-1).tolist()
                    self.union_sets.append(u[1 : 1 + u[0]])
                last = i + 1 == len(self.layers)
                gamma = self.gamma_final if last else self.gamma_in[i + 1]
                pending = PendingBoundary(b, res, gamma, self.partial, "moe", sent=True)
                continue
            self.router(pending, mlp.router.weight_stream, (self.union, self.weights))
            res, x = pending.residual_out, pending.x
            pending.residual.deallocate(True)
            if getattr(self, "union_log", None) is not None:  # debugging: union size per layer (host read, eager)
                self.union_log.append(int(ttnn.to_torch(ttnn.get_device_tensors(self.union)[0]).reshape(-1)[0]))
            self.gate_up(x, self.union, self.weights, dw.gate_up_stream, self.act)
            x.deallocate(True)
            self.down(self.act, self.union, self.weights, dw.down_stream, self.partial, send=(b, "moe"))
            last = i + 1 == len(self.layers)
            gamma = self.gamma_final if last else self.gamma_in[i + 1]
            pending = PendingBoundary(b, res, gamma, self.partial, "moe", sent=True)
        if self.prescreen:
            self.lm_head(pending, self.lm_weights, self.lists1, extra_defines=[("LM_TOPK", str(self.prescreen))])
            self.rescore(self.lists1, pending.x, self.lm_w8, self.lists)
        else:
            self.lm_head(pending, self.lm_weights, self.lists)
        if getattr(self, "argmax_gather", None) is not None:
            self.argmax_gather(self.lists, self.tokens, self.result, accept=self.accept, rows_cap=self.active_rows)
        pending.x.deallocate(True)
        pending.residual.deallocate(True)
        pending.residual_out.deallocate(True)

    def set_context_target(self, ctx_input):
        """Capture straight into the drafter's context input ([1, 1, 32, 5 * hidden] BF16 tile tensor, L1
        interleaved, width-sharded over the mesh columns; DFlashDraft.t_ctx) instead of self.features."""
        self.ctx_input = ctx_input

    def _capture_to_ctx(self, L, res):
        from ..fused_decode.boundary import KERNEL_DIR

        slot = self.capture_layers.index(L)
        mesh, b, H, M = self.mesh, self.boundary, self.H, self.M
        shard = self.ctx_input.shape[-1]  # per-device width (a mesh tensor's shape is the device shard's)
        lo, hi = slot * H, (slot + 1) * H
        mesh_program = ttnn.MeshProgramDescriptor()
        ct = [M, H, lo] + ttnn.TensorAccessorArgs(self.ctx_input).get_compile_time_args()
        for coord in b.coords:
            d = coord[1]
            c0, c1 = max(lo, d * shard), min(hi, (d + 1) * shard)
            rt = ttnn.RuntimeArgs()
            rt[b.core.x][b.core.y] = [res.buffer_address(), self.ctx_input.buffer_address(), c0, max(c0, c1), d * shard]
            kernel = ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "rowtile_to_ctx.cpp"),
                core_ranges=b.cores,
                compile_time_args=ct,
                runtime_args=rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0
                ),
            )
            mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = ttnn.ProgramDescriptor(
                kernels=[kernel], semaphores=[], cbs=[]
            )
        ttnn.generic_op([res, self.ctx_input], mesh_program)

    def set_context_target_rowtile(self, buf, split):
        """Capture into a per-device row-tile buffer: device d receives global context tiles split[d] = [s, e) of the
        layer-major 5 x hidden features (whole [M, 32] tiles, one L1 copy per layer and device)."""
        self.ctx_rowtile = (buf, split)

    def _capture_rowtile(self, L, res):
        from ..fused_decode.boundary import KERNEL_DIR

        buf, split = self.ctx_rowtile
        slot = self.capture_layers.index(L)
        b, kt = self.boundary, self.H // TILE
        tb = self.M * 64
        lo, hi = slot * kt, (slot + 1) * kt
        mesh_program = ttnn.MeshProgramDescriptor()
        for coord in b.coords:
            s_, e = split[coord[1]]
            a, z = max(lo, s_), min(hi, e)
            n = max(0, z - a)
            rt = ttnn.RuntimeArgs()
            if b.mc:
                # the residual is sharded over the helpers (b.PH pages each): each copies its part of the range
                shard = b.PH * 2048
                src0, src1 = (a - lo) * tb, (a - lo) * tb + n * tb
                for h, c in enumerate(b.helpers):
                    u, v = max(src0, h * shard), min(src1, (h + 1) * shard)
                    rt[c.x][c.y] = [
                        res.buffer_address() + (u - h * shard) if v > u else 0,
                        buf.buffer_address() + (a - s_) * tb + (u - src0) if v > u else 0,
                        max(0, v - u),
                    ]
                cores = b.helper_cores
            else:
                rt[b.core.x][b.core.y] = [
                    res.buffer_address() + (a - lo) * tb if n else 0,
                    buf.buffer_address() + (a - s_) * tb if n else 0,
                    n * tb,
                ]
                cores = b.cores
            kernel = ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "l1_copy.cpp"),
                core_ranges=cores,
                compile_time_args=[1 if buf.memory_config().buffer_type == ttnn.BufferType.DRAM else 0],
                runtime_args=rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0
                ),
            )
            mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = ttnn.ProgramDescriptor(
                kernels=[kernel], semaphores=[], cbs=[]
            )
        ttnn.generic_op([res, buf], mesh_program)

    def _capture(self, L, res):
        if getattr(self, "ctx_rowtile", None) is not None:
            return self._capture_rowtile(L, res)
        if getattr(self, "ctx_input", None) is not None:
            return self._capture_to_ctx(L, res)
        old = self.features.get(L)
        if old is not None:
            old.deallocate(True)
        self.features[L] = ttnn.sharded_to_interleaved(res, ttnn.DRAM_MEMORY_CONFIG)

    def _sdpa_pc(self, sliding_window):
        # DFLASH_SDPA_GRID="x,y" / DFLASH_SDPA_K=n: verify SDPA program config override (short-context tuning)
        grid, k = os.environ.get("DFLASH_SDPA_GRID"), os.environ.get("DFLASH_SDPA_K")
        if not grid and not k:
            return sdpa_decode_program_config(sliding_window)
        base = sdpa_decode_program_config(sliding_window)
        gx, gy = (int(v) for v in grid.split(",")) if grid else (8, 8)
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
            q_chunk_size=0,
            k_chunk_size=int(k) if k else base.k_chunk_size,
            exp_approx_mode=False,
        )

    def _sdpa(self, i):
        attn = self.layers[i].self_attn
        k_cache, v_cache = self.kv_cache[i]
        if self.row_attn is not None and (self.use_row_attn or attn.config.sliding_window):
            # short-context row attention (dflash/row_attention.py); the output buffer is reused by every layer (o_proj
            # consumes it before the next layer's attention)
            return self.row_attn(
                self.q,
                k_cache,
                v_cache,
                self.pos,
                self.page_table,
                attn.weights.decode_sinks,
                attn.config.scaling,
                window=attn.config.sliding_window or 0,
            )
        return ttnn.transformer.paged_scaled_dot_product_attention_decode(
            self.q,
            k_cache,
            v_cache,
            cur_pos_tensor=self.pos,
            sliding_window_size=attn.config.sliding_window,
            attention_sink=attn.weights.decode_sinks,
            page_table_tensor=self.page_table,
            scale=attn.config.scaling,
            program_config=self._sdpa_pc(attn.config.sliding_window),
            compute_kernel_config=attn.program_config.get_compute_kernel_config(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _forward_unfused(self):
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
            sdpa = self._sdpa(i)
            if self.debug is not None and i == 0:
                d0 = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
                self.debug.append(("sdpa0", d0(self.q), d0(sdpa)))
            self.o_proj(sdpa, attn.weights.o_proj_stream, self.partial)
            if self.row_attn is None or sdpa is not self.row_attn.out:
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
                self._capture(i, res)
        res.deallocate(True)
        self.lm_head(x, self.lm_weights, self.lists)
        if getattr(self, "argmax_gather", None) is not None:
            self.argmax_gather(self.lists, self.tokens, self.result, accept=self.accept)
        x.deallocate(True)

    def _dbg(self, tag, res, x):
        if self.debug is None:
            return
        rows = lambda t: rowtile_to_rows(ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()[0, 0], self.M, self.H)
        self.debug.append((tag, rows(res), rows(x)))

    def argmax(self):
        """Host: the target's argmax token for each of the M rows."""
        return merge_argmax(read_lists(self.lists, self.mesh), self.M, self.vd)

    def host_features(self, n):
        """Host: [n, 5 * hidden] drafter context features of the first n rows (layer order of capture_layers)."""
        out = []
        for L in self.capture_layers:
            t = ttnn.to_torch(ttnn.get_device_tensors(self.features[L])[0]).float()[0, 0]
            out.append(rowtile_to_rows(t, self.M, self.H)[:n])
        return torch.cat(out, dim=-1)
