# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Multi-row decode MoE over the union of the rows' routed experts (DFlash verify: M token rows of one user).

The multi-row router (LinearStream rows = M, out_mode 4) writes
  * union: UINT16 [1, 64] = [U, the U union expert ids ascending],
  * weights: BF16 [1, 32 M], row r's softmax weight of union expert i at entry 32 r + i (0 when row r did not route
    to it).
ExpertGateUpUnion streams the gate|up columns of the U union experts (the ExpertGateUpStream weight layout) for all M
rows at once ([M, 32] activation tiles), applies the SwiGLU and scales row r by w[r][i]; ExpertDownUnion streams the
down columns of the same experts and sums over them (K = U segments [w-scaled act_i | w[:, i] bias multipliers]) into
the row-tile partial sum the boundary all-reduces. Every union expert's weights are read once for the M rows.
"""

import torch

import ttnn

from .stream import (
    MAX_PACKET_BYTES,
    NBIAS,
    TILE,
    TILE_BYTES,
    TINY_BYTES,
    _BankStreamOp,
    _f32_bits,
    _page_bytes,
    columns_per_bank,
)

MAX_EXPERTS = 32


def union_buffers(mesh_device, rows):
    """(union, weights) router outputs (L1 interleaved, replicated)."""
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    uni = ttnn.from_torch(
        torch.zeros(1, 64, dtype=torch.int32),
        dtype=ttnn.uint16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    w = ttnn.from_torch(
        torch.zeros(1, 32 * rows),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    return uni, w


def act_buffer(mesh_device, inter_pad, rows, slots=MAX_EXPERTS, memory_config=ttnn.L1_MEMORY_CONFIG):
    """[slots, I_pad / 32 [rows, 32] tiles] BF16 activation buffer (one row-major page per slot), L1 interleaved."""
    return ttnn.from_torch(
        torch.zeros(slots, inter_pad * rows),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


_IDENTITY = {}


def bfp4_identity(mesh_device, dtype=ttnn.bfloat4_b):
    """The 32x32 identity as one tile of `dtype` (exact: entries 0 / 1), replicated in DRAM."""
    key = (id(mesh_device), dtype)
    cached = _IDENTITY.get(key)
    if cached is None or cached[0] is not mesh_device:
        t = ttnn.from_torch(
            torch.eye(TILE)[None, None],
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        cached = _IDENTITY[key] = (mesh_device, t)
    return cached[1]


class ExpertGateUpUnion(_BankStreamOp):
    """(x, union, weights, gate_up_stream, act) -> act. x: the boundary's normed hidden in the row-tile layout (M rows);
    act (act_buffer): slot i = expert union[i]'s SwiGLU rows scaled by w[r][i]."""

    def __init__(
        self,
        mesh_device,
        hidden,
        inter_pad,
        swiglu_limit,
        alpha,
        readers,
        rows,
        dense_silu=False,
        max_experts=MAX_EXPERTS,
        weight_dtype=ttnn.bfloat4_b,
    ):
        """dense_silu: plain silu(g) * u (a dense MLP run as one union expert, e.g. the DFlash drafter)."""
        super().__init__(mesh_device, readers)
        self.dense_silu = dense_silu
        self.max_experts = max_experts
        self.rows = rows
        self.kx = hidden // TILE
        self.kt = self.kx + 1
        self.inter_pad = inter_pad
        self.P = inter_pad // TILE // self.banks
        assert self.P % readers == 0, f"{self.P} pairs per bank do not split over {readers} readers"
        self.pairs = self.P // readers
        self.limit, self.alpha = swiglu_limit, alpha
        self.weight_dtype = weight_dtype
        self.tile_bytes = TILE_BYTES[weight_dtype]
        self.col_bytes = self.kt * self.tile_bytes
        self.expert_stride = 2 * self.P * self.col_bytes
        self.page_bytes = _page_bytes(self.kt, self.tile_bytes)

    def weight_rows(self, num_experts):
        return num_experts * 2 * self.P * self.kt * TILE

    def __call__(self, x, union, weights, w_stream, act):
        M = self.rows
        cb_x, cb_w, cb_idx, cb_scr, cb_sc, cb_id, cb_act = 0, 1, 2, 3, 4, 5, 16
        tiny = TINY_BYTES * M
        cbs = [
            self._cb(cb_x, ttnn.bfloat16, tiny, self.kt, tiny=True, rows=M),
            self._cb(cb_w, self.weight_dtype, self.tile_bytes, 3 * self.kt),
            self._cb(cb_idx, ttnn.uint16, 128, 1),
            self._cb(cb_scr, ttnn.bfloat16, 128 + 64 * M, 1),
            self._cb(cb_sc, ttnn.bfloat16, tiny, self.max_experts, tiny=True, rows=M),
            self._cb(cb_id, self.weight_dtype, self.tile_bytes, 1),
            self._cb(cb_act, ttnn.bfloat16, tiny, 4, tiny=True, rows=M),
        ]
        ident = bfp4_identity(self.mesh_device, self.weight_dtype)
        reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for core, bank, r, vc in self.assignments:
            reader_rt[core.x][core.y] = [
                w_stream.buffer_address(),
                union.buffer_address(),
                bank,
                vc,
                r * 2 * self.pairs * self.col_bytes,
            ]
            writer_rt[core.x][core.y] = [
                x.buffer_address(),
                act.buffer_address(),
                bank * self.P + r * self.pairs,
                weights.buffer_address(),
                union.buffer_address(),
                ident.buffer_address(),
            ]
        reader_ct = [cb_w, cb_idx, self.kt, 2 * self.pairs, self.expert_stride, self.tile_bytes, self.page_bytes]
        reader_ct += ttnn.TensorAccessorArgs(union).get_compile_time_args()
        act_page = self.inter_pad // TILE * tiny
        writer_ct = [cb_x, cb_act, self.kx, NBIAS, self.pairs, M, act_page, cb_sc, cb_id, cb_scr, self.tile_bytes]
        for t in (x, act, weights, union, ident):
            writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        compute_ct = [cb_x, cb_w, cb_act, self.kt, self.pairs]
        compute_ct += [_f32_bits(v) for v in (self.limit, -self.limit, -3.0e38, self.alpha, 1.0 / self.alpha, 1.0)]
        compute_ct += [cb_sc, cb_id]
        program = self._program(
            "gate_up_union",
            cbs,
            reader_ct,
            reader_rt,
            writer_ct,
            writer_rt,
            compute_ct,
            "gate_up_union",
            defines=[("ROWS", str(M))] + ([("DENSE_SILU", "1")] if self.dense_silu else []),
        )
        return ttnn.generic_op([w_stream, x, union, weights, ident, act], program)


class ExpertDownUnion(_BankStreamOp):
    """(act, union, weights, down_stream, out) -> out: the row-tile partial sum (M rows) of
    sum_i w[r][i] * (act_i @ Wd_i + bd_i) over the union experts (the ExpertDownStream weight layout)."""

    def __init__(
        self,
        mesh_device,
        hidden,
        inter_pad,
        readers,
        rows,
        chunk=2,
        expert_outer=False,
        max_experts=MAX_EXPERTS,
        weight_dtype=ttnn.bfloat4_b,
        num_buffers=3,
    ):
        """num_buffers: weight blocks in the reader's ring (expert-outer; 2 when the blocks are large).
        expert_outer: kernels/stream_down_union2_*: each core reads its columns of one expert as one contiguous
        block and accumulates its (<= 4) output tiles across the experts in DST (larger DRAM reads)."""
        super().__init__(mesh_device, readers)
        self.rows = rows
        self.hidden = hidden
        self.inter_pad = inter_pad
        self.seg_tiles = inter_pad // TILE + 1
        self.C = columns_per_bank(hidden, self.banks)
        assert self.C % readers == 0, f"{self.C} columns per bank do not split over {readers} readers"
        self.cols = self.C // readers
        self.weight_dtype = weight_dtype
        self.tile_bytes = TILE_BYTES[weight_dtype]
        self.seg_bytes = self.seg_tiles * self.tile_bytes
        # expert-outer reads a core's columns of one expert (contiguous) in packets of <= 16 KB
        self.page_bytes = _page_bytes(self.seg_tiles, self.tile_bytes)
        assert expert_outer or self.seg_bytes <= MAX_PACKET_BYTES, "one expert column segment must fit one NOC packet"
        self.expert_stride = self.C * self.seg_bytes
        self.chunk = chunk
        self.max_experts = max_experts
        self.expert_outer = expert_outer
        self.num_buffers = num_buffers
        assert not expert_outer or self.cols <= 4, "expert-outer down keeps a core's output tiles in DST"

    def weight_rows(self, num_experts):
        return num_experts * self.C * self.seg_tiles * TILE

    def __call__(self, act, union, weights, w_stream, out, send=None):
        """send = (boundary, site): fuse the boundary all-reduce's fabric send (DecodeBoundary.sending_program)."""
        M = self.rows
        cb_in0, cb_w, cb_idx, cb_scr, cb_out = 0, 1, 2, 3, 16
        tiny = TINY_BYTES * M
        block = (self.cols if self.expert_outer else self.chunk) * self.seg_tiles
        cbs = [
            self._cb(cb_in0, ttnn.bfloat16, tiny, self.max_experts * self.seg_tiles, tiny=True, rows=M),
            self._cb(cb_w, self.weight_dtype, self.tile_bytes, self.num_buffers * block),
            self._cb(cb_idx, ttnn.uint16, 128, 1),
            self._cb(cb_scr, ttnn.bfloat16, 128 + 64 * M, 1),
            self._cb(cb_out, ttnn.bfloat16, tiny, 4, tiny=True, rows=M),
        ]
        reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for core, bank, r, vc in self.assignments:
            reader_rt[core.x][core.y] = [
                w_stream.buffer_address(),
                union.buffer_address(),
                bank,
                vc,
                r * self.cols * self.seg_bytes,
            ]
            writer_rt[core.x][core.y] = [
                act.buffer_address(),
                weights.buffer_address(),
                out.buffer_address(),
                bank * self.C + r * self.cols,
                union.buffer_address(),
            ]
        if self.expert_outer:
            reader_ct = [cb_w, cb_idx, self.seg_tiles, self.cols, self.expert_stride, self.tile_bytes, self.page_bytes]
            reader_ct.append(self.num_buffers)
        else:
            reader_ct = [cb_w, cb_idx, self.seg_tiles, self.cols, self.chunk, self.expert_stride, self.tile_bytes]
        reader_ct += ttnn.TensorAccessorArgs(union).get_compile_time_args()
        act_page = self.inter_pad // TILE * tiny
        writer_ct = [cb_in0, cb_out, cb_scr, self.seg_tiles, M, NBIAS, self.cols, self.hidden // TILE, act_page]
        for t in (act, weights, out, union):
            writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        writer_ct += send[0].notify_args() if send else [0, 0, 0, 0]
        compute_ct = [cb_in0, cb_w, cb_out, self.seg_tiles, self.cols]
        if not self.expert_outer:
            compute_ct.append(self.chunk)
        name = "down_union2" if self.expert_outer else "down_union"

        def build():
            return self._program(name, cbs, reader_ct, reader_rt, writer_ct, writer_rt, compute_ct, name)

        program = send[0].sending_program(build, out, send[1], self.cores) if send else build()
        return ttnn.generic_op([w_stream, act, union, weights, out], program)
