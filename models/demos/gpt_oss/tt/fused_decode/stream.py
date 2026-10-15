# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DRAM-streaming decode matmuls of the fused gpt-oss decode layer (one token per device, TP shards).

At decode a projection reads its whole weight to produce one row. ttnn's matmuls carry that row in 32-row tiles
and read the weights tile by tile (the indexed sparse_matmul reaches about 220 GB/s on the 1x4 Blackhole mesh).
These ttnn.generic_op ops (kernels/stream_*.cpp) stream the weights per DRAM bank instead:

* Layout: a weight is stored width-sharded over the DRAM banks, bank b holding a fixed set of output columns,
  column-major (all K tiles of a column contiguous), so a worker core reads its columns as one contiguous byte range
  in large NOC packets (transaction-id pipelined, one column in flight ahead of compute). Expert weights are stored
  expert-major, so the columns of one routed expert are contiguous too.
* Bias: every column carries one extra K tile whose first NBIAS rows hold the BF16 bias as a sum of residual terms
  rounded to the weight format (q1 = q(b), q2 = q(b - q1), ...); the activation's extra 1x32 tile holds the matching
  multipliers (ones, or the routing score for the expert down projection), so the matmul adds the bias.
* Compute: worker cores next to each DRAM bank run custom_mm with 1x32 activation tiles (LoFi, FP32 accumulation).
  The activation is the flat normed hidden of the layer boundary (fused_decode/boundary.py: value h at byte 2 h, so its
  32-value groups are the 1x32 tiles, read in one piece), or for o_proj the SDPA output gathered head by head.

Ops: LinearStream (dense projection + bias: QKV straight into the head-split layout, o_proj straight into the flat
partial sum the boundary all-reduces, router + top-k + softmax), ExpertGateUpStream (routed gate|up + bias + SwiGLU,
scaled by the routing score) and ExpertDownStream (routed down + bias, summed over the routed experts, into the flat
partial sum).
"""

import os
import struct
from pathlib import Path

import torch

import ttnn

from .config import DECODE_BOUNDARY_PREFETCH_ALL

KERNEL_DIR = Path(__file__).parent / "kernels"
NBIAS = 4  # residual terms carrying a bias (BFP4: relative error ~ 8^-NBIAS of the 16-value group maximum)
TILE = ttnn.TILE_SIZE
BFP4_TILE_BYTES = 576
BFP8_TILE_BYTES = 1088
TILE_BYTES = {ttnn.bfloat4_b: BFP4_TILE_BYTES, ttnn.bfloat8_b: BFP8_TILE_BYTES, ttnn.bfloat16: 2048}
TINY_BYTES = 2 * TILE  # one 1x32 BF16 tile
MAX_PACKET_BYTES = 16384  # Blackhole NOC max burst


def _f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def _bfp4_round(tiles):
    """Round [n, 32, 32] tiles exactly as the device stores them in BFP4 (host conversion)."""
    t = ttnn.from_torch(tiles.float(), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)
    return ttnn.to_torch(t).float().reshape(tiles.shape)


def bias_tiles(bias):
    """[..., 32] BF16 bias rows -> [..., 32, 32] tiles whose first NBIAS rows sum (in BFP4) to the bias."""
    lead = bias.shape[:-1]
    b = bias.reshape(-1, TILE).float()
    out = torch.zeros(b.shape[0], TILE, TILE)
    r = b
    for i in range(NBIAS):
        t = torch.zeros(b.shape[0], TILE, TILE)
        t[:, 0] = r
        q = _bfp4_round(t)[:, 0]
        out[:, i] = q
        r = r - q
    return out.reshape(*lead, TILE, TILE)


def _columns_with_bias(w, bias):
    """w [E, K, N], bias [E, N] -> [E, n_tiles, k_tiles + 1, 32, 32] tile columns (bias tile last)."""
    E, K, N = w.shape
    n_tiles, k_tiles = N // TILE, K // TILE
    cols = w.reshape(E, k_tiles, TILE, n_tiles, TILE).permute(0, 3, 1, 2, 4)
    b = bias_tiles(bias.reshape(E, n_tiles, TILE)).to(cols.dtype)
    return torch.cat([cols, b.unsqueeze(2)], dim=2)


def _bank_columns(cols, banks):
    """[E, C * banks, k, 32, 32] -> [rows, banks * 32]: bank b's shard = its C columns as one column of tiles."""
    C = cols.shape[1] // banks
    return torch.cat([cols[:, b * C : (b + 1) * C].reshape(-1, TILE) for b in range(banks)], dim=-1)


def columns_per_bank(n, banks):
    return -(-n // TILE // banks)


def stream_gate_up_layout(gate_up, gate_up_bias, banks):
    """One TP shard of [E, hidden, 2 * I_pad] packed [gate | up] weights and [E, 2 * I_pad] biases -> [rows, banks * 32]:
    bank b's shard is a column of tiles ordered expert -> (gate j, up j) for the P pairs j in [b * P, (b + 1) * P)
    -> K tile (hidden / 32 weight tiles + 1 bias tile)."""
    E, H, W = gate_up.shape
    half = W // TILE // 2
    P = half // banks
    assert P * banks == half, f"{half} column pairs do not split over {banks} banks"
    cols = _columns_with_bias(gate_up, gate_up_bias)
    order = [c for bank in range(banks) for j in range(bank * P, (bank + 1) * P) for c in (j, half + j)]
    return _bank_columns(cols[:, order], banks)


def stream_down_layout(down, down_bias, banks):
    """One TP shard of [E, I_pad, hidden] down weights and [E, hidden] biases -> [rows, banks * 32]: bank b holds output
    columns [b * C, (b + 1) * C) (C = ceil(hidden / 32 / banks), zero padded), ordered expert -> column -> K tile
    (I_pad / 32 weight tiles + 1 bias tile)."""
    E, K, N = down.shape
    pad = columns_per_bank(N, banks) * banks * TILE - N
    cols = _columns_with_bias(torch.nn.functional.pad(down, (0, pad)), torch.nn.functional.pad(down_bias, (0, pad)))
    return _bank_columns(cols, banks)


def stream_linear_layout(w, bias, banks, order=None):
    """[K, N] weights and [N] bias -> [rows, banks * 32]: bank b holds output columns [b * C, (b + 1) * C)
    (C = ceil(N / 32 / banks), zero padded), ordered column -> K tile (K / 32 weight tiles + 1 bias tile). order:
    the output column of each slot (bank b's slots are [b * C, (b + 1) * C)), default the identity."""
    K, N = w.shape
    pad = columns_per_bank(N, banks) * banks * TILE - N
    cols = _columns_with_bias(
        torch.nn.functional.pad(w, (0, pad)).unsqueeze(0), torch.nn.functional.pad(bias, (0, pad)).unsqueeze(0)
    )
    if order is not None:
        cols = cols[:, order]
    return _bank_columns(cols, banks)


def qkv_column_order(n_tiles, q_cols, banks):
    """Slot order of the QKV stream with fused RoPE + KV-cache write (QKVRopeCache): the K / V columns (>= q_cols) go
    first on each bank, round-robin, then the Q columns, then the bank padding. A K / V column's cache update (BFP8
    conversion + DRAM write by the writer core) then overlaps the bank's later Q columns, and no bank ends the op on
    several of them (in column order one bank held all of V)."""
    C = -(-n_tiles // banks)
    slots = [[] for _ in range(banks)]
    for i, n in enumerate(range(q_cols, n_tiles)):
        slots[i % banks].append(n)
    q = iter(range(q_cols))
    pad = iter(range(n_tiles, C * banks))
    for sl in slots:
        while len(sl) < C:
            n = next(q, None)
            sl.append(n if n is not None else next(pad))
    assert all(len(sl) == C for sl in slots) and next(q, None) is None
    return [n for sl in slots for n in sl]


def linear_stream_rows(mesh_device, k, n):
    return columns_per_bank(n, mesh_device.dram_grid_size().x) * (k // TILE + 1) * TILE


def stream_weight_memory_config(mesh_device, rows):
    banks = mesh_device.dram_grid_size().x
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))}),
            (rows, TILE),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


def as_stream_tensor(mesh_device, layouts, rows, dtype, cache_file_name):
    """Per-device streamed layouts ([rows, banks * 32] each, one per mesh column / TP shard, or None to load from the
    cache) -> one DRAM width-sharded tensor per device. The cached file keeps the shard layout, so cache names carry
    the layout parameters."""
    t = torch.stack(layouts).unsqueeze(0) if layouts is not None else None
    return ttnn.as_tensor(
        t,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=(None, 1)),
        cache_file_name=cache_file_name,
        memory_config=stream_weight_memory_config(mesh_device, rows),
    )


def _page_bytes(tiles, tile_bytes):
    """Largest NOC packet holding whole tiles that divides a `tiles`-tile block."""
    return max(d for d in range(1, tiles + 1) if tiles % d == 0 and d * tile_bytes <= MAX_PACKET_BYTES) * tile_bytes


class _BankStreamOp:
    """Worker placement shared by the streaming ops: `readers` cores per DRAM bank (the first `active_banks` banks),
    starting at the bank's optimal NOC0 reader core and growing along its row, each with a NOC virtual channel
    distinct from earlier cores on the same row (as the DeepSeek DRAM-streaming matmul assigns them). The cores are
    coalesced into rectangles: dispatch writes kernel binaries and arguments per core range, and single-core ranges
    made multi-reader programs measurably slower to launch under trace."""

    def __init__(self, mesh_device, readers, active_banks=None, exclude=(), coord=None):
        """exclude: logical (x, y) cores to leave free (a fused program's other kernels); a bank's readers then also
        spill onto the neighbouring rows when its own row is full. coord: place for that device's own optimal bank
        workers (harvesting differs per chip) instead of the first device's."""
        self.mesh_device = mesh_device
        self.banks = mesh_device.dram_grid_size().x
        self.readers = readers
        if coord is not None:
            a = mesh_device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0, coord)
            bank_workers = [a[b] for b in sorted(a)]
        else:
            bank_workers = mesh_device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
        shift = int(os.environ.get("DFLASH_BANK_X_SHIFT", "0")) if exclude else 0  # experiment: fused MoE only
        if shift:
            bank_workers = [ttnn.CoreCoord(c.x + shift, c.y) if c.x > 3 else c for c in bank_workers]
        assert len(bank_workers) == self.banks
        bank_workers = bank_workers[: active_banks or self.banks]
        grid = mesh_device.compute_with_storage_grid_size()
        exclude = {tuple(c) for c in exclude}
        used = {(c.x, c.y) for c in bank_workers} | exclude
        placed = []
        for bank, core in enumerate(bank_workers):
            cores = [] if (core.x, core.y) in exclude else [core]
            # along the row (+dx before -dx); with exclusions, nearest first (Manhattan) over the rows within 2
            cand = [(dx, 0) for d in range(1, grid.x) for dx in (d, -d)]
            if exclude:
                cand = sorted(
                    ((dx, dy) for dx in range(-grid.x, grid.x) for dy in range(-2, 3)),
                    key=lambda p: (abs(p[0]) + abs(p[1]), abs(p[1]), abs(p[0]), p[0] < 0, p[1] < 0),
                )
            for dx, dy in cand:
                x, y = core.x + dx, core.y + dy
                if len(cores) < readers and 0 <= x < grid.x and 0 <= y < grid.y and (x, y) not in used:
                    cores.append(ttnn.CoreCoord(x, y))
                    used.add((x, y))
            assert len(cores) == readers, f"no room for {readers} readers next to bank {bank} worker {core}"
            placed += [(c, bank, r) for r, c in enumerate(cores)]
        vcs = []
        for i, (core, bank, r) in enumerate(placed):
            vc = (bank + r) & 0x3
            for j in range(i):
                if placed[j][0].y == core.y and vcs[j] == vc:
                    vc = (vc + 1) & 0x3
            vcs.append(vc)
        self.assignments = [(c, b, r, v) for (c, b, r), v in zip(placed, vcs)]
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(self.assignments[0][0], self.assignments[0][0])])
        for c, *_ in self.assignments[1:]:
            self.cores = self.cores.merge(ttnn.CoreRangeSet([ttnn.CoreRange(c, c)]))

    def _cb(self, index, dtype, page, pages, tiny=False, rows=1):
        """tiny: [rows, 32] tiles (page = 64 rows bytes for BF16)."""
        kwargs = dict(buffer_index=index, data_format=dtype, page_size=page)
        if tiny:
            kwargs["tile"] = ttnn.TileDescriptor(ttnn.Tile([rows, TILE]))
        return ttnn.CBDescriptor(
            total_size=page * pages, core_ranges=self.cores, format_descriptors=[ttnn.CBFormatDescriptor(**kwargs)]
        )

    def _program(
        self,
        name,
        cbs,
        reader_ct,
        reader_rt,
        writer_ct,
        writer_rt,
        compute_ct,
        compute="matmul",
        compute_rt=None,
        defines=None,
    ):
        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / f"stream_{name}_reader.cpp"),
                core_ranges=self.cores,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
                defines=defines or [],
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / f"stream_{name}_writer.cpp"),
                core_ranges=self.cores,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
                defines=defines or [],
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / f"stream_{compute}_compute.cpp"),
                core_ranges=self.cores,
                compile_time_args=compute_ct,
                runtime_args=compute_rt if compute_rt is not None else [],
                defines=defines or [],
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False
                ),
            ),
        ]
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


class ExpertGateUpStream(_BankStreamOp):
    """(x, ids, scores, weights, act) -> act. x: the flat BF16 normed hidden (fused_decode/boundary.py); ids / scores:
    UINT16 / BF16 row-major buffers whose first k entries are the routed experts and their softmax weights; act: [k,
    I_pad] BF16 row-major L1, receives w_e * SwiGLU(gate_e, up_e) per routed expert e."""

    def __init__(self, mesh_device, hidden, inter_pad, num_sel, swiglu_limit, alpha, readers):
        super().__init__(mesh_device, readers)
        self.kx = hidden // TILE
        self.kt = self.kx + 1
        self.num_sel = num_sel
        self.inter_pad = inter_pad
        self.P = inter_pad // TILE // self.banks
        assert self.P % readers == 0, f"{self.P} pairs per bank do not split over {readers} readers"
        self.pairs = self.P // readers
        self.limit, self.alpha = swiglu_limit, alpha
        self.col_bytes = self.kt * BFP4_TILE_BYTES
        self.expert_stride = 2 * self.P * self.col_bytes
        self.page_bytes = _page_bytes(self.kt, BFP4_TILE_BYTES)

    def weight_rows(self, num_experts):
        return num_experts * 2 * self.P * self.kt * TILE

    def __call__(self, x, ids, scores, weights, act):
        cb_x, cb_w, cb_idx, cb_scr, cb_act = 0, 1, 2, 3, 16
        cbs = [
            self._cb(cb_x, ttnn.bfloat16, TINY_BYTES, self.kt, tiny=True),
            self._cb(cb_w, ttnn.bfloat4_b, BFP4_TILE_BYTES, 3 * self.kt),
            self._cb(cb_idx, ttnn.uint16, 64, 1),
            self._cb(cb_scr, ttnn.bfloat16, 64, 1),
            self._cb(cb_act, ttnn.bfloat16, TINY_BYTES, 4, tiny=True),
        ]
        reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for core, bank, r, vc in self.assignments:
            reader_rt[core.x][core.y] = [
                weights.buffer_address(),
                ids.buffer_address(),
                bank,
                vc,
                r * 2 * self.pairs * self.col_bytes,
            ]
            writer_rt[core.x][core.y] = [
                x.buffer_address(),
                act.buffer_address(),
                bank * self.P + r * self.pairs,
                scores.buffer_address(),
            ]
        reader_ct = [cb_w, cb_idx, self.kt, 2 * self.pairs, self.num_sel, self.expert_stride, BFP4_TILE_BYTES]
        reader_ct += [self.page_bytes, 64] + ttnn.TensorAccessorArgs(ids).get_compile_time_args()
        writer_ct = [
            cb_x,
            cb_act,
            self.kx,
            NBIAS,
            self.pairs,
            self.num_sel,
            self.inter_pad * 2,
            TILE * TILE * 2,
            cb_scr,
        ]
        for t in (x, act, scores):
            writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        compute_ct = [cb_x, cb_w, cb_act, self.kt, self.pairs, self.num_sel]
        compute_ct += [_f32_bits(v) for v in (self.limit, -self.limit, -3.0e38, self.alpha, 1.0 / self.alpha, 1.0)]
        program = self._program("gate_up", cbs, reader_ct, reader_rt, writer_ct, writer_rt, compute_ct, "gate_up")
        return ttnn.generic_op([weights, x, ids, scores, act], program)


class ExpertDownStream(_BankStreamOp):
    """(act, ids, scores, weights, out) -> out. act: [k, I_pad] BF16 row-major (score-weighted SwiGLU outputs);
    out: the flat BF16 partial sum (fused_decode/boundary.py: value h at byte 2 h) receives
    sum_e w_e * (act_e @ Wd_e + bd_e) (one matmul per output column over the K = k segments [w_e * act_e | w_e] of
    the routed experts)."""

    def __init__(self, mesh_device, hidden, inter_pad, num_sel, readers):
        super().__init__(mesh_device, readers)
        self.hidden = hidden
        self.num_sel = num_sel
        self.inter_pad = inter_pad
        self.seg_tiles = inter_pad // TILE + 1
        self.C = columns_per_bank(hidden, self.banks)
        assert self.C % readers == 0, f"{self.C} columns per bank do not split over {readers} readers"
        self.cols = self.C // readers
        self.seg_bytes = self.seg_tiles * BFP4_TILE_BYTES
        assert self.seg_bytes <= MAX_PACKET_BYTES, "one expert column segment must fit one NOC packet"
        self.expert_stride = self.C * self.seg_bytes

    def weight_rows(self, num_experts):
        return num_experts * self.C * self.seg_tiles * TILE

    def __call__(self, act, ids, scores, weights, out, send=None):
        """send = (boundary, site): fuse the boundary all-reduce's fabric send (fused_decode/boundary.py
        sending_program)."""
        cb_in0, cb_w, cb_idx, cb_scr, cb_out = 0, 1, 2, 3, 16
        kt = self.num_sel * self.seg_tiles
        cbs = [
            self._cb(cb_in0, ttnn.bfloat16, TINY_BYTES, kt, tiny=True),
            self._cb(cb_w, ttnn.bfloat4_b, BFP4_TILE_BYTES, 3 * kt),
            self._cb(cb_idx, ttnn.uint16, 64, 1),
            self._cb(cb_scr, ttnn.bfloat16, 128, 1),
            self._cb(cb_out, ttnn.bfloat16, TINY_BYTES, 4, tiny=True),
        ]
        reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for core, bank, r, vc in self.assignments:
            reader_rt[core.x][core.y] = [
                weights.buffer_address(),
                ids.buffer_address(),
                bank,
                vc,
                r * self.cols * self.seg_bytes,
            ]
            writer_rt[core.x][core.y] = [
                act.buffer_address(),
                scores.buffer_address(),
                out.buffer_address(),
                bank * self.C + r * self.cols,
            ]
        reader_ct = [cb_w, cb_idx, self.seg_tiles, self.cols, self.num_sel, self.expert_stride, BFP4_TILE_BYTES]
        reader_ct += ttnn.TensorAccessorArgs(ids).get_compile_time_args()
        writer_ct = [cb_in0, cb_out, cb_scr, self.seg_tiles, self.num_sel, NBIAS, self.cols, self.hidden // TILE]
        writer_ct += [self.inter_pad * 2]
        for t in (act, scores, out):
            writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        writer_ct += send[0].notify_args() if send else [0, 0, 0, 0]
        compute_ct = [cb_in0, cb_w, cb_out, kt, self.cols]

        def build():
            return self._program("down", cbs, reader_ct, reader_rt, writer_ct, writer_rt, compute_ct)

        program = send[0].sending_program(build, out, send[1], self.cores) if send else build()
        return ttnn.generic_op([weights, act, ids, scores, out], program)


class LinearStream(_BankStreamOp):
    """Dense decode linear y = x @ W + b for one token, weights streamed per DRAM bank (stream_linear_layout).

    x: x_pages = 0: the flat BF16 normed hidden of the layer boundary (fused_decode/boundary.py); otherwise a BF16
    32x32-tile tensor (any TensorAccessor layout; a DRAM source is staged in L1 first) whose activation is row
    (j / x_pages) of tile page (j % x_pages) for j < k_tiles (x_pages = 2: the [heads, 64] SDPA output, head by head).
    prefetch_cols: cap on the weight columns buffered ahead (default: all of them when a boundary is fused in, else 3).
    out_mode:
      3: attention heads, out = (Q, K, V) BF16 [heads, head_tiles * 32] tile tensors, head h in row h,
         heads = (q_cols, k_cols) output tiles of Q and K;
      4: router top-k + softmax, out = (ids, scores) [1, 32] UINT16 / BF16 row-major buffers, heads = (k, n_experts);
         rows > 1: per-row top-k, out = (union, weights): UINT16 [1, 64] = [U, the U union expert ids ascending],
         BF16 [1, 32 M]: row r's weight of union expert i at entry 32 r + i (0 if not routed);
      5: out = the flat BF16 partial sum the boundary all-reduces (output column n -> bytes [64 n, 64 n + 64)).
      6: (rows > 1, LM head) per-row argmax: out = a UINT32 buffer receiving, per worker core i, [key, index] per row
         at byte 8 M i (key: BF16 order key of the row's largest output on that core; argmax_lists merges them).
    rows = M > 1 (DFlash verify): M token rows per op, in [M, 32] activation / output tiles (kernels/
    stream_linear_writer.cpp item 5): x is the boundary's row-tile layout (x_pages = 0) or an [M, 32, 32 x_pages]
    source (one token row per user), out_mode 5 writes the row-tile partial, out_mode 3 [M, 32, head_tiles * 32] heads.
    """

    def __init__(
        self,
        mesh_device,
        k_tiles,
        n,
        weight_dtype,
        readers=1,
        x_pages=None,
        out_mode=3,
        out_page_bytes=2048,
        heads=(0, 0),
        head_tiles=2,
        prefetch_cols=None,
        rows=1,
        x_tile_rows=False,
        gather_split=False,
    ):
        """gather_split (x_pages > 0, rows > 1): each core gathers 1 / n of the activation tiles and copies them to the
        other cores (kernels/stream_linear_writer.cpp X_GATHER_SPLIT) instead of every core gathering all of them."""
        n_tiles = -(-n // TILE)
        banks = mesh_device.dram_grid_size().x
        self.C = -(-n_tiles // banks)
        super().__init__(mesh_device, readers, active_banks=-(-n_tiles // self.C))
        assert self.C % readers == 0, f"{self.C} columns per bank do not split over {readers} readers"
        self.cols = self.C // readers
        self.n_tiles = n_tiles
        self.kx = k_tiles
        self.kt = k_tiles + 1
        self.x_pages = 0 if x_pages is None else x_pages
        assert out_mode in (3, 4, 5, 6), f"out_mode {out_mode}"
        self.out_mode = out_mode
        self.out_page_bytes = out_page_bytes
        self.heads = heads
        self.head_tiles = head_tiles
        self.prefetch_cols = prefetch_cols
        self.rows = rows
        # x_tile_rows: x is a [32, 32 k_tiles] BF16 tile tensor whose rows 0 .. rows - 1 are the token rows
        self.x_tile_rows = x_tile_rows
        self.gather_split = gather_split and self.x_pages > 0 and rows > 1 and len(self.assignments) > 1
        assert not x_tile_rows or (x_pages is None and rows > 1)
        assert rows in (1, 2, 4, 8), "custom_mm in0 tiles are [1, 2, 4 or 8] x 32"
        assert out_mode != 6 or rows > 1, "out_mode 6 (per-row argmax) is the multi-row LM head"
        self.weight_dtype = weight_dtype
        self.tile_bytes = TILE_BYTES[weight_dtype]
        self.col_bytes = self.kt * self.tile_bytes
        self.page_bytes = _page_bytes(self.kt, self.tile_bytes)

    def weight_rows(self):
        return self.C * self.kt * TILE

    def _slot_columns(self, bank, r, order):
        """Output column of each of a worker's weight slots (the weights' slot order, stream_linear_layout)."""
        first = bank * self.C + r * self.cols
        return [order[first + c] for c in range(self.cols)]

    def __call__(
        self,
        x,
        weights,
        out,
        send=None,
        exchange=None,
        rope_kv=None,
        build_only=False,
        consumer_cores=None,
        extra_defines=None,
        extra_writer_rt=None,
    ):
        """send = (boundary, site), out_mode 5: fuse the boundary all-reduce's fabric send (fused_decode/boundary.py
        sending_program). x may be a fused_decode.boundary.PendingBoundary: the boundary producing x then runs inside
        this op (DecodeBoundary.consumer_parts) while the weight readers stream ahead. exchange (out_mode 3): an object
        whose writer_args() / program() add the fused decode LM head's top-k candidate merge and exchange
        (fused_decode/terminal.py TerminalExchange). rope_kv (out_mode 3, one user): a QKVRopeCache; Q and K are rotated
        in this op (kernels/stream_qkv_rope_compute.cpp) and K / V are written straight into the paged KV cache at the
        current position instead of the K / V head tensors (replacing rotary_embedding_llama_fused_qk and
        paged_fused_update_cache)."""
        pending = None
        if not isinstance(x, ttnn.Tensor):
            pending = x
            fused_kernels, fused_cbs, fused_semaphores, fused_io = pending.boundary.consumer_parts(
                pending, consumer_cores or self.cores, notify_in_reader=consumer_cores is not None
            )
            x = pending.x
        # the boundary's x mirror: read x from this core's L1 (kernels/stream_linear_writer.cpp X_LOCAL)
        mirror = getattr(pending.boundary, "x_mirror", None) if pending is not None else None
        x_local = mirror is not None and self.x_pages == 0 and not self.x_tile_rows
        if self.out_mode == 3:
            outs = tuple(out)
        elif self.out_mode == 6:
            outs = (out,) * 3
        elif self.out_mode == 4:
            outs = (out[0], out[1], out[1])
        else:
            outs = (out,) * 3
        cb_x, cb_w, cb_scr, cb_stage, cb_out = 0, 1, 3, 4, 16
        M = self.rows
        stage = (
            (self.kx if self.x_tile_rows else self.x_pages * M)
            if x.memory_config().buffer_type == ttnn.BufferType.DRAM
            else 0
        )
        # A fused boundary's consumer can buffer all its weight columns: the readers stream them while the boundary
        # runs (fused_decode.config.DECODE_BOUNDARY_PREFETCH_ALL).
        w_cols = max(3, self.cols) if pending is not None and DECODE_BOUNDARY_PREFETCH_ALL else 3
        if self.prefetch_cols is not None:
            # Large columns (the LM head's ~97 KB): cap the ring at what L1 holds.
            w_cols = min(w_cols, max(3, self.prefetch_cols))
        cbs = [
            self._cb(cb_x, ttnn.bfloat16, TINY_BYTES * M, self.kt, tiny=True, rows=M),
            self._cb(cb_w, self.weight_dtype, self.tile_bytes, w_cols * self.kt),
            # router scratch (multi-row: union ids, [M, 32] weights, one logit row, top-k stage)
            self._cb(cb_scr, ttnn.bfloat16, 64 if M == 1 else 128 + 64 * M + 128 + 64 * M, 1),
        ]
        if stage:
            cbs.append(self._cb(cb_stage, ttnn.bfloat16, 2048, stage))
        cbs.append(self._cb(cb_out, ttnn.bfloat16, TINY_BYTES * M, 4, tiny=True, rows=M))
        cb_y, cb_rot, cb_cs, cb_kv = 5, 6, 7, 12
        order = None
        if rope_kv is not None:
            assert self.out_mode == 3 and exchange is None
            assert (M > 1) == rope_kv.rows_mode, "multi-row QKV takes QKVRopeCache.rows (cos / sin tables)"
            order = qkv_column_order(self.n_tiles, self.heads[0], self.banks)
            # scratch + this core's cache tiles (multi-row: 3 scratch pages, up to two tiles per K / V column)
            kv_pages = 1 + self.cols if M == 1 else 3 + 2 * self.cols
            cbs += [
                self._cb(cb_y, ttnn.bfloat16, TINY_BYTES * M, 2, tiny=True, rows=M),
                self._cb(cb_rot, ttnn.bfloat8_b, BFP8_TILE_BYTES, 2),
                self._cb(cb_cs, ttnn.bfloat16, TINY_BYTES * M, 2 * self.head_tiles, tiny=True, rows=M),
                self._cb(cb_kv, ttnn.bfloat8_b, BFP8_TILE_BYTES, kv_pages),
            ]
        reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        gs_noc = []
        if self.gather_split:
            for c, *_ in self.assignments:
                p = self.mesh_device.worker_core_from_logical_core(c)
                gs_noc += [p.x, p.y]
        for i, (core, bank, r, vc) in enumerate(self.assignments):
            reader_rt[core.x][core.y] = [weights.buffer_address(), bank, vc, r * self.cols * self.col_bytes]
            writer_rt[core.x][core.y] = [
                mirror.buffer_address() if x_local else x.buffer_address(),
                outs[0].buffer_address(),
                bank * self.C + r * self.cols,
                outs[1].buffer_address(),
                outs[2].buffer_address(),
            ] + (exchange.writer_runtime_args(i) if exchange is not None else [])
            if self.out_mode == 6:
                writer_rt[core.x][core.y] = list(writer_rt[core.x][core.y]) + [i]
            if self.gather_split:
                writer_rt[core.x][core.y] = list(writer_rt[core.x][core.y]) + [i, len(self.assignments)] + gs_noc
            if rope_kv is not None:
                writer_rt[core.x][core.y] = (
                    list(writer_rt[core.x][core.y]) + rope_kv.writer_runtime_args() + self._slot_columns(bank, r, order)
                )
            if extra_writer_rt is not None:
                writer_rt[core.x][core.y] = list(writer_rt[core.x][core.y]) + list(extra_writer_rt)
        reader_ct = [cb_w, self.kt, self.cols, self.tile_bytes, self.page_bytes, w_cols]
        writer_ct = [cb_x, cb_out, cb_scr, self.kx, self.x_pages, NBIAS, self.cols, self.n_tiles, self.out_mode]
        writer_ct += [self.out_page_bytes, stage, cb_stage, self.heads[0], self.heads[1], self.head_tiles]
        for t in (x, *outs):
            writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        writer_ct += send[0].notify_args() if send else [0, 0, 0, 0]
        writer_ct += [1 if pending is not None else 0]
        writer_ct += exchange.writer_compile_args() if exchange is not None else [0, 0, 0, 0]
        compute_ct = [cb_x, cb_w, cb_out, self.kt, self.cols]
        compute, compute_rt, defines = "matmul", None, None
        if rope_kv is not None:
            writer_ct += [cb_rot, cb_cs, cb_kv] + rope_kv.writer_compile_args()
            compute = "qkv_rope"
            compute_ct += [cb_y, cb_rot, cb_cs, self.heads[0] + self.heads[1], self.heads[0], self.head_tiles]
            compute_rt = ttnn.RuntimeArgs()
            for core, bank, r, vc in self.assignments:
                compute_rt[core.x][core.y] = self._slot_columns(bank, r, order)
            defines = [("QKV_ROPE_KV", "1")]
        if M > 1:
            defines = (defines or []) + [("ROWS", str(M))]
        if self.x_tile_rows:
            defines = defines + [("X_TILE_ROWS", "1")]
        if self.gather_split:
            defines = (defines or []) + [("X_GATHER_SPLIT", "1")]
        if x_local:
            defines = (defines or []) + [("X_LOCAL", "1")]
        b_ = pending.boundary if pending is not None else None
        if b_ is not None and getattr(b_, "mc", False) and b_.direct_notify:
            defines = (defines or []) + [("BND_WAIT_N", str(b_.H))]
        if extra_defines:
            defines = (defines or []) + list(extra_defines)
        if os.environ.get("LIN_T") and rope_kv is not None:  # (profiling: the QKV writer's per-column timeline)
            defines = (defines or []) + [("LIN_T", "1")]
        # DFLASH_KV_SPLIT=1 (multi-row QKV with the K / V cache write): the reader converts half the rows of the first
        # (K / V) output column to BFP8 while the writer converts the others (kernels/stream_linear_reader.cpp)
        kv_split = rope_kv is not None and M > 1 and os.environ.get("DFLASH_KV_SPLIT", "1") == "1"
        if kv_split:
            defines = (defines or []) + [
                ("KV_SPLIT", "1"),
                ("KV_SPLIT_SEM", "2"),
                ("KV_SPLIT_CB_OUT", str(cb_out)),
                ("KV_SPLIT_CB_STAGE", "13"),
            ]
            cbs.append(self._cb(13, ttnn.uint32, 64 * M, 1))

        def build():
            program = self._program(
                "linear",
                cbs,
                reader_ct,
                reader_rt,
                writer_ct,
                writer_rt,
                compute_ct,
                compute=compute,
                compute_rt=compute_rt,
                defines=defines,
            )
            if self.gather_split:
                program.semaphores = list(program.semaphores) + [
                    ttnn.SemaphoreDescriptor(1, ttnn.CoreType.WORKER, self.cores, 0)
                ]
            if pending is not None:
                program.kernels = list(program.kernels) + fused_kernels
                program.cbs = list(program.cbs) + fused_cbs
                program.semaphores = list(program.semaphores) + list(fused_semaphores)
            if kv_split:
                assert all(sd.id != 2 for sd in program.semaphores)
                program.semaphores = list(program.semaphores) + [
                    ttnn.SemaphoreDescriptor(2, ttnn.CoreType.WORKER, self.cores, 0)
                ]
            return program

        assert not (send and pending), "an op is either a boundary producer or its consumer"
        assert not (send and exchange), "the LM-head exchange and a boundary send do not combine"
        if exchange is not None:
            assert self.out_mode == 3
            program = exchange.program(build, self.cores)
        else:
            program = send[0].sending_program(build, out, send[1], self.cores) if send else build()
        unique = [t for i, t in enumerate(outs) if all(t is not o for o in outs[:i])]
        extra = [t for t in fused_io if all(t is not u for u in (x, *unique))] if pending is not None else []
        if exchange is not None:
            extra += [t for t in exchange.io if all(t is not u for u in (x, *unique, *extra))]
        if rope_kv is not None:
            for t in rope_kv.io:
                if all(t is not u for u in (x, *unique, *extra)):
                    extra.append(t)
        if build_only:
            return program, [weights, x, *unique, *extra]
        ttnn.generic_op([weights, x, *unique, *extra], program)
        return out


class QKVRopeCache:
    """The fused QKV op's RoPE + KV-cache inputs (LinearStream rope_kv, one user): rot = [I, T] (BF8 32x32: identity and
    the rotary pair-rotation matrix of rotary_embedding_llama), the position's cos / sin rows (the fused-QK RoPE inputs,
    row 0 of the first user's [32, head_dim] shard), the position tensor, the page table and the paged K / V caches
    ([blocks, kv_heads, block_size, head_dim] BF8 tiles). page_table None: unpaged caches ([users, kv_heads, positions,
    head_dim]; the one user's rows are block 0 with block_size = positions)."""

    def __init__(self, rot, cos, sin, pos, page_table, k_cache, v_cache):
        self.rows_mode = False
        self.paged = page_table is not None
        self.num_kv_heads, self.block_size = k_cache.shape[1], k_cache.shape[2]
        # Unpaged: the page-table accessor slot holds the position tensor (never read).
        self.io = [rot, cos, sin, pos, page_table if self.paged else pos, k_cache, v_cache]

    def writer_compile_args(self):
        ct = [self.num_kv_heads, self.block_size, BFP8_TILE_BYTES, int(self.paged)]
        for t in self.io:
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        return ct

    def writer_runtime_args(self):
        return [t.buffer_address() for t in self.io]

    @classmethod
    def rows(cls, rot, cos_table, sin_table, pos, page_table, k_cache, v_cache):
        """Multi-row QKV (LinearStream rows = M, one user, DFlash verify): pos is the [M] INT32 position tensor of the
        rows (consecutive); cos_table / sin_table the row-major [.., positions, head_dim] BF16 RoPE tables (one page per
        position, RotarySetup.cos_matrix / sin_matrix)."""
        obj = cls(rot, cos_table, sin_table, pos, page_table, k_cache, v_cache)
        obj.rows_mode = True
        return obj


_ROPE_ROTATION_TILES = {}


def rope_rotation_tiles(mesh_device):
    """[I, T] as one BF8 [1, 1, 64, 32] DRAM tensor (two tiles), replicated: exact (entries 0 / +-1). One per mesh."""
    cached = _ROPE_ROTATION_TILES.get(id(mesh_device))
    if cached is None or cached[0] is not mesh_device:  # the entry holds the mesh, so its id is not reused meanwhile
        cached = _ROPE_ROTATION_TILES[id(mesh_device)] = (mesh_device, _rope_rotation_tiles(mesh_device))
    return cached[1]


def _rope_rotation_tiles(mesh_device):
    from models.tt_transformers.tt.common import get_rot_transformation_mat

    t = get_rot_transformation_mat(dhead=TILE).reshape(TILE, TILE).float()
    rot = torch.cat([torch.eye(TILE), t]).reshape(1, 1, 2 * TILE, TILE)
    return ttnn.from_torch(
        rot,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def merge_argmax(device_lists, rows, vocab_per_device):
    """device_lists: per device a flat int64 tensor [cores * rows * 2] of (key, local index) -> [rows] global argmax
    ids (ties to the lower id)."""
    best = [(-(1 << 40), 0)] * rows
    for d, lst in enumerate(device_lists):
        t = lst.reshape(-1, rows, 2)
        for c in range(t.shape[0]):
            for r in range(rows):
                key, idx = int(t[c, r, 0]), int(t[c, r, 1])
                key = key - (1 << 32) if key >= (1 << 31) else key  # signed order key
                gid = d * vocab_per_device + idx
                if key > best[r][0] or (key == best[r][0] and gid < best[r][1]):
                    best[r] = (key, gid)
    return [b[1] for b in best]


def read_lists(lists, mesh_device):
    """Host: the per-device out_mode 6 lists of a [1, n] UINT32 mesh tensor in one readback."""
    t = ttnn.to_torch(lists, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    return [row.reshape(-1).long() & 0xFFFFFFFF for row in t.reshape(mesh_device.get_num_devices(), -1)]
