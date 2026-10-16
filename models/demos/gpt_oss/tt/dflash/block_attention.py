# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Attention of the DFlash verify block with one K / V cache read for all M rows (fused_decode/kernels/ba_*.cpp),
row attention's design (dflash/row_attention.py: its own kernels, fused into the QKV program so the history is read
while QKV runs) with the rows sharing the reads:

  - the M rows' queries of kv head g are Tt = M / 4 packed tiles (tile t, rows 8 i .. 8 i + 7 = block row 4 t + i's
    query heads 8 g .. 8 g + 7: every tile row is real; the verify QKV writes Q in this layout with spec_tt);
  - one core per (kv head g, history slice s): the block's history (positions 0 .. pos[M - 1]) is cut into A <= S
    contiguous slices of at least min_chunks 32-position chunks; each slice core reads its chunks once and runs all Tt
    Q tiles over them (row group i of tile t masked causally at pos[4 t + i]), in groups of G chunks with a running
    softmax (any length streams through L1);
  - the slices merge their softmax partials (m, l, O) up a binary tree (log2 A rounds, NOC writes of the partial into
    the parent's L1); slice 0 (with the sink logits) writes out = O / l as the packed output, which o_proj reads with
    spec_tt.
"""

import os
import struct
from pathlib import Path

import torch

import ttnn

from ..fused_decode.boundary import KERNEL_DIR

TILE = ttnn.TILE_SIZE
TREE_SEM = 9
LAND_SEM, COPY_SEM = 10, 11  # BA_SHARE: groups landed in the leader / copied by its partner
# BA_KDIR: a snapshot directory of the ba_*.cpp kernels (tests run against it while the sources change)
BA_DIR = Path(os.environ["BA_KDIR"]) if os.environ.get("BA_KDIR") else KERNEL_DIR


def _f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


class BlockAttention:
    def __init__(
        self,
        mesh_device,
        rows,
        n_heads,
        n_kv,
        head_dim,
        block_size,
        cores,
        slices,
        out_mc=None,
        uniform=False,
        window=0,
        q_rows=False,
        min_chunks=None,
    ):
        """cores: cores_needed(n_kv, rows, slices) logical cores: (kv head g, tile group q, slice s) = cores[(g Tq + q)
        slices + s], each tile group (BA_TPC packed Q tiles per core, Tq = Tt / BA_TPC groups) with its own slices and
        merge tree. uniform: every row attends [0, pos[M - 1]] (the drafter's block: no causal mask inside it), no
        sinks; window: uniform, the rows attend the last `window` positions [pos[M - 1] + 1 - window, pos[M - 1]]; else
        a sliding-window layer, row r attends [pos[r] + 1 - window, pos[r]].
        q_rows: q comes in the rows layout [1, M, heads, 64] (the reader gathers the packed tiles: no repacking op).
        """
        assert head_dim == 64 and n_heads // n_kv == 8 and rows % 4 == 0
        self.mesh, self.M, self.nq, self.nkv, self.hd = mesh_device, rows, n_heads, n_kv, head_dim
        self.block, self.S, self.Tt = block_size, slices, rows // 4
        self.uniform, self.window, self.q_rows = uniform, window, q_rows
        self.TC = self.tiles_per_core(rows)
        self.Tq = self.Tt // self.TC
        # BA_SHARE=1 (2 tile groups): one DRAM read of each K / V group per (kv head, slice), the partner copies it
        self.share = self.Tq == 2 and os.environ.get("BA_SHARE", "1") == "1"
        assert len(cores) == self.cores_needed(n_kv, rows, slices)
        self.core_list = list(cores)
        self.grid = mesh_device.compute_with_storage_grid_size()
        assert len(self.core_list) < 0xFF
        self.cores = _rects(self.core_list)
        self.min_chunks = min_chunks or int(os.environ.get("BA_MIN_CHUNKS", "4"))
        self.G = int(os.environ.get("BA_G", "8"))
        self.cap = int(os.environ.get("BA_CAP", "3"))  # 4 clashes with the verify pass's L1 buffers
        self.rounds = max(1, (slices - 1).bit_length())
        self.mask_lib = ttnn.from_torch(
            self._mask_lib(),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        # the packed Q (a producer writes it in this layout: the verify QKV with spec_tt) and the packed output (o_proj
        # reads it with spec_tt); BA_L1=1: both in L1 (the consumer's reads skip DRAM latency; o_proj reads rows
        # straight from an L1 source, a DRAM one is staged first)
        mc = ttnn.L1_MEMORY_CONFIG if os.environ.get("BA_L1", "1") == "1" else ttnn.DRAM_MEMORY_CONFIG
        rep = ttnn.ReplicateTensorToMesh(mesh_device)
        self.q, self.out = (
            ttnn.from_torch(
                torch.zeros(1, n_kv, self.Tt * TILE, head_dim),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=out_mc or mc,
                mesh_mapper=rep,
            )
            for _ in range(2)
        )

    @staticmethod
    def _mask_lib():
        """The mask library: segment (a, b, f) (columns [a, b) kept, a, b = 0 .. 32, half f) = 8 rows x 16 BF16 columns
        of a 32x32 tile face, 0 at the kept columns and -1e30 elsewhere (64 words)."""
        w = torch.zeros(33, 33, 2, 8, 8, dtype=torch.int64)
        cols = torch.arange(32).reshape(2, 8, 2)  # [f, c, lo / hi half of the word]
        for a in range(33):
            for k in range(33):
                cut = ((cols < a) | (cols >= k)).long() * 0xF14A
                w[a, k] = (cut[..., 0] | (cut[..., 1] << 16))[:, None, :].expand(2, 8, 8)
        w = w.reshape(1, -1)
        return torch.where(w >= 1 << 31, w - (1 << 32), w).to(torch.int32)

    @staticmethod
    def tiles_per_core(rows):
        tc = int(os.environ.get("BA_TPC", "1"))
        assert (rows // 4) % tc == 0
        return tc

    @classmethod
    def cores_needed(cls, n_kv, rows, slices):
        return n_kv * (rows // 4 // cls.tiles_per_core(rows)) * slices

    def noc_cores(self):
        return [self.mesh.worker_core_from_logical_core(c) for c in self.core_list]

    def notify_order(self):
        """NOC coordinates of the cores in slice order (the producer's signals reach slice 0 first: the slices that
        have work at short context)."""
        noc = self.noc_cores()
        return [noc[i] for i in sorted(range(len(noc)), key=lambda i: (i % self.S, i))]

    def parts(self, q, k_cache, v_cache, pos, page_table, sinks, scale, wait):
        """(kernels, cbs, io); q: the packed Q [1, nkv, Tt * 32, 64] (tile (g, t, d) at page (g Tt + t) 2 + d); pos:
        INT32 [>= M] (row r at pos[r], non-decreasing, consecutive rows within 32 positions); page_table: INT32 [*,
        blocks] (row 0 used); sinks: the heads' [heads, 32] BF16 sink tile; wait = (semaphore id, count): the QKV
        writers' signal that q and the block's K / V rows are written."""
        Tt, G, cap, S = self.TC, self.G, self.cap, self.S  # Tt: this core's packed Q tiles
        sinks = sinks if sinks is not None else self.mask_lib  # uniform: no sinks (a placeholder address)
        kv_dtype = k_cache.dtype
        kv_tile = {ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}[kv_dtype]
        pt_page = page_table.shape[-1] * 4
        pos_page = pos.shape[-1] * 4
        f32, bf16 = ttnn.float32, ttnn.bfloat16
        partial = 4 * Tt  # [m, l, O_0, O_1] per Q tile
        (cb_q, cb_k, cb_v, cb_aux, cb_scr, cb_qs, cb_s, cb_p) = range(8)
        cb_m, cb_l, cb_o = 8, 10, 12  # + t: Q tile t's running state
        (cb_ab, cb_out, cb_send, cb_recv, cb_scr2, cb_lacc) = range(14, 20)

        def cb(i, dtype, page, pages):
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=self.cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dtype, page_size=page)],
            )

        cbs = [
            cb(cb_q, bf16, 2048, 2 * Tt),
            cb(cb_k, kv_dtype, kv_tile, 2 * G * cap),
            cb(cb_v, kv_dtype, kv_tile, 2 * G * cap),
            cb(cb_aux, bf16, 2048, (4 if self.window and not self.uniform else 2) * Tt + 4),
            cb(cb_scr, bf16, 256 + max(pt_page, 64), 1),
            cb(cb_qs, bf16, 2048, 2 * Tt),
            cb(cb_s, f32, 4096, Tt * (G + 1)),
            cb(cb_p, bf16, 2048, G + 1),
            cb(cb_ab, f32, 4096, 2),
            cb(cb_out, bf16, 2048, 2 * Tt),
            cb(cb_send, f32, 4096, partial),
            cb(cb_recv, f32, 4096, partial * self.rounds),
            cb(cb_scr2, bf16, max(64, pos_page), 1),
            cb(cb_lacc, f32, 4096, 1),
        ]
        for t in range(Tt):  # the running state, double-buffered (the update pushes before it pops)
            cbs += [cb(cb_m + t, f32, 4096, 2), cb(cb_l + t, f32, 4096, 2), cb(cb_o + t, f32, 4096, 4)]
        reader_ct = [
            cb_q,
            cb_k,
            cb_v,
            cb_aux,
            cb_scr,
            cb_recv,
            self.M,
            self.nkv,
            self.block,
            kv_tile,
            pt_page,
            pos_page,
        ]
        reader_ct += [S, self.min_chunks, G, cap, partial]
        for t in (q, k_cache, v_cache, pos, page_table, sinks, self.mask_lib):
            reader_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        writer_ct = [cb_out, cb_scr2, cb_send, cb_recv, self.M, pos_page, S, self.min_chunks, partial]
        for t in (self.out, pos):
            writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        compute_ct = [cb_q, cb_k, cb_v, cb_aux, cb_qs, cb_s, cb_p, cb_m, cb_l, cb_o, cb_ab, cb_out, cb_send, cb_recv]
        compute_ct += [cb_lacc, _f32_bits(scale), self.M, G]
        # every argument is common (no per-core runtime args: dispatch writes those core by core): each core finds its
        # index in the core list from its logical coordinates (a byte table over the grid) and its tree parent's NOC
        # coordinates from a table over the core list
        noc = self.noc_cores()
        gx, gy = self.grid.x, self.grid.y
        idx = [0xFF] * (gx * gy)
        for i, c in enumerate(self.core_list):
            idx[c.y * gx + c.x] = i
        idx += [0xFF] * (-len(idx) % 4)
        idx_words = [sum(idx[w + k] << (8 * k) for k in range(4)) for w in range(0, len(idx), 4)]
        nocs = [(p.x << 8) | p.y for p in noc] + [0] * (len(noc) % 2)
        noc_words = [nocs[w] | (nocs[w + 1] << 16) for w in range(0, len(nocs), 2)]
        reader_common = (
            [
                q.buffer_address(),
                k_cache.buffer_address(),
                v_cache.buffer_address(),
                pos.buffer_address(),
                page_table.buffer_address(),
                sinks.buffer_address(),
                self.mask_lib.buffer_address(),
            ]
            + idx_words
            + noc_words
        )
        writer_common = [self.out.buffer_address(), pos.buffer_address()] + idx_words + noc_words
        defines = [("BA_WAIT_SEM", str(wait[0])), ("BA_WAIT_N", str(wait[1])), ("BA_TREE_SEM", str(TREE_SEM))]
        defines += [("BA_TPC", str(self.TC)), ("BA_TQ", str(self.Tq)), ("BA_GX", str(gx))]
        defines += [("BA_NIDX", str(len(idx_words))), ("BA_CB_POS", str(cb_scr2))]
        if self.share:
            defines += [("BA_SHARE", "1"), ("BA_LAND_SEM", str(LAND_SEM)), ("BA_COPY_SEM", str(COPY_SEM))]
        if os.environ.get("DFLASH_BA_CAP0"):  # groups in flight before Q (kernels/ba_reader.cpp)
            defines.append(("BA_CAP0", os.environ["DFLASH_BA_CAP0"]))
        if self.uniform:
            defines.append(("BA_UNIFORM", "1"))
        if self.q_rows:
            defines.append(("BA_Q_ROWS", "1"))
        if self.window:  # uniform: one window for all rows; else per row (a sliding-window layer)
            defines.append(("BA_WINDOW" if self.uniform else "BA_SLIDE", str(self.window)))
        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(BA_DIR / "ba_reader.cpp"),
                core_ranges=self.cores,
                compile_time_args=reader_ct,
                runtime_args=[],
                common_runtime_args=reader_common,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
                defines=defines,
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(BA_DIR / "ba_writer.cpp"),
                core_ranges=self.cores,
                compile_time_args=writer_ct,
                runtime_args=[],
                common_runtime_args=writer_common,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
                defines=defines,
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(BA_DIR / "ba_compute.cpp"),
                core_ranges=self.cores,
                compile_time_args=compute_ct,
                runtime_args=[],
                defines=[(k, v) for k, v in defines if k in ("BA_TPC", "BA_UNIFORM", "BA_WINDOW", "BA_SLIDE")]
                + ([("BA_PROF", "1")] if os.environ.get("BA_PROF") == "1" else [])
                + ([("BA_ABLATE", os.environ["BA_ABLATE"])] if os.environ.get("BA_ABLATE") else []),
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=getattr(ttnn.MathFidelity, os.environ.get("BA_FID", "HiFi2")),
                    fp32_dest_acc_en=True,
                    math_approx_mode=False,
                ),
            ),
        ]
        io = [q, k_cache, v_cache, pos, page_table, sinks, self.mask_lib, self.out]
        return kernels, cbs, [t for i, t in enumerate(io) if all(t is not u for u in io[:i])]

    def semaphores(self):
        """The op's own program semaphores (besides the wait semaphore)."""
        sems = [ttnn.SemaphoreDescriptor(TREE_SEM, ttnn.CoreType.WORKER, self.cores, 0)]
        if self.share:
            sems += [ttnn.SemaphoreDescriptor(i, ttnn.CoreType.WORKER, self.cores, 0) for i in (LAND_SEM, COPY_SEM)]
        return sems

    def __call__(self, q, k_cache, v_cache, pos, page_table, sinks, scale):
        """Standalone (tests): q already written."""
        kernels, cbs, io = self.parts(q, k_cache, v_cache, pos, page_table, sinks, scale, wait=(3, 0))
        sems = self.semaphores() + [ttnn.SemaphoreDescriptor(3, ttnn.CoreType.WORKER, self.cores, 0)]
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs))
        return self.out


def _rects(cores):
    """The cores as few rectangles as possible (row runs, stacked when consecutive rows share the run): dispatch writes
    kernel binaries and arguments per core range."""
    rows = {}
    for c in cores:
        rows.setdefault(c.y, []).append(c.x)
    runs = []  # (x0, x1, y)
    for y in sorted(rows):
        xs = sorted(rows[y])
        x0 = prev = xs[0]
        for x in xs[1:] + [None]:
            if x is None or x != prev + 1:
                runs.append((x0, prev, y))
                x0 = x
            prev = x if x is not None else prev
    rects = []  # [x0, x1, y0, y1]
    for x0, x1, y in runs:
        r = next((r for r in rects if r[0] == x0 and r[1] == x1 and r[3] == y - 1), None)
        if r is not None:
            r[3] = y
        else:
            rects.append([x0, x1, y, y])
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(a, c), ttnn.CoreCoord(b, d)) for a, b, c, d in rects])


def free_cores(mesh_device, taken, n):
    """n logical worker cores not in taken ((x, y) set), row-major from the grid's top-left."""
    grid = mesh_device.compute_with_storage_grid_size()
    out = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in taken]
    assert len(out) >= n, f"{len(out)} free cores, {n} needed"
    return out[:n]
