# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Attention of a DFlash block (M rows of one sequence) with one K / V cache read for all M rows.

The M rows share their K / V history. Passing them to paged_scaled_dot_product_attention_decode as M users makes
every row stream the whole cache from DRAM (M reads). Its spec_multi_pos_tiles mode folds candidates of one batch row
onto that row (one K / V read, a causal bound per candidate) but needs one kv head per batch row, and with
spec_rows_per_candidate = 8 it packs 4 candidates per 32-row Q tile (8 query heads each: no padding rows). So:
  - each device's cache [blocks, nkv, block, 64] is viewed as [blocks * nkv, 1, block, 64] (kv head h of block b =
    view block nkv * b + h) and batch row h gets the page table nkv * pt + h: batch h = kv head h;
  - batch h's Q row-tile j holds rows 4 j .. 4 j + 3, each as kv head h's 8 query heads (tile rows 8 i .. 8 i + 7 =
    row 4 j + i); its bound is row 4 j + 3's position (the op derives the others: consecutive positions);
  - the sink tile of (h, j) holds kv head h's 8 sinks in every 8-row group;
  - fused_decode/kernels/spec_attn_copy.cpp writes the bounds (once per pass), packs Q, and unpacks the output into
    the [1, M, heads, 64] layout the o_proj streams read.
uniform=True (the drafter: every row attends the same positions [0, pos[M - 1]], not causal inside the block): the
same packed layout with spec_rows_per_candidate = 32 (the op masks each 32-row tile with its one bound, so every row
gets it) and every tile's bound = pos[M - 1]; no sinks.
A window W > 0 (spec mode has no sliding window): each pass the page table starts at the block holding position
pos[M - 1] + 1 - W (the cached K are rotated: only the causal bounds depend on positions), so the rows attend the
last W .. W + block - 1 positions, read once.
"""

import os

import torch

import ttnn

from ..fused_decode.boundary import KERNEL_DIR

TILE = ttnn.TILE_SIZE
CORE = ttnn.CoreCoord(0, 0)


class SharedKVAttention:
    def __init__(
        self,
        mesh_device,
        rows,
        n_heads,
        n_kv,
        head_dim,
        page_table,
        block_size,
        program_config,
        out_mc,
        window=0,
        uniform=False,
    ):
        """page_table: the sequence's host [1, blocks] INT32 page table; out_mc: the output's memory config."""
        assert head_dim == 64 and n_heads // n_kv == 8 and n_kv * 8 <= 16, "kv head h's query heads in rows 8h..8h+7"
        assert rows % 4 == 0
        self.mesh, self.M, self.nq, self.nkv, self.hd = mesh_device, rows, n_heads, n_kv, head_dim
        # spec mode holds every candidate's Q / score tiles per core: K chunks over the whole grid
        self.pc = program_config or ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(
                *[int(v) for v in os.environ.get("SKV_GRID", "11,10").split(",")]
            ),
            q_chunk_size=0,
            k_chunk_size=int(os.environ.get("SKV_K", "128")),
            exp_approx_mode=False,
            max_cores_per_head_batch=int(os.environ.get("SKV_CORES", "55")),
        )
        self.block, self.window, self.uniform = block_size, window, uniform
        self.Tt = rows // 4  # packed Q row-tiles per kv head
        # SKV_SPLIT=1: every packed row-tile is its own spec batch row (nkv * Tt rows, 1 tile each: the same memory
        # layout): each core's work per K chunk halves, the cache is read Tt times instead of once
        self.split = os.environ.get("SKV_SPLIT", "1") == "1"
        self.B = n_kv * self.Tt if self.split else n_kv
        rep = ttnn.ReplicateTensorToMesh(mesh_device)
        dram = ttnn.DRAM_MEMORY_CONFIG
        pt = page_table.reshape(1, -1).int()
        self.pt_bytes = pt.shape[-1] * 4
        if window:
            # positions < W + block (+ the block's rows), in whole 64-byte rows (the kernel's NOC writes)
            self.pt_n = -(-(window // block_size + 2) // 16) * 16
            spec_pt = torch.zeros(self.B, self.pt_n, dtype=torch.int32)
        else:
            self.pt_n = pt.shape[-1]
            spec_pt = torch.cat([pt * n_kv + b * n_kv // self.B for b in range(self.B)])
        self.page_table = ttnn.from_torch(spec_pt, dtype=ttnn.int32, device=mesh_device, mesh_mapper=rep)
        self.q = ttnn.from_torch(
            torch.zeros(1, self.B, n_kv * self.Tt * TILE // self.B, head_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=dram,
            mesh_mapper=rep,
        )
        self.pos = ttnn.from_torch(
            torch.zeros(n_kv * self.Tt, dtype=torch.int32), dtype=ttnn.int32, device=mesh_device, mesh_mapper=rep
        )
        self.out = ttnn.from_torch(
            torch.zeros(1, rows, n_heads, head_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=out_mc,
            mesh_mapper=rep,
        )
        self.views, self.sinks = {}, {}
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(CORE, CORE)])

    def prepare(self, caches, sinks):
        """Before any trace capture (persistent tensors): the [blocks * nkv, 1, block, 64] views of the layers' K / V
        caches and, per sink tensor (the heads' [heads, 32] tile of sink logits / scale per device), the per-(kv head,
        row-tile) sink tiles that spec mode reads."""
        for c in caches:
            s = c.shape
            self.views[c.buffer_address()] = ttnn.reshape(c, [s[0] * s[1], 1, s[2], s[3]])
        for t in sinks:
            if t.buffer_address() in self.sinks:
                continue
            per_dev = []
            for d in ttnn.get_device_tensors(t):
                x = ttnn.to_torch(d).float().reshape(-1, TILE)  # [heads, 32]: column 0 = sink / scale
                # tile (h, j): kv head h's 8 sinks in each 8-row group
                per_dev.append(torch.cat([x[8 * h : 8 * h + 8].repeat(4 * self.Tt, 1) for h in range(self.nkv)]))
            self.sinks[t.buffer_address()] = ttnn.from_torch(
                torch.cat(per_dev),
                dtype=t.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0),
            )

    def _copy(self, mode, io, ct_extra=()):
        """SPEC_POS: one core (spec_attn_copy.cpp); SPEC_Q / SPEC_OUT: one core per kv head moving its 256-byte row
        segments with the NOC (spec_attn_copy_noc.cpp)."""
        noc = mode != "SPEC_POS"
        if noc:
            cores = [ttnn.CoreCoord(h, 0) for h in range(self.nkv)]
            scr = 2 * self.Tt * 2048
        else:
            cores = [CORE]
            scr = 256 + ((self.pt_n + 32) * 4 + 63) // 64 * 64 * (self.B + 1)
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(cores[0], cores[-1])])
        ct = [self.M, self.nkv, 0, *ct_extra]
        for t in io:
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        rta = ttnn.RuntimeArgs()
        for h, c in enumerate(cores):
            rta[c.x][c.y] = [t.buffer_address() for t in io] + ([h] if noc else [])
        k = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / ("spec_attn_copy_noc.cpp" if noc else "spec_attn_copy.cpp")),
            core_ranges=crs,
            compile_time_args=ct,
            runtime_args=rta,
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0),
            defines=[(mode, "1"), ("SPEC_B", str(self.B))]
            + ([("SPEC_UNIFORM", "1")] if self.uniform and mode == "SPEC_POS" else []),
        )
        cb = ttnn.CBDescriptor(
            total_size=scr,
            core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=scr)],
        )
        ttnn.generic_op(list(io), ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[cb]))

    def begin(self, pos, base_page_table):
        """Once per pass, before the layers: pos INT32 [>= M] (row r attends [0, pos[r]], non-decreasing; uniform:
        every row [0, pos[M - 1]]; with a window the last W .. W + block - 1 positions); base_page_table: the
        sequence's device page table (row 0 used)."""
        ct = [self.window, self.block, self.pt_n, self.pt_bytes, pos.shape[-1] * 4]
        self._copy("SPEC_POS", (pos, self.pos, base_page_table, self.page_table), ct)

    def load_q(self, q):
        """q [1, >= M, heads, 64] BF16 tiles -> the packed Q (a producer that writes the packed layout itself, e.g. the
        verify pass's QKV with spec_tt, skips this)."""
        self._copy("SPEC_Q", (q, self.q))

    def attend(self, k_cache, v_cache, sinks, scale, compute_kernel_config=None):
        """The attention of the packed Q -> the packed output [1, nkv, Tt * 32, 64] (DRAM; the o_proj stream reads it
        with spec_tt; the caller deallocates it)."""
        return ttnn.transformer.paged_scaled_dot_product_attention_decode(
            self.q,
            self.views[k_cache.buffer_address()],
            self.views[v_cache.buffer_address()],
            cur_pos_tensor=self.pos,
            attention_sink=None if sinks is None else self.sinks[sinks.buffer_address()],
            page_table_tensor=self.page_table,
            scale=scale,
            program_config=self.pc,
            compute_kernel_config=compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            spec_multi_pos_tiles=self.nkv * self.Tt // self.B,
            spec_rows_per_candidate=32 if self.uniform else 8,
        )

    def __call__(self, q, k_cache, v_cache, sinks, scale, compute_kernel_config=None):
        """Q copy in, attention, output copy out -> self.out [1, M, heads, 64] (standalone use / tests)."""
        self.load_q(q)
        o = self.attend(k_cache, v_cache, sinks, scale, compute_kernel_config)
        self._copy("SPEC_OUT", (o, self.out))
        o.deallocate(True)
        return self.out
