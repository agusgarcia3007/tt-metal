# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""K / V rows of one sequence -> its paged BF16 draft cache in one single-core op (fused_decode/kernels/
draft_kv_write.cpp): replaces one paged_update_cache per row (the op's per-row tile read-modify-write races when rows of
a call share a tile), and takes its positions from a device tensor, so it can be traced."""

import ttnn

from ..fused_decode.boundary import KERNEL_DIR

TILE = ttnn.TILE_SIZE
CORE = ttnn.CoreCoord(0, 0)


class DraftKVWrite:
    def __init__(self, mesh_device, rows, kv_heads, head_dim, block_size, head_grid_x=8):
        """rows: rows written per call (row r of the K / V head tensors: user r, sharded one user per core on a
        head_grid_x-wide row-major grid)."""
        self.mesh_device = mesh_device
        self.rows, self.kv_heads, self.head_tiles, self.block = rows, kv_heads, head_dim // TILE, block_size
        self.coords = []
        for r in range(rows):
            p = mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(r % head_grid_x, r // head_grid_x))
            self.coords += [p.x, p.y]
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(CORE, CORE)])
        self.scratch = 512 + (2 * rows * self.head_tiles + 4 * kv_heads * self.head_tiles) * 2048

    def __call__(self, k, v, k_cache, v_cache, pos, page_table, head_offset=0):
        """k, v: [1, 32, kv_heads, head_dim] BF16 height-sharded (one user row per core); pos: INT32 [32] (row r's
        position, < 0 skipped; the valid rows' positions are consecutive); page_table: INT32 [*, blocks] (row 0)."""
        ct = [0, self.rows, self.kv_heads, self.head_tiles, self.block, page_table.shape[-1] * 4, head_offset]
        for t in (k_cache, v_cache, pos, page_table):
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        rt = ttnn.RuntimeArgs()
        rt[CORE.x][CORE.y] = [
            k.buffer_address(),
            v.buffer_address(),
            k_cache.buffer_address(),
            v_cache.buffer_address(),
            pos.buffer_address(),
            page_table.buffer_address(),
        ] + self.coords
        # K on BRISC / NOC0 and V on NCRISC / NOC1 of the same core, in parallel
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "draft_kv_write.cpp"),
                core_ranges=self.cores,
                compile_time_args=ct,
                runtime_args=rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=(ttnn.DataMovementProcessor.RISCV_0, ttnn.DataMovementProcessor.RISCV_1)[sel],
                    noc=(ttnn.NOC.NOC_0, ttnn.NOC.NOC_1)[sel],
                ),
                defines=[("KV_SEL", str(sel))],
            )
            for sel in (0, 1)
        ]
        cb = ttnn.CBDescriptor(
            total_size=self.scratch,
            core_ranges=self.cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=self.scratch)
            ],
        )
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[cb])
        ttnn.generic_op([k, v, pos, page_table, k_cache, v_cache], program)
