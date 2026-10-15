# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Two-stage verify LM head (DFLASH_LM_PRESCREEN): stage 1 streams the BFP4 LM head (half the BF8 head's bytes)
keeping each row's K best columns per core (LinearStream out_mode 6 + LM_TOPK); stage 2 (this op,
fused_decode/kernels/lm_rescore_*.cpp) rescores each row's K best candidates of the device with the BF8 weights:
one core per (row r, candidate j) reads the candidate's BF8 column tile and row r of x and computes the BF8 logit
with the stream matmul (the same LoFi custom_mm over the same K tiles as the BF8 LM head, so the logit is the BF8
head's), and writes [key, column] into list j, row r of the argmax merge's per-core lists (dflash/argmax.py).
On 1304 captured verify rows the BF8 argmax ranked at most 5th among the BFP4 logits (K = 8)."""

import os

import ttnn

from ..fused_decode.boundary import KERNEL_DIR
from ..fused_decode.stream import BFP8_TILE_BYTES, NBIAS, TILE, columns_per_bank


class LMRescore:
    def __init__(self, mesh_device, rows, K, n1, hidden, n_cols, cores_start=(0, 0)):
        """n1: stage-1 cores (lists); n_cols: the BF8 stream's (padded) columns per device."""
        self.mesh, self.M, self.K, self.n1 = mesh_device, rows, K, n1
        self.kx = hidden // TILE
        self.kt = self.kx + 1
        self.C = columns_per_bank(n_cols, mesh_device.dram_grid_size().x)
        self.col_bytes = self.kt * BFP8_TILE_BYTES
        x0, y0 = cores_start
        grid = mesh_device.compute_with_storage_grid_size()
        n = rows * K
        # DFLASH_RS_MCAST=1: row r's candidates on grid row y0 + r (x0 .. x0 + K - 1); x row r gathered once there
        self.mcast = os.environ.get("DFLASH_RS_MCAST", "0") == "1"
        # one K x rows rectangle (single-core ranges launch measurably slower under trace): candidate j of row r on
        # (x0 + j, y0 + r)
        assert K <= grid.x - x0 and rows <= grid.y - y0
        self.core_list = [ttnn.CoreCoord(x0 + i // rows, y0 + i % rows) for i in range(n)]
        self.cores = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x0 + K - 1, y0 + rows - 1))]
        )

    def __call__(self, lists1, x, w8, lists2):
        cb_x, cb_w, cb_scr, cb_meta, cb_out = 0, 1, 3, 4, 16
        tiny = ttnn.TileDescriptor(ttnn.Tile([1, TILE]))

        def cb(i, dtype, page, pages, tile=None):
            fmt = dict(buffer_index=i, data_format=dtype, page_size=page)
            if tile is not None:
                fmt["tile"] = tile
            return ttnn.CBDescriptor(
                total_size=page * pages, core_ranges=self.cores, format_descriptors=[ttnn.CBFormatDescriptor(**fmt)]
            )

        cbs = [
            cb(cb_x, ttnn.bfloat16, 2 * TILE, self.kt, tiny),
            cb(cb_w, ttnn.bfloat8_b, BFP8_TILE_BYTES, self.kt),
            cb(cb_scr, ttnn.uint32, self.n1 * self.K * 8, 1),
            cb(cb_meta, ttnn.uint32, 32, 1),
            cb(cb_out, ttnn.bfloat16, 2 * TILE, 1, tiny),
        ]
        l1_bytes = self.n1 * self.M * self.K * 8
        reader_ct = [cb_x, cb_w, cb_meta, cb_scr, self.n1, self.K, self.M, self.kx, self.col_bytes, self.C, NBIAS]
        for t in (lists1, x):
            reader_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        writer_ct = [cb_out, cb_meta, self.M, self.K * self.M * 8]
        writer_ct += ttnn.TensorAccessorArgs(lists2).get_compile_time_args()
        reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, c in enumerate(self.core_list):
            r, j = i % self.M, i // self.M
            reader_rt[c.x][c.y] = [lists1.buffer_address(), x.buffer_address(), w8.buffer_address(), r, j]
            if self.mcast:  # the row's other cores (x0 + 1 .. x0 + K - 1, same grid row) as a NOC rectangle
                a_ = self.mesh.worker_core_from_logical_core(ttnn.CoreCoord(c.x - j + 1, c.y))
                b_ = self.mesh.worker_core_from_logical_core(ttnn.CoreCoord(c.x - j + self.K - 1, c.y))
                reader_rt[c.x][c.y] = list(reader_rt[c.x][c.y]) + [a_.x, a_.y, b_.x, b_.y]
            writer_rt[c.x][c.y] = [lists2.buffer_address(), r, j]
        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "lm_rescore_reader.cpp"),
                core_ranges=self.cores,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
                defines=[(d, "1") for d in ("RS_SKIP_X", "RS_SKIP_W") if os.environ.get(d)]
                + ([("RS_MCAST", "1")] if self.mcast else []),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "lm_rescore_writer.cpp"),
                core_ranges=self.cores,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "stream_matmul_compute.cpp"),
                core_ranges=self.cores,
                compile_time_args=[cb_x, cb_w, cb_out, self.kt, 1],
                runtime_args=[],
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False
                ),
            ),
        ]
        assert l1_bytes == lists1.shape[0] * lists1.shape[-1] * 4, (l1_bytes, lists1.shape)
        sems = [ttnn.SemaphoreDescriptor(0, ttnn.CoreType.WORKER, self.cores, 0)] if self.mcast else []
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        ttnn.generic_op([lists1, x, w8, lists2], program)
