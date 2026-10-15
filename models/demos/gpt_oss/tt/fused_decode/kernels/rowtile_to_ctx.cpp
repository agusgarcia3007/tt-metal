// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// DFlash drafter context capture (dflash/verify.py): one layer's residual rows, held in the row-tile layout of
// fused_decode/boundary.py on this core (tile j = [rows, 32], faces [rows, 16]), into this device's width shard of the
// drafter's context input ([32, 5 * hidden] BF16 tile tensor, L1 interleaved, device d holding global columns
// [d * shard_cols, (d + 1) * shard_cols)): global column c = slot * hidden + v. Copies 16-value (32-byte) face-row
// segments: rows 0 .. rows - 1 of the destination tiles (faces 0 / 1).
//
// runtime args: [src_l1_addr, dst_addr, col_begin, col_end (global, this device's part of this layer), shard_col0]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t c_begin = get_arg_val<uint32_t>(2);
    const uint32_t c_end = get_arg_val<uint32_t>(3);
    const uint32_t shard_col0 = get_arg_val<uint32_t>(4);

    constexpr uint32_t rows = get_compile_time_arg_val(0);
    constexpr uint32_t hidden = get_compile_time_arg_val(1);
    constexpr uint32_t layer_col0 = get_compile_time_arg_val(2);  // slot * hidden
    constexpr auto dst_args = TensorAccessorArgs<3>();
    const auto s_dst = TensorAccessor(dst_args, dst_addr, 2048);

    for (uint32_t c = c_begin; c < c_end; c += 16) {
        const uint32_t v = c - layer_col0;
        const uint32_t local = c - shard_col0;
        const uint32_t src_seg = src + (v / 32) * rows * 64 + ((v % 32) / 16) * rows * 32;
        const uint64_t dst_tile = s_dst.get_noc_addr(local / 32);
        const uint32_t dst_face = ((local % 32) / 16) * 512;
        for (uint32_t r = 0; r < rows; ++r) {
            noc_async_write(src_seg + r * 32, dst_tile + dst_face + r * 32, 32);
        }
    }
    noc_async_write_barrier();
}
