// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Union-of-experts down projection (TRISC) for the multi-row decode MoE (DFlash verify; fused_decode/union.py
// ExpertDownUnion): per output column, out = in0 . W over K = U segments, streamed in blocks of `chunk` segments and
// accumulated in DST (custom_mm, [rows, 32] in0 tiles, LoFi, FP32 accumulation; the split accumulators are merged on
// the last block).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/pack.h"

void kernel_main() {
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t seg_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t cols = get_compile_time_arg_val(4);
    constexpr uint32_t chunk = get_compile_time_arg_val(5);

    constexpr bool transpose = false;
    constexpr bool split_acc = true;
    constexpr bool dense_packing = false;
    custom_mm_block_init<transpose, split_acc, dense_packing>(cb_in0, cb_w, cb_out);

    uint32_t num_sel = 0;
    UNPACK(num_sel = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    MATH(num_sel = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    PACK(num_sel = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    const uint32_t blocks = (num_sel + chunk - 1) / chunk;

    cb_wait_front(cb_in0, num_sel * seg_tiles);
    for (uint32_t c = 0; c < cols; ++c) {
        tile_regs_acquire();
        for (uint32_t b = 0; b < blocks; ++b) {
            const uint32_t e0 = b * chunk;
            const uint32_t n = num_sel - e0 < chunk ? num_sel - e0 : chunk;
            cb_wait_front(cb_w, chunk * seg_tiles);
            if (b + 1 == blocks) {
                custom_mm_block<true>(cb_in0, cb_w, e0 * seg_tiles, 0, 0, n * seg_tiles);
            } else {
                custom_mm_block<false>(cb_in0, cb_w, e0 * seg_tiles, 0, 0, n * seg_tiles);
            }
            cb_pop_front(cb_w, chunk * seg_tiles);
        }
        tile_regs_commit();
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }
    cb_pop_front(cb_in0, num_sel * seg_tiles);
    custom_mm_block_uninit<dense_packing>();
}
