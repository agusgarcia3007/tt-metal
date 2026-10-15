// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Union-of-experts down projection, expert-outer order (TRISC) (fused_decode/union.py ExpertDownUnion,
// expert_outer=True): one DST acquire holds this core's `cols` output tiles; per union expert e (one weight block of
// cols segments, kernels/stream_down_union2_reader.cpp) every column c accumulates in0 segment e . W_e[:, c] into DST c
// (custom_mm, [rows, 32] in0 tiles, LoFi, FP32 accumulation; split accumulators merged with the last expert).

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
    static_assert(cols <= 4, "the output tiles of a core stay in DST (4 FP32 tiles)");

    constexpr bool transpose = false;
    constexpr bool split_acc = true;
    constexpr bool dense_packing = false;
    custom_mm_block_init<transpose, split_acc, dense_packing>(cb_in0, cb_w, cb_out);

    uint32_t num_sel = 0;
    UNPACK(num_sel = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    MATH(num_sel = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    PACK(num_sel = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));

    cb_wait_front(cb_in0, num_sel * seg_tiles);
    tile_regs_acquire();
    for (uint32_t e = 0; e < num_sel; ++e) {
        cb_wait_front(cb_w, cols * seg_tiles);
        for (uint32_t c = 0; c < cols; ++c) {
            if (e + 1 == num_sel) {
                custom_mm_block<true>(cb_in0, cb_w, e * seg_tiles, c * seg_tiles, c, seg_tiles);
            } else {
                custom_mm_block<false>(cb_in0, cb_w, e * seg_tiles, c * seg_tiles, c, seg_tiles);
            }
        }
        cb_pop_front(cb_w, cols * seg_tiles);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t c = 0; c < cols; ++c) {
        cb_reserve_back(cb_out, 1);
        pack_tile(c, cb_out);
        cb_push_back(cb_out, 1);
    }
    tile_regs_release();
    cb_pop_front(cb_in0, num_sel * seg_tiles);
    custom_mm_block_uninit<dense_packing>();
}
