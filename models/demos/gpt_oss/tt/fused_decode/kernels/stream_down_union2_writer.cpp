// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Down-projection activation gather + output scatter (BRISC, NOC1) for the union-of-experts down stream, expert-outer
// order (same as kernels/stream_down_union_writer.cpp; multi-row
// decode MoE, DFlash verify; fused_decode/union.py ExpertDownUnion).
//
// 1. U (router union buffer) -> the compute threads (mailboxes).
// 2. in0 = U segments of seg_tiles [rows, 32] tiles: segment i = slot i of the activation buffer (expert i's
//    weight-scaled SwiGLU rows, kernels/stream_gate_up_union_writer.cpp) followed by a tile whose row r holds
//    w[r][i] in its first `nbias` entries (sum_i w[r][i] * b_i through the bias rows of the weight stream).
// 3. Output tile n ([rows, 32] of the expert sum) -> bytes [64 rows n, 64 rows (n + 1)) of the row-tile partial sum
//    the boundary all-reduces (columns past the hidden size are dropped).
//
// runtime args: [act_addr, weights_addr, out_addr, col0, union_addr]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ckernel.h"
#include "ckernel_defs.h"

void kernel_main() {
    const uint32_t act_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);
    const uint32_t col0 = get_arg_val<uint32_t>(3);
    const uint32_t uni_addr = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(2);
    constexpr uint32_t seg_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t rows = get_compile_time_arg_val(4);
    constexpr uint32_t nbias = get_compile_time_arg_val(5);
    constexpr uint32_t cols = get_compile_time_arg_val(6);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t act_page_bytes = get_compile_time_arg_val(8);
    constexpr auto act_args = TensorAccessorArgs<9>();
    constexpr auto w_args = TensorAccessorArgs<act_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto uni_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr uint32_t notify_ct = uni_args.next_compile_time_args_offset();
    constexpr uint32_t notify = get_compile_time_arg_val(notify_ct);
    constexpr uint32_t notify_x = get_compile_time_arg_val(notify_ct + 1);
    constexpr uint32_t notify_y = get_compile_time_arg_val(notify_ct + 2);
    constexpr uint32_t notify_sem = get_compile_time_arg_val(notify_ct + 3);

    constexpr uint32_t tiny_bytes = 64 * rows;

    const auto s_act = TensorAccessor(act_args, act_addr, act_page_bytes);
    const auto s_w = TensorAccessor(w_args, w_addr, 64 * rows);
    const auto s_out = TensorAccessor(out_args, out_addr, 2048);
    const auto s_uni = TensorAccessor(uni_args, uni_addr, 128);

#ifdef MOE_FUSED
    // Fused MoE program (dflash/moe_fused.py): wait for the router (program semaphore 1); runtime args 5..8: the
    // counters address, router noc x, y, the gate|up core count.
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1)), 1);
    const uint64_t ctr_noc = get_noc_addr(get_arg_val<uint32_t>(6), get_arg_val<uint32_t>(7), get_arg_val<uint32_t>(5));
    const uint32_t producers = get_arg_val<uint32_t>(8);
#endif
    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);
    noc_async_read(s_uni.get_noc_addr(0), scr, 128);
    noc_async_read(s_w.get_noc_addr(0), scr + 128, 64 * rows);
    noc_async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr uint16_t* uni = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr);
    const uint32_t num_sel = uni[0];
    ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, num_sel);
    ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, num_sel);
    ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, num_sel);
    volatile tt_l1_ptr uint16_t* w = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr + 128);

    cb_reserve_back(cb_in0, num_sel * seg_tiles);
    const uint32_t in0 = get_write_ptr(cb_in0);
    for (uint32_t e = 0; e < num_sel; ++e) {
#ifdef MOE_FUSED
        // Slot e is complete once all gate|up cores counted it; push each slot as it arrives (compute waits per slot).
        volatile tt_l1_ptr uint32_t* poll = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + 128 + 64 * rows);
        do {
            noc_async_read(ctr_noc + 16 * e, scr + 128 + 64 * rows, 16);
            noc_async_read_barrier();
            invalidate_l1_cache();
        } while (poll[0] < producers);
#endif
        noc_async_read(s_act.get_noc_addr(e), in0 + e * seg_tiles * tiny_bytes, act_page_bytes);
        volatile tt_l1_ptr uint16_t* aug =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(in0 + (e * seg_tiles + seg_tiles - 1) * tiny_bytes);
        for (uint32_t i = 0; i < 32 * rows; ++i) {
            aug[i] = (i < 16 * rows && i % 16 < nbias) ? w[(i / 16) * 32 + e] : 0;
        }
#ifdef MOE_FUSED
        noc_async_read_barrier();
        cb_push_back(cb_in0, seg_tiles);
#endif
    }
#ifndef MOE_FUSED
    noc_async_read_barrier();
    cb_push_back(cb_in0, num_sel * seg_tiles);
#endif

    for (uint32_t c = 0; c < cols; ++c) {
        cb_wait_front(cb_out, 1);
        const uint32_t n = col0 + c;
        if (n < n_tiles) {
            const uint32_t src = get_read_ptr(cb_out);
            noc_async_write(src, s_out.get_noc_addr(n * tiny_bytes / 2048, (n * tiny_bytes) % 2048), tiny_bytes);
            noc_async_writes_flushed();
        }
        cb_pop_front(cb_out, 1);
    }
    noc_async_write_barrier();
    if constexpr (notify) {
        noc_semaphore_inc(get_noc_addr(notify_x, notify_y, get_semaphore(notify_sem)), 1);
        noc_async_atomic_barrier();
    }
}
