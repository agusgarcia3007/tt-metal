// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Two-stage DFlash LM head, stage 2 reader (NCRISC) (dflash/lm_rescore.py): this core rescores candidate j of row r.
// 1. The stage-1 (BFP4) lists of row r: n1 cores x K [key, index] entries sorted descending -> this row's K best
//    (ties: the lower index); candidate j = column c of this device.
// 2. c's BF8 weight column tile (tile c / 32 of the BF8 LM-head stream: bank (c / 32) / C, slot (c / 32) % C, kt
//    tiles contiguous) -> cb_w; row r of the normed x (row-tile layout, [rows, 32] tiles: face f of tile t at byte
//    512 t + 256 f, row r at +32 r) as kx 1x32 tiles + the bias tile (1.0 in the first nbias entries) -> cb_x.
// 3. [lane, c] -> cb_meta for the writer.
//
// runtime args: [lists1_addr, x_addr, w_addr, r, j]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t lists_addr = get_arg_val<uint32_t>(0);
    const uint32_t x_addr = get_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_arg_val<uint32_t>(2);
    const uint32_t r = get_arg_val<uint32_t>(3);
    const uint32_t j = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_meta = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(3);
    constexpr uint32_t n1 = get_compile_time_arg_val(4);
    constexpr uint32_t K = get_compile_time_arg_val(5);
    constexpr uint32_t rows = get_compile_time_arg_val(6);
    constexpr uint32_t kx = get_compile_time_arg_val(7);
    constexpr uint32_t col_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t C = get_compile_time_arg_val(9);
    constexpr uint32_t nbias = get_compile_time_arg_val(10);
    constexpr auto l_args = TensorAccessorArgs<11>();
    constexpr auto x_args = TensorAccessorArgs<l_args.next_compile_time_args_offset()>();
    constexpr uint32_t tiny = 64;                                // one 1x32 BF16 tile
    constexpr uint32_t row_tile = 64 * rows;                     // one [rows, 32] BF16 tile of x
    const auto s_l = TensorAccessor(l_args, lists_addr, K * 8);  // one page per (stage-1 core, row)
    const auto s_x = TensorAccessor(x_args, x_addr, 2048);

    // 1. this row's stage-1 entries
    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);
    for (uint32_t i = 0; i < n1; ++i) {
        noc_async_read(s_l.get_noc_addr(i * rows + r), scr + i * K * 8, K * 8);
    }
    // 2b. x row r (independent of the selection)
    cb_reserve_back(cb_x, kx + 1);
    const uint32_t xl = get_write_ptr(cb_x);
#ifdef RS_MCAST
    // x row r once per row: candidate 0's core gathers it and multicasts it to the row's other K - 1 cores (one grid
    // row, runtime args 5..8: their NOC rectangle), then sets their program semaphore 0
    if (j == 0) {
#endif
#ifndef RS_SKIP_X  // (debugging: timing without the x reads)
        for (uint32_t t = 0; t < kx; ++t) {
            const uint32_t b = t * row_tile;
            const uint64_t src = s_x.get_noc_addr(b / 2048) + (b % 2048);
            noc_async_read(src + 32 * r, xl + t * tiny, 32);
            noc_async_read(src + row_tile / 2 + 32 * r, xl + t * tiny + 32, 32);
        }
#endif
#ifdef RS_MCAST
        noc_async_read_barrier();
        const uint32_t mx0 = get_arg_val<uint32_t>(5), my0 = get_arg_val<uint32_t>(6);
        const uint32_t mx1 = get_arg_val<uint32_t>(7), my1 = get_arg_val<uint32_t>(8);
        noc_async_write_multicast(xl, get_noc_multicast_addr(mx0, my0, mx1, my1, xl), kx * tiny, K - 1);
        noc_async_write_barrier();
        volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
        sem[0] = 1;
        noc_semaphore_set_multicast(
            get_semaphore(0), get_noc_multicast_addr(mx0, my0, mx1, my1, get_semaphore(0)), K - 1);
        noc_async_write_barrier();
    } else {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0)), 1);
    }
#endif
    volatile tt_l1_ptr uint16_t* ones = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(xl + kx * tiny);
    for (uint32_t i = 0; i < 32; ++i) {
        ones[i] = i < nbias ? 0x3F80 : 0;
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr int32_t* e = reinterpret_cast<volatile tt_l1_ptr int32_t*>(scr);
    int32_t bk[K];
    uint32_t bi[K];
    for (uint32_t k = 0; k < K; ++k) {
        bk[k] = INT32_MIN;
        bi[k] = 0xFFFFFFFFu;
    }
    for (uint32_t q = 0; q < n1 * K; ++q) {
        const int32_t key = e[2 * q];
        const uint32_t idx = static_cast<uint32_t>(e[2 * q + 1]);
        if (key > bk[K - 1] || (key == bk[K - 1] && idx < bi[K - 1])) {
            uint32_t p = K - 1;
            while (p > 0 && (key > bk[p - 1] || (key == bk[p - 1] && idx < bi[p - 1]))) {
                bk[p] = bk[p - 1];
                bi[p] = bi[p - 1];
                --p;
            }
            bk[p] = key;
            bi[p] = idx;
        }
    }
    const uint32_t c = bi[j];
    // 2a. the candidate's BF8 column tile
    const uint32_t n = c / 32;
    cb_reserve_back(cb_w, kx + 1);
#ifndef RS_SKIP_W  // (debugging: timing without the weight read)
    noc_async_read(
        get_noc_addr_from_bank_id<true>(n / C, w_addr + (n % C) * col_bytes), get_write_ptr(cb_w), col_bytes);
#endif
    cb_reserve_back(cb_meta, 1);
    volatile tt_l1_ptr uint32_t* m = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_meta));
    m[0] = c % 32;
    m[1] = c;
    cb_push_back(cb_meta, 1);
    cb_push_back(cb_x, kx + 1);
    noc_async_read_barrier();
    cb_push_back(cb_w, kx + 1);
}
