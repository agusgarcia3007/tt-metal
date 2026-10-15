// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Decode layer boundary, receiver (NCRISC) (fused_decode/boundary.py: DecodeBoundary).
//
// Feeds the boundary compute: the RMSNorm weight (`tiles` BF16 pages read from DRAM) and the reduce scaler (both
// once per call), the residual stream (cb_res) and, once every device's partial sum has arrived (the receive
// semaphore reaches `expected`: the local copy + the remote packets), the `slots` x `tiles` receive buffer (cb_recv).
// The semaphore is reset for the next use of this boundary before the slots are published. expected = 0: cb_recv
// holds an already all-reduced sum (no wait).
//
// residual source (res_mode):
//   0: cb_res is backed by the flat residual tensor (published as is);
//   1: entry: the residual is the [1, 1, 1, hidden] BF16 row-major embedding (one page), copied into cb_res and
//      zero padded to `tiles` pages.
//
// runtime args: [gamma_addr, sem_addr, res_src_addr]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t gamma_addr = get_arg_val<uint32_t>(0);
    const uint32_t sem_addr = get_arg_val<uint32_t>(1);
    const uint32_t res_src_addr = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_res = get_compile_time_arg_val(0);
    constexpr uint32_t cb_recv = get_compile_time_arg_val(1);
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(3);
    constexpr uint32_t tiles = get_compile_time_arg_val(4);
    constexpr uint32_t slots = get_compile_time_arg_val(5);
    constexpr uint32_t expected = get_compile_time_arg_val(6);
    constexpr uint32_t res_mode = get_compile_time_arg_val(7);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(8);  // hidden * 2
    constexpr uint32_t do_norm = get_compile_time_arg_val(9);
    // rows > 1 (multi-row decode): `tiles` pages hold `rows` token rows in the row-tile layout of
    // fused_decode/boundary.py (gamma in that layout too); cb_p gets the row-sum matrix of the multi-row RMSNorm.
    constexpr uint32_t rows = get_compile_time_arg_val(10);
    constexpr uint32_t cb_p = get_compile_time_arg_val(11);
    constexpr auto gamma_args = TensorAccessorArgs<12>();
    constexpr auto res_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

    constexpr uint32_t page = 2048;

    if constexpr (do_norm) {
        cb_reserve_back(cb_gamma, tiles);
        if constexpr (rows == 1) {
            const auto s_gamma = TensorAccessor(gamma_args, gamma_addr, tiles * page);
            noc_async_read(s_gamma.get_noc_addr(0), get_write_ptr(cb_gamma), tiles * page);
        } else {
            // Multi-row gamma: one 2 KB page per tile, interleaved over the DRAM banks (parallel reads).
            const auto s_gamma = TensorAccessor(gamma_args, gamma_addr, page);
            for (uint32_t t = 0; t < tiles; ++t) {
                noc_async_read(s_gamma.get_noc_addr(t), get_write_ptr(cb_gamma) + t * page, page);
            }
        }
    }

    if constexpr (res_mode >= 1) {  // 1: entry embedding rows; 2: zero residual (zero-filled only)
        const auto s_res = TensorAccessor(res_args, res_src_addr, row_bytes);
        cb_reserve_back(cb_res, tiles);
        const uint32_t dst = get_write_ptr(cb_res);
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        for (uint32_t z = 0; z < tiles * page; z += MEM_ZEROS_SIZE) {
            const uint32_t n = tiles * page - z < MEM_ZEROS_SIZE ? tiles * page - z : MEM_ZEROS_SIZE;
            noc_async_read(zeros, dst + z, n);
        }
        noc_async_read_barrier();
        if constexpr (res_mode == 2) {
        } else if constexpr (rows == 1) {
            noc_async_read(s_res.get_noc_addr(0), dst, row_bytes);
        } else {
            // Embedding row r (page r of the [1, 1, rows, hidden] row-major input): its values 32 j .. 32 j + 31 go to
            // row r of row tile j (16 in face 0, 16 in face 1 of the [rows, 32] tile).
            for (uint32_t r = 0; r < rows; ++r) {
                const uint64_t src = s_res.get_noc_addr(r);
                for (uint32_t j = 0; j < row_bytes / 64; ++j) {
                    const uint32_t t = dst + j * rows * 64;
                    noc_async_read(src + j * 64, t + r * 32, 32);
                    noc_async_read(src + j * 64 + 32, t + rows * 32 + r * 32, 32);
                }
            }
        }
    } else {
        cb_reserve_back(cb_res, tiles);
    }

    if constexpr (do_norm) {
        // Reduce scaler: 1.0 in the first row of each face of a BF16 tile (scalar sum of all 1024 values).
        cb_reserve_back(cb_scaler, 1);
        volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_scaler));
        for (uint32_t i = 0; i < page / 4; ++i) {
            s[i] = 0;
        }
        for (uint32_t f = 0; f < 4; ++f) {
            for (uint32_t i = 0; i < 8; ++i) {
                s[f * 128 + i] = 0x3F803F80;
            }
        }
        cb_push_back(cb_scaler, 1);
        if constexpr (rows > 1) {
            // P[R][R'] = 1 iff R = R' (mod rows), BF16 32x32 tile (faces of 16x16, row-major).
            cb_reserve_back(cb_p, 1);
            volatile tt_l1_ptr uint32_t* pm = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_p));
            for (uint32_t R = 0; R < 32; ++R) {
                for (uint32_t C = 0; C < 32; C += 2) {  // two BF16 values per word (C even, C + 1 odd)
                    const uint32_t lo = (R % rows == C % rows) ? 0x3F80u : 0u;
                    const uint32_t hi = (R % rows == (C + 1) % rows) ? 0x3F80u : 0u;
                    pm[(((R / 16) * 2 + C / 16) * 256 + (R % 16) * 16 + C % 16) / 2] = lo | (hi << 16);
                }
            }
            cb_push_back(cb_p, 1);
        }
    }

    // The residual and the receive slots go to compute first (the sum does not need gamma); gamma's reads complete
    // while compute adds.
    if constexpr (res_mode >= 1) {
        noc_async_read_barrier();  // the entry embedding rows (gamma may still be in flight: one more barrier below)
    }
    cb_push_back(cb_res, tiles);
    if constexpr (slots > 0) {
        if constexpr (expected > 0) {
            DeviceZoneScopedN("BND_R_WAIT");
            volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
            noc_semaphore_wait_min(sem, expected);
            noc_semaphore_set(sem, 0);
        }
        cb_reserve_back(cb_recv, slots * tiles);
        cb_push_back(cb_recv, slots * tiles);
    }
    noc_async_read_barrier();
    if constexpr (do_norm) {
        cb_push_back(cb_gamma, tiles);
    }
}
