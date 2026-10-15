// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// DFlash drafter q / k prep, compute (TRISC) (dflash/qk_prep.py): per head (tile rows) RMSNorm over head_dim (64 =
// two tiles) with the q_norm / k_norm weight, then RoPE (rotary_embedding_llama decode form, per 32-column tile:
// y = x * cos + (x . T) * sin), packed into this core's shards of the Q / K outputs. Both RMSNorm reductions run
// before any row-broadcast op.

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_q = get_compile_time_arg_val(0);
    constexpr uint32_t cb_k = get_compile_time_arg_val(1);
    constexpr uint32_t cb_g = get_compile_time_arg_val(2);
    constexpr uint32_t cb_cs = get_compile_time_arg_val(3);  // cos 0, 1, sin 0, 1
    constexpr uint32_t cb_t = get_compile_time_arg_val(4);
    constexpr uint32_t cb_scl = get_compile_time_arg_val(5);
    constexpr uint32_t cb_sq = get_compile_time_arg_val(6);
    constexpr uint32_t cb_rs = get_compile_time_arg_val(7);  // 2 tiles: q, k
    constexpr uint32_t cb_t1 = get_compile_time_arg_val(8);
    constexpr uint32_t cb_xn = get_compile_time_arg_val(9);
    constexpr uint32_t cb_t2 = get_compile_time_arg_val(10);
    constexpr uint32_t cb_qo = get_compile_time_arg_val(11);  // Q output shard
    constexpr uint32_t cb_ko = get_compile_time_arg_val(12);  // K output shard
    constexpr uint32_t inv_n = get_compile_time_arg_val(13);  // float bits of 1 / 64
    constexpr uint32_t eps = get_compile_time_arg_val(14);

    // A. 1 / rms per head row, q then k
    cb_wait_front(cb_scl, 1);
    cb_reserve_back(cb_rs, 2);
    for (uint32_t x = 0; x < 2; ++x) {
        const uint32_t cb_x = x == 0 ? cb_q : cb_k;
        cb_wait_front(cb_x, 2);
        compute_kernel_hw_startup(cb_x, cb_x, cb_sq);
        tile_regs_acquire();
        mul_init(cb_x, cb_x);
        mul_tiles(cb_x, cb_x, 0, 0, 0);
        mul_tiles(cb_x, cb_x, 1, 1, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        tile_regs_commit();
        cb_reserve_back(cb_sq, 1);
        tile_regs_wait();
        pack_reconfig_data_format(cb_sq);
        pack_tile(0, cb_sq);
        tile_regs_release();
        cb_push_back(cb_sq, 1);
        cb_wait_front(cb_sq, 1);
        compute_kernel_hw_startup(cb_scl, cb_sq, cb_rs);  // REDUCE_ROW: scaler to SrcA, data to SrcB
        tile_regs_acquire();
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_scl, cb_rs);
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_scl, 0, 0, 0);
        reduce_uninit(cb_sq);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, inv_n);
        add_unary_tile(0, eps);
        rsqrt_tile_init();
        rsqrt_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_rs);
        pack_tile<true>(0, cb_rs, x);
        tile_regs_release();
        cb_pop_front(cb_sq, 1);
    }
    cb_push_back(cb_rs, 2);

    // B. per tensor and 32-column tile: xn = (x * rs) * g; y = xn * cos + (xn . T) * sin
    cb_wait_front(cb_rs, 2);
    cb_wait_front(cb_g, 4);
    cb_wait_front(cb_cs, 4);
    cb_wait_front(cb_t, 1);
    for (uint32_t x = 0; x < 2; ++x) {
        const uint32_t cb_x = x == 0 ? cb_q : cb_k;
        const uint32_t cb_o = x == 0 ? cb_qo : cb_ko;
        cb_reserve_back(cb_o, 2);
        for (uint32_t d = 0; d < 2; ++d) {
            compute_kernel_hw_startup(cb_x, cb_rs, cb_t1);
            tile_regs_acquire();
            mul_bcast_cols_init(cb_x, cb_rs);
            mul_tiles_bcast_cols(cb_x, cb_rs, d, x, 0);
            tile_regs_commit();
            cb_reserve_back(cb_t1, 1);
            tile_regs_wait();
            pack_reconfig_data_format(cb_t1);
            pack_tile(0, cb_t1);
            tile_regs_release();
            cb_push_back(cb_t1, 1);

            cb_wait_front(cb_t1, 1);
            compute_kernel_hw_startup(cb_t1, cb_g, cb_xn);
            tile_regs_acquire();
            mul_bcast_rows_init(cb_t1, cb_g);
            mul_tiles_bcast_rows(cb_t1, cb_g, 0, 2 * x + d, 0);
            tile_regs_commit();
            cb_reserve_back(cb_xn, 1);
            tile_regs_wait();
            pack_reconfig_data_format(cb_xn);
            pack_tile(0, cb_xn);
            tile_regs_release();
            cb_push_back(cb_xn, 1);
            cb_pop_front(cb_t1, 1);

            cb_wait_front(cb_xn, 1);
            compute_kernel_hw_startup(cb_t, cb_xn, cb_t2);  // matmul: in1 to SrcA, in0 to SrcB
            tile_regs_acquire();
            matmul_init(cb_xn, cb_t, 0);
            matmul_tiles(cb_xn, cb_t, 0, 0, 0);
            tile_regs_commit();
            cb_reserve_back(cb_t2, 1);
            tile_regs_wait();
            pack_reconfig_data_format(cb_t2);
            pack_tile(0, cb_t2);
            tile_regs_release();
            cb_push_back(cb_t2, 1);

            cb_wait_front(cb_t2, 1);
            compute_kernel_hw_startup(cb_t2, cb_cs, cb_o);
            tile_regs_acquire();
            mul_bcast_rows_init(cb_t2, cb_cs);
            mul_tiles_bcast_rows(cb_t2, cb_cs, 0, 2 + d, 0);  // (x . T) * sin
            mul_tiles_bcast_rows(cb_xn, cb_cs, 0, d, 1);      // x * cos
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_o);
            pack_tile<true>(0, cb_o, d);
            tile_regs_release();
            cb_pop_front(cb_t2, 1);
            cb_pop_front(cb_xn, 1);
        }
        cb_push_back(cb_o, 2);
        cb_pop_front(cb_x, 2);
    }
}
