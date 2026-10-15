// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// K / V rows of one sequence -> its paged BF16 cache, one core (BRISC), for the DFlash drafter (dflash/draft.py).
//
// Row r (r < rows, position pos[r]; pos < 0: skipped) of the K and V head tensors ([1, 32 users, 32 head rows,
// head_dim] BF16, height-sharded one user per core: user r's two 32x32 tiles at the same L1 address on core r, NOC
// coordinates in the runtime args) goes to row pos % 32 of tile ((blk * kv_heads + h) * (block / 32) + (pos % block) /
// 32) * head_tiles + t of the paged cache (blk = page_table[0][pos / block]), for every kv head h and head tile t. The
// valid rows have consecutive positions, so they touch at most two 32-row cache tiles per (h, t): those are read once,
// patched, and written back (whole tiles: DRAM accesses are 64-byte aligned, and one core avoids read-modify-write
// races between rows sharing a tile, which ttnn's paged_update_cache has).
//
// head_offset: the cache's kv head h is head row head_offset + h of the source tiles (several layers' K / V heads in
// one tensor).
//
// runtime args: [k_src, v_src, k_cache, v_cache, pos, page_table, (noc_x, noc_y) x rows]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t k_src = get_arg_val<uint32_t>(0);
    const uint32_t v_src = get_arg_val<uint32_t>(1);
    const uint32_t k_cache = get_arg_val<uint32_t>(2);
    const uint32_t v_cache = get_arg_val<uint32_t>(3);
    const uint32_t pos_addr = get_arg_val<uint32_t>(4);
    const uint32_t pt_addr = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_scr = get_compile_time_arg_val(0);
    constexpr uint32_t rows = get_compile_time_arg_val(1);
    constexpr uint32_t kv_heads = get_compile_time_arg_val(2);
    constexpr uint32_t head_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t block = get_compile_time_arg_val(4);
    constexpr uint32_t pt_row_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t head_offset = get_compile_time_arg_val(6);
    constexpr auto kc_args = TensorAccessorArgs<7>();
    constexpr auto vc_args = TensorAccessorArgs<kc_args.next_compile_time_args_offset()>();
    constexpr auto pos_args = TensorAccessorArgs<vc_args.next_compile_time_args_offset()>();
    constexpr auto pt_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();

    constexpr uint32_t tile = 2048;
    constexpr uint32_t src_tiles = 2 * rows * head_tiles;          // K then V, row-major (row, t)
    constexpr uint32_t dst_tiles = 2 * 2 * kv_heads * head_tiles;  // (kv, group, h, t)
    const auto s_kc = TensorAccessor(kc_args, k_cache, tile);
    const auto s_vc = TensorAccessor(vc_args, v_cache, tile);
    const auto s_pos = TensorAccessor(pos_args, pos_addr, 128);
    const auto s_pt = TensorAccessor(pt_args, pt_addr, pt_row_bytes);

    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);
    const uint32_t src = scr + 256;
    const uint32_t dst = src + src_tiles * tile;

    noc_async_read(s_pos.get_noc_addr(0), scr, 128);
    noc_async_read_barrier();
    invalidate_l1_cache();
    int32_t pos[rows];
    uint32_t first = rows, last = 0;
    for (uint32_t r = 0; r < rows; ++r) {
        pos[r] = reinterpret_cast<volatile tt_l1_ptr int32_t*>(scr)[r];
        if (pos[r] >= 0) {
            first = first == rows ? r : first;
            last = r;
        }
    }
    if (first == rows) {
        return;
    }
    const uint32_t p0 = pos[first], p1 = pos[last];
    const uint32_t bi0 = p0 / block, bi1 = p1 / block;
    noc_async_read(s_pt.get_noc_addr(0) + ((bi0 * 4) & ~63u), scr + 128, 64);
    noc_async_read(s_pt.get_noc_addr(0) + ((bi1 * 4) & ~63u), scr + 192, 64);
    // Source rows (each valid row's K and V tiles from its user core).
    for (uint32_t r = first; r <= last; ++r) {
        if (pos[r] < 0) {
            continue;
        }
        const uint32_t x = get_arg_val<uint32_t>(6 + 2 * r), y = get_arg_val<uint32_t>(7 + 2 * r);
        for (uint32_t t = 0; t < head_tiles; ++t) {
            noc_async_read(get_noc_addr(x, y, k_src + t * tile), src + (r * head_tiles + t) * tile, tile);
            noc_async_read(get_noc_addr(x, y, v_src + t * tile), src + ((rows + r) * head_tiles + t) * tile, tile);
        }
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t blk0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + 128)[bi0 & 15];
    const uint32_t blk1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + 192)[bi1 & 15];
    const uint32_t tr0 = (p0 % block) / 32, tr1 = (p1 % block) / 32;
    const uint32_t groups = (bi0 != bi1 || tr0 != tr1) ? 2 : 1;
    auto cache_tile = [&](uint32_t g, uint32_t h, uint32_t t) -> uint32_t {
        const uint32_t blk = g == 0 ? blk0 : blk1, tr = g == 0 ? tr0 : tr1;
        return ((blk * kv_heads + h) * (block / 32) + tr) * head_tiles + t;
    };
    auto dst_l1 = [&](uint32_t kv, uint32_t g, uint32_t h, uint32_t t) -> uint32_t {
        return dst + (((kv * 2 + g) * kv_heads + h) * head_tiles + t) * tile;
    };
    for (uint32_t g = 0; g < groups; ++g) {
        for (uint32_t h = 0; h < kv_heads; ++h) {
            for (uint32_t t = 0; t < head_tiles; ++t) {
                noc_async_read(s_kc.get_noc_addr(cache_tile(g, h, t)), dst_l1(0, g, h, t), tile);
                noc_async_read(s_vc.get_noc_addr(cache_tile(g, h, t)), dst_l1(1, g, h, t), tile);
            }
        }
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    // Patch: source row h of the user tile -> cache row pos % 32 (16 values in each of the two faces of its half).
    for (uint32_t r = first; r <= last; ++r) {
        if (pos[r] < 0) {
            continue;
        }
        const uint32_t p = pos[r];
        const uint32_t g = (p / block == bi0 && (p % block) / 32 == tr0) ? 0 : 1;
        const uint32_t rr = p % 32;
        const uint32_t dst_off = (rr < 16 ? 0 : 1024) + (rr % 16) * 32;
        for (uint32_t kv = 0; kv < 2; ++kv) {
            for (uint32_t h = 0; h < kv_heads; ++h) {
                const uint32_t hs = head_offset + h;
                const uint32_t src_off = (hs < 16 ? 0 : 1024) + (hs % 16) * 32;
                for (uint32_t t = 0; t < head_tiles; ++t) {
                    const uint32_t s = src + ((kv * rows + r) * head_tiles + t) * tile + src_off;
                    const uint32_t d = dst_l1(kv, g, h, t) + dst_off;
                    for (uint32_t f = 0; f < 2; ++f) {
                        volatile tt_l1_ptr uint32_t* sw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s + f * 512);
                        volatile tt_l1_ptr uint32_t* dw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(d + f * 512);
                        for (uint32_t w = 0; w < 8; ++w) {
                            dw[w] = sw[w];
                        }
                    }
                }
            }
        }
    }
    for (uint32_t g = 0; g < groups; ++g) {
        for (uint32_t h = 0; h < kv_heads; ++h) {
            for (uint32_t t = 0; t < head_tiles; ++t) {
                noc_async_write(dst_l1(0, g, h, t), s_kc.get_noc_addr(cache_tile(g, h, t)), tile);
                noc_async_write(dst_l1(1, g, h, t), s_vc.get_noc_addr(cache_tile(g, h, t)), tile);
            }
        }
    }
    noc_async_write_barrier();
}
