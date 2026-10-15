// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Shared-KV block attention, copies (dflash/shared_kv_attention.py), one core (BRISC). Spec batch row h = kv head h;
// its Q row-tile j packs the M rows' candidates 4 j .. 4 j + 3 (rows 8 i .. 8 i + 7 of the tile = row 4 j + i's query
// heads 8 h .. 8 h + 7: SDPA's spec_rows_per_candidate = 8). One of:
//   SPEC_POS (once per pass): pos [>= M] -> pos_spec [nkv * M / 4]: row-tile j's bound = pos[4 j + 3] (its last
//            candidate); with a window W > 0 the window's page-table view: shift = the whole blocks before position
//            pos[M - 1] + 1 - W; bounds - shift * block, and pt_spec row h = nkv * pt[shift + j] + h for j < pt_n (a
//            sequence that starts at block `shift`: the cached K are rotated, only the causal bounds depend on
//            positions). The rows then attend [shift * block, pos[r]]: W .. W + block - 1 positions.
//   SPEC_Q: q [1, >= M, heads, 64] (row r's 2 tiles: dims 0..31, 32..63; heads in tile rows) -> q_spec [1, nkv,
//           M * 8, 64].
//   SPEC_OUT: o_spec [1, nkv, M * 8, 64] -> o [1, M, heads, 64] (row r's head rows 8 h .. 8 h + 7 from batch h).
// In a 32x32 tile, rows 8 i .. 8 i + 7 of the 16-column half f are 256 contiguous bytes: face (i / 2) * 2 + f,
// offset (i % 2) * 256.
//
// runtime args: SPEC_POS [pos, pos_spec, pt, pt_spec]; SPEC_Q [q, q_spec]; SPEC_OUT [o_spec, o]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t tile = 2048;  // BF16 tile
// byte offset of rows 8 i .. 8 i + 7, columns 16 f .. 16 f + 15 in a tile
constexpr uint32_t rows8(uint32_t i, uint32_t f) { return ((i >> 1) * 2 + f) * 512 + (i & 1) * 256; }
inline void copy256(uint32_t dst, uint32_t src) {
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    for (uint32_t w = 0; w < 64; ++w) {
        d[w] = s[w];
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t M = get_compile_time_arg_val(0);
    constexpr uint32_t nkv = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(2);
    constexpr uint32_t Tt = M / 4;  // packed row-tiles per kv head
    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);
#if defined(SPEC_POS)
    constexpr uint32_t window = get_compile_time_arg_val(3);
    constexpr uint32_t block = get_compile_time_arg_val(4);
    constexpr uint32_t pt_n = get_compile_time_arg_val(5);       // pt_spec entries per kv head (window > 0)
    constexpr uint32_t pt_bytes = get_compile_time_arg_val(6);   // pt row bytes
    constexpr uint32_t pos_bytes = get_compile_time_arg_val(7);  // pos page bytes
    constexpr auto p_args = TensorAccessorArgs<8>();
    constexpr auto ps_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto pt_args = TensorAccessorArgs<ps_args.next_compile_time_args_offset()>();
    constexpr auto pts_args = TensorAccessorArgs<pt_args.next_compile_time_args_offset()>();
    const auto s_p = TensorAccessor(p_args, get_arg_val<uint32_t>(0), pos_bytes);
    const auto s_ps = TensorAccessor(ps_args, get_arg_val<uint32_t>(1), nkv * Tt * 4);
    const uint32_t pscr = scr;  // positions (128 B), pos_spec at + 128
    noc_async_read(s_p.get_noc_addr(0), pscr, pos_bytes < 128 ? pos_bytes : 128);
    noc_async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr int32_t* p = reinterpret_cast<volatile tt_l1_ptr int32_t*>(pscr);
    int32_t shift = 0;
    if constexpr (window > 0) {
        const int32_t lo = p[M - 1] + 1 - static_cast<int32_t>(window);
        shift = lo > 0 ? lo / static_cast<int32_t>(block) : 0;
        const auto s_pt = TensorAccessor(pt_args, get_arg_val<uint32_t>(2), pt_bytes);
        const auto s_pts = TensorAccessor(pts_args, get_arg_val<uint32_t>(3), pt_n * 4);
        // pt entries [shift, shift + pt_n) (the read starts at a 64-byte boundary)
        const uint32_t ptscr = scr + 256;
        const uint32_t a0 = (static_cast<uint32_t>(shift) * 4) & ~63u;
        const uint32_t skip = static_cast<uint32_t>(shift) - a0 / 4;
        uint32_t nbytes = (skip + pt_n) * 4;
        nbytes = a0 + nbytes > pt_bytes ? pt_bytes - a0 : nbytes;
        noc_async_read(s_pt.get_noc_addr(0) + a0, ptscr, (nbytes + 63) & ~63u);
        noc_async_read_barrier();
        invalidate_l1_cache();
        volatile tt_l1_ptr int32_t* src = reinterpret_cast<volatile tt_l1_ptr int32_t*>(ptscr) + skip;
        const uint32_t out = ptscr + (((skip + pt_n) * 4 + 63) & ~63u);
        volatile tt_l1_ptr int32_t* o = reinterpret_cast<volatile tt_l1_ptr int32_t*>(out);
        const uint32_t avail = nbytes / 4 - skip;
        for (uint32_t h = 0; h < nkv; ++h) {
            for (uint32_t j = 0; j < pt_n; ++j) {
                o[h * pt_n + j] = static_cast<int32_t>(nkv) * src[j < avail ? j : avail - 1] + static_cast<int32_t>(h);
            }
            noc_async_write(out + h * pt_n * 4, s_pts.get_noc_addr(h), pt_n * 4);
        }
    }
    volatile tt_l1_ptr int32_t* q = reinterpret_cast<volatile tt_l1_ptr int32_t*>(pscr + 128);
    for (uint32_t h = 0; h < nkv; ++h) {
        for (uint32_t j = 0; j < Tt; ++j) {
            q[h * Tt + j] = p[4 * j + 3] - shift * static_cast<int32_t>(block);
        }
    }
    noc_async_write(pscr + 128, s_ps.get_noc_addr(0), nkv * Tt * 4);
#elif defined(SPEC_Q)
    constexpr auto q_args = TensorAccessorArgs<3>();
    constexpr auto qs_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    const auto s_q = TensorAccessor(q_args, get_arg_val<uint32_t>(0), tile);
    const auto s_qs = TensorAccessor(qs_args, get_arg_val<uint32_t>(1), tile);
    const uint32_t dst = scr + 2 * M * tile;  // nkv * Tt * 2 packed tiles
    for (uint32_t t = 0; t < 2 * M; ++t) {
        noc_async_read(s_q.get_noc_addr(t), scr + t * tile, tile);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    for (uint32_t h = 0; h < nkv; ++h) {
        for (uint32_t r = 0; r < M; ++r) {
            const uint32_t j = r / 4, i = r % 4;
            for (uint32_t d = 0; d < 2; ++d) {
                const uint32_t qt = scr + (2 * r + d) * tile, pt = dst + ((h * Tt + j) * 2 + d) * tile;
                for (uint32_t f = 0; f < 2; ++f) {
                    copy256(pt + rows8(i, f), qt + rows8(h, f));
                }
            }
        }
    }
    for (uint32_t t = 0; t < nkv * Tt * 2; ++t) {
        noc_async_write(dst + t * tile, s_qs.get_noc_addr(t), tile);
    }
#else
    constexpr auto os_args = TensorAccessorArgs<3>();
    constexpr auto o_args = TensorAccessorArgs<os_args.next_compile_time_args_offset()>();
    const auto s_os = TensorAccessor(os_args, get_arg_val<uint32_t>(0), tile);
    const auto s_o = TensorAccessor(o_args, get_arg_val<uint32_t>(1), tile);
    const uint32_t dst = scr + nkv * Tt * 2 * tile;  // the M rows' 2 output tiles
    for (uint32_t t = 0; t < nkv * Tt * 2; ++t) {
        noc_async_read(s_os.get_noc_addr(t), scr + t * tile, tile);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    for (uint32_t r = 0; r < M; ++r) {
        const uint32_t j = r / 4, i = r % 4;
        for (uint32_t d = 0; d < 2; ++d) {
            const uint32_t ot = dst + (2 * r + d) * tile;
            for (uint32_t h = 0; h < nkv; ++h) {
                const uint32_t st = scr + ((h * Tt + j) * 2 + d) * tile;
                for (uint32_t f = 0; f < 2; ++f) {
                    copy256(ot + rows8(h, f), st + rows8(i, f));
                }
            }
            for (uint32_t f = 0; f < 2; ++f) {
                for (uint32_t h = 0; h < nkv; ++h) {
                    noc_async_write(ot + rows8(h, f), s_o.get_noc_addr(2 * r + d) + rows8(h, f), 256);
                }
            }
        }
    }
#endif
    noc_async_write_barrier();
}
