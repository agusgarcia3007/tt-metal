// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Shared-KV block attention, Q pack / output unpack (dflash/shared_kv_attention.py) as direct NOC moves of the
// 256-byte row-group segments (spec_attn_copy.cpp's SPEC_Q / SPEC_OUT layouts, no staging copies on the core):
//   SPEC_Q: segment (rows 8 h .. 8 h + 7, half f) of q tile (r, d) -> segment (rows 8 i .., half f) of q_spec tile
//           (h, j, d), r = 4 j + i: read straight into the packed tile in L1, then the packed tiles written out;
//   SPEC_OUT: segment (rows 8 i .., half f) of o_spec tile (h, j, d) -> segment (rows 8 h .., half f) of o tile
//             (r, d): read from o_spec, written into o.
// Two cores split the work (runtime arg 2: this core's share 0 / 1 of the kv heads h).
//
// runtime args: SPEC_Q [q, q_spec, part]; SPEC_OUT [o_spec, o, part]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t tile = 2048;  // BF16 tile
constexpr uint32_t rows8(uint32_t i, uint32_t f) { return ((i >> 1) * 2 + f) * 512 + (i & 1) * 256; }
}  // namespace

void kernel_main() {
    constexpr uint32_t M = get_compile_time_arg_val(0);
    constexpr uint32_t nkv = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(2);
    constexpr uint32_t Tt = M / 4;
    constexpr auto a_args = TensorAccessorArgs<3>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto s_a = TensorAccessor(a_args, get_arg_val<uint32_t>(0), tile);
    const auto s_b = TensorAccessor(b_args, get_arg_val<uint32_t>(1), tile);
    const uint32_t h = get_arg_val<uint32_t>(2);  // this core's kv head
    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);
#if defined(SPEC_Q)
    // kv head h's Tt * 2 packed tiles, assembled in L1 from the M rows' segments
    for (uint32_t r = 0; r < M; ++r) {
        const uint32_t j = r / 4, i = r % 4;
        for (uint32_t d = 0; d < 2; ++d) {
            const uint64_t src = s_a.get_noc_addr(2 * r + d);
            const uint32_t dst = scr + (j * 2 + d) * tile;
            for (uint32_t f = 0; f < 2; ++f) {
                noc_async_read(src + rows8(h, f), dst + rows8(i, f), 256);
            }
        }
    }
    noc_async_read_barrier();
    for (uint32_t t = 0; t < Tt * 2; ++t) {
        noc_async_write(scr + t * tile, s_b.get_noc_addr(h * Tt * 2 + t), tile);
    }
#else
    // kv head h's output segments: o_spec tile (h, j, d) rows 8 i .. -> o tile (r, d) rows 8 h ..
    for (uint32_t t = 0; t < Tt * 2; ++t) {
        noc_async_read(s_a.get_noc_addr(h * Tt * 2 + t), scr + t * tile, tile);
    }
    noc_async_read_barrier();
    for (uint32_t r = 0; r < M; ++r) {
        const uint32_t j = r / 4, i = r % 4;
        for (uint32_t d = 0; d < 2; ++d) {
            const uint64_t dst = s_b.get_noc_addr(2 * r + d);
            for (uint32_t f = 0; f < 2; ++f) {
                noc_async_write(scr + (j * 2 + d) * tile + rows8(i, f), dst + rows8(h, f), 256);
            }
        }
    }
#endif
    noc_async_write_barrier();
}
