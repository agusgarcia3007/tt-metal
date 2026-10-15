// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// On-device merge of the multi-row LM head's argmax lists (LinearStream out_mode 6), one core (BRISC);
// dflash/argmax.py.
//   mode 0 (per device): the n_lists core lists ([key, index] per row, signed order keys) -> row 0 of a UINT32 tile:
//          entries 2 r, 2 r + 1 = the row's best key and global vocab id (device offset + index; ties to the lower).
//   mode 1 (after the all-gather of those tiles, one tile per device): the best over the devices per row ->
//          kind 0: tokens_out [last, best[1] .. best[rows - 1]] (last = tokens_src[0]): the drafter's 7 drafts as the
//                  verify block; kind 1: out [tokens_src[0 .. rows - 1], best[0 .. rows - 1]] (the host's one read).
//
// runtime args: mode 0: [lists_addr, dev_addr, vocab_offset]; mode 1: [gathered_addr, out_addr, tokens_src_addr]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t mode = get_compile_time_arg_val(0);
    constexpr uint32_t rows = get_compile_time_arg_val(1);
    constexpr uint32_t n_in = get_compile_time_arg_val(2);  // mode 0: core lists; mode 1: devices
    constexpr uint32_t kind = get_compile_time_arg_val(3);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(4);
    constexpr uint32_t in_page = get_compile_time_arg_val(5);
    constexpr uint32_t out_page = get_compile_time_arg_val(6);
    constexpr uint32_t src_page = get_compile_time_arg_val(7);
    constexpr auto in_args = TensorAccessorArgs<8>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr auto src_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    static_assert(2 * rows <= 16, "row 0, face 0 of a UINT32 tile holds 8 rows");

    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t arg2 = get_arg_val<uint32_t>(2);
    const auto s_in = TensorAccessor(in_args, in_addr, in_page);
    const auto s_out = TensorAccessor(out_args, out_addr, out_page);

    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);
    int32_t best_key[rows];
    uint32_t best_id[rows];
    for (uint32_t r = 0; r < rows; ++r) {
        best_key[r] = INT32_MIN;
        best_id[r] = 0xFFFFFFFFu;
    }
    auto take = [&](uint32_t r, int32_t key, uint32_t id) {
        if (key > best_key[r] || (key == best_key[r] && id < best_id[r])) {
            best_key[r] = key;
            best_id[r] = id;
        }
    };
    if constexpr (mode == 0) {
        noc_async_read(s_in.get_noc_addr(0), scr, in_page);
        noc_async_read_barrier();
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* l = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr);
        for (uint32_t c = 0; c < n_in; ++c) {
            for (uint32_t r = 0; r < rows; ++r) {
                take(r, static_cast<int32_t>(l[(c * rows + r) * 2]), arg2 + l[(c * rows + r) * 2 + 1]);
            }
        }
        // row 0 of the output tile (face 0, first 16 entries); the rest stays zero
        volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr);
        for (uint32_t i = 0; i < out_page / 4; ++i) {
            t[i] = 0;
        }
        for (uint32_t r = 0; r < rows; ++r) {
            t[2 * r] = static_cast<uint32_t>(best_key[r]);
            t[2 * r + 1] = best_id[r];
        }
        noc_async_write(scr, s_out.get_noc_addr(0), out_page);
    } else {
        const auto s_src = TensorAccessor(src_args, arg2, src_page);
        for (uint32_t g = 0; g < n_in; ++g) {
            noc_async_read(s_in.get_noc_addr(g), scr + g * in_page, 64);  // row 0, entries 0 .. 15
        }
        const uint32_t tok = scr + n_in * in_page;
        noc_async_read(s_src.get_noc_addr(0), tok, src_page);
        noc_async_read_barrier();
        invalidate_l1_cache();
        for (uint32_t g = 0; g < n_in; ++g) {
            volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + g * in_page);
            for (uint32_t r = 0; r < rows; ++r) {
                take(r, static_cast<int32_t>(t[2 * r]), t[2 * r + 1]);
            }
        }
        volatile tt_l1_ptr uint32_t* src_tok = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tok);
        volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr);  // reuse
        const uint32_t last = src_tok[0];
        if constexpr (kind == 0) {
            o[0] = last;
            for (uint32_t r = 1; r < rows; ++r) {
                o[r] = best_id[r];
            }
        } else {
            uint32_t blk[rows];
            for (uint32_t r = 0; r < rows; ++r) {
                blk[r] = src_tok[r];
            }
            for (uint32_t r = 0; r < rows; ++r) {
                o[r] = blk[r];
                o[rows + r] = best_id[r];
            }
        }
        noc_async_write(scr, s_out.get_noc_addr(0), out_page);
    }
    noc_async_write_barrier();
}
