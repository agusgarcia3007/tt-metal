// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// On-device merge of the multi-row LM head's argmax lists (LinearStream out_mode 6), one core (BRISC);
// dflash/argmax.py.
//   mode 0 (per device): the n_lists core lists ([key, index] per row, signed order keys) -> row 0 of a UINT32 tile:
//          entries 2 r, 2 r + 1 = the row's best key and global vocab id (device offset + index; ties to the lower).
//   mode 2 (ARGMAX_RING): mode 0 + a fabric exchange of the per-device bests + mode 1, one program (below);
//   mode 1 (after the all-gather of those tiles, one tile per device): the best over the devices per row ->
//          kind 0: tokens_out [last, best[1] .. best[rows - 1]] (last = tokens_src[0]): the drafter's 7 drafts as the
//                  verify block; kind 1: out [tokens_src[0 .. rows - 1], best[0 .. rows - 1]] (the host's one read).
//
// ACCEPT (mode 1, kind 1): also the round's acceptance on device (dflash/runner.py device rounds): k = the matching
// drafts (block[1 + i] == best[i]), start = the verify position tensor's first entry; writes the next round's inputs:
// drafter tokens [best[k], mask ..], positions / RoPE rows / attention bound from start' = start + k + 1, the verify
// positions start' .. start' + rows - 1 and the drafter's context rows start .. start + k (the accepted block rows).
//
// runtime args: mode 0: [lists_addr, dev_addr, vocab_offset]; mode 1: [gathered_addr, out_addr, tokens_src_addr]
//   (+ ACCEPT: [vpos, t_toks, t_pos, t_rot, t_bound, t_cpos, t_crot addresses])

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
#ifdef ARGMAX_RING
    // mode 2 (one program per device, the all-gather replaced by the fabric): this device's best per row (as mode 0)
    // -> its payload [key, id] x rows in L1; the sender kernel (boundary_sender.cpp, same core) writes it into slot
    // `dev` of every device (this one by a NOC copy) and bumps their receive semaphore; then the best over the
    // ARGMAX_RING slots (as mode 1). runtime args 3..: [vocab offset, payload, slots, receive semaphore, dev]
    if constexpr (mode == 2) {
        noc_async_read(s_in.get_noc_addr(0), scr, in_page);
        noc_async_read_barrier();
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* l = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr);
        const uint32_t voff = get_arg_val<uint32_t>(3);
        for (uint32_t c = 0; c < n_in; ++c) {
            for (uint32_t r = 0; r < rows; ++r) {
                take(r, static_cast<int32_t>(l[(c * rows + r) * 2]), voff + l[(c * rows + r) * 2 + 1]);
            }
        }
        const uint32_t payload = get_arg_val<uint32_t>(4), slots = get_arg_val<uint32_t>(5);
        constexpr uint32_t slot_bytes = 64;  // [key, id] x <= 8 rows
        volatile tt_l1_ptr uint32_t* pl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(payload);
        for (uint32_t r = 0; r < rows; ++r) {
            pl[2 * r] = static_cast<uint32_t>(best_key[r]);
            pl[2 * r + 1] = best_id[r];
        }
        // the sender (local mode) copies the payload into this device's own slot too: every device's slot arrives
        // with one increment of the receive semaphore
        noc_semaphore_inc(get_noc_addr(get_semaphore(ARGMAX_SEND_SEM)), 1);  // the sender may go
        volatile tt_l1_ptr uint32_t* recv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(6));
        noc_semaphore_wait(recv, ARGMAX_RING);
        noc_semaphore_set(recv, 0);
        invalidate_l1_cache();
        for (uint32_t g = 0; g < ARGMAX_RING; ++g) {
            volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slots + g * slot_bytes);
            for (uint32_t r = 0; r < rows; ++r) {
                take(r, static_cast<int32_t>(t[2 * r]), t[2 * r + 1]);
            }
        }
    }
#endif
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
        constexpr uint32_t n_tiles = mode == 1 ? n_in : 0;  // mode 2: merged above
        for (uint32_t g = 0; g < n_tiles; ++g) {
            noc_async_read(s_in.get_noc_addr(g), scr + g * in_page, 64);  // row 0, entries 0 .. 15
        }
        const uint32_t tok = scr + (mode == 1 ? n_in * in_page : in_page);
        noc_async_read(s_src.get_noc_addr(0), tok, src_page);
        noc_async_read_barrier();
        invalidate_l1_cache();
        for (uint32_t g = 0; g < n_tiles; ++g) {
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
#ifdef ACCEPT
        if constexpr (kind == 1) {
            constexpr uint32_t ROWS_D = 32;  // drafter input rows
            constexpr auto vp_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
            constexpr auto tk_args = TensorAccessorArgs<vp_args.next_compile_time_args_offset()>();
            constexpr auto tp_args = TensorAccessorArgs<tk_args.next_compile_time_args_offset()>();
            constexpr auto tr_args = TensorAccessorArgs<tp_args.next_compile_time_args_offset()>();
            constexpr auto tb_args = TensorAccessorArgs<tr_args.next_compile_time_args_offset()>();
            constexpr auto cp_args = TensorAccessorArgs<tb_args.next_compile_time_args_offset()>();
            constexpr auto cr_args = TensorAccessorArgs<cp_args.next_compile_time_args_offset()>();
            constexpr uint32_t ab = mode == 2 ? 8 : 3;  // the ACCEPT tensors' runtime args
            const auto s_vp = TensorAccessor(vp_args, get_arg_val<uint32_t>(ab), rows * 4);
            const auto s_tk = TensorAccessor(tk_args, get_arg_val<uint32_t>(ab + 1), ROWS_D * 4);
            const auto s_tp = TensorAccessor(tp_args, get_arg_val<uint32_t>(ab + 2), ROWS_D * 4);
            const auto s_tr = TensorAccessor(tr_args, get_arg_val<uint32_t>(ab + 3), ROWS_D * 4);
            const auto s_tb = TensorAccessor(tb_args, get_arg_val<uint32_t>(ab + 4), ROWS_D * 4);
            const auto s_cp = TensorAccessor(cp_args, get_arg_val<uint32_t>(ab + 5), ROWS_D * 4);
            const auto s_cr = TensorAccessor(cr_args, get_arg_val<uint32_t>(ab + 6), ROWS_D * 4);
            const uint32_t a = scr + 256;  // 7 x 128 B staging after the result
            noc_async_read(s_vp.get_noc_addr(0), a, rows * 4);
            noc_async_read_barrier();
            invalidate_l1_cache();
            const int32_t start = reinterpret_cast<volatile tt_l1_ptr int32_t*>(a)[0];
            uint32_t k = 0;
#ifdef ACCEPT_ROWS
            constexpr uint32_t verified = ACCEPT_ROWS;  // only rows [0, ACCEPT_ROWS) were verified
#else
            constexpr uint32_t verified = rows;
#endif
            while (k + 1 < verified && o[1 + k] == best_id[k]) {
                ++k;
            }
            const int32_t s2 = start + static_cast<int32_t>(k) + 1;
            volatile tt_l1_ptr int32_t* vp = reinterpret_cast<volatile tt_l1_ptr int32_t*>(a);
            volatile tt_l1_ptr int32_t* tk = reinterpret_cast<volatile tt_l1_ptr int32_t*>(a + 128);
            volatile tt_l1_ptr int32_t* tp = reinterpret_cast<volatile tt_l1_ptr int32_t*>(a + 256);
            volatile tt_l1_ptr int32_t* tr = reinterpret_cast<volatile tt_l1_ptr int32_t*>(a + 384);
            volatile tt_l1_ptr int32_t* tb = reinterpret_cast<volatile tt_l1_ptr int32_t*>(a + 512);
            volatile tt_l1_ptr int32_t* cp = reinterpret_cast<volatile tt_l1_ptr int32_t*>(a + 640);
            volatile tt_l1_ptr int32_t* cr = reinterpret_cast<volatile tt_l1_ptr int32_t*>(a + 768);
            for (uint32_t i = 0; i < ROWS_D; ++i) {
                const bool blk = i < rows;
                if (blk) {
                    vp[i] = s2 + static_cast<int32_t>(i);
                }
                tk[i] = i == 0 ? static_cast<int32_t>(best_id[k]) : static_cast<int32_t>(ACCEPT_MASK_ID);
                tp[i] = blk ? s2 + static_cast<int32_t>(i) : -1;
                tr[i] = blk ? s2 + static_cast<int32_t>(i) : 0;
                tb[i] = blk ? s2 + static_cast<int32_t>(rows) - 1 : -1;
                cp[i] = i <= k ? start + static_cast<int32_t>(i) : -1;
                cr[i] = i <= k ? start + static_cast<int32_t>(i) : 0;
            }
            noc_async_write(a, s_vp.get_noc_addr(0), rows * 4);
            noc_async_write(a + 128, s_tk.get_noc_addr(0), ROWS_D * 4);
            noc_async_write(a + 256, s_tp.get_noc_addr(0), ROWS_D * 4);
            noc_async_write(a + 384, s_tr.get_noc_addr(0), ROWS_D * 4);
            noc_async_write(a + 512, s_tb.get_noc_addr(0), ROWS_D * 4);
            noc_async_write(a + 640, s_cp.get_noc_addr(0), ROWS_D * 4);
            noc_async_write(a + 768, s_cr.get_noc_addr(0), ROWS_D * 4);
        }
#endif
    }
    noc_async_write_barrier();
}
