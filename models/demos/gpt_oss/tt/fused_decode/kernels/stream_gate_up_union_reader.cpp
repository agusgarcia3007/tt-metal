// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Union-of-experts gate|up weight streamer (NCRISC, NOC0) for the multi-row decode MoE (DFlash verify;
// fused_decode/union.py ExpertGateUpUnion). As kernels/stream_gate_up_reader.cpp, but the experts are the runtime
// union of the rows' routed experts: the router's union buffer holds [U, id_0 .. id_{U-1}] (UINT16).
//
// runtime args: [w_addr, union_addr, bank_id, vc, reader_offset_bytes]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#ifdef MOE_PRED
#include "moe_order.hpp"
#endif

void kernel_main() {
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t uni_addr = get_arg_val<uint32_t>(1);
    const uint32_t bank_id = get_arg_val<uint32_t>(2);
    const uint32_t vc = get_arg_val<uint32_t>(3);
    const uint32_t reader_offset = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_idx = get_compile_time_arg_val(1);
    constexpr uint32_t kt = get_compile_time_arg_val(2);
    constexpr uint32_t cols = get_compile_time_arg_val(3);
    constexpr uint32_t expert_stride = get_compile_time_arg_val(4);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(6);
    constexpr auto uni_args = TensorAccessorArgs<7>();

    constexpr uint32_t col_bytes = kt * tile_bytes;
    // Packets of up to 16 KB regardless of tile boundaries (the column is one contiguous block in DRAM and in the
    // ring): full packets then a tail.
    constexpr uint32_t pkt = NOC_MAX_BURST_SIZE < col_bytes ? NOC_MAX_BURST_SIZE : col_bytes;
    constexpr uint32_t full_pkts = col_bytes / pkt;
    constexpr uint32_t tail = col_bytes - full_pkts * pkt;
    (void)page_bytes;
    const uint32_t cb_start = get_write_ptr(cb_w);  // the ring's first column (nothing reserved yet)
#ifdef MOE_PRED
    // runtime args 5..7: hub (router) noc x, y, this layer's prediction address. Prefetch the predicted experts'
    // columns into MOE_PRED_CB while the boundary and the router run.
    constexpr uint32_t P = MOE_PRED;
    uint32_t pred[P];
    cb_reserve_back(cb_idx, 1);
    moe_order::read_pred<P>(
        get_arg_val<uint32_t>(5),
        get_arg_val<uint32_t>(6),
        get_arg_val<uint32_t>(7),
        get_write_ptr(cb_idx) + 128,
        MOE_PRED_SEM,
        pred);
    cb_reserve_back(MOE_PRED_CB, P * cols * kt);
    const uint32_t pre = get_write_ptr(MOE_PRED_CB);
    for (uint32_t k = 0; k < P; ++k) {
        const uint64_t src = get_noc_addr_from_bank_id<true>(bank_id, w_addr + pred[k] * expert_stride + reader_offset);
        noc_async_read(src, pre + k * cols * col_bytes, cols * col_bytes);
    }
#endif

#if defined(MOE_FUSED) && !defined(MOE_NO_ROUTER)
    // fused MoE program (dflash/moe_fused.py): the router's union is written (program semaphore 1)
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1)), 1);
#endif
    cb_reserve_back(cb_idx, 1);
    const uint32_t idx_l1 = get_write_ptr(cb_idx);
    const auto s_uni = TensorAccessor(uni_args, uni_addr, 128);
    noc_async_read(s_uni.get_noc_addr(0), idx_l1, 128);
    noc_async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr uint16_t* uni = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(idx_l1);
    const uint32_t num_sel = uni[0];
    uint32_t order[32];
    uint32_t e0 = 0;  // first union slot streamed from DRAM (in `order`)
#ifdef MOE_PRED
    uint32_t kept_pred[P];
    const uint32_t kept = moe_order::make_order<P>(pred, uni, order, kept_pred);
    noc_async_read_barrier();                  // the prefetch
    cb_push_back(MOE_PRED_CB, P * cols * kt);  // compute takes the routed ones, skips the others
    DeviceTimestampedData("MOE_T", 402);
    e0 = kept;
#else
    for (uint32_t e = 0; e < num_sel; ++e) {
        order[e] = e;
    }
#endif
    const uint64_t w_base = get_noc_addr_from_bank_id<true>(bank_id, w_addr);
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    noc_async_read_one_packet_set_state<true>(w_base, pkt, vc);

    // Triple-buffered columns, one extra column in flight (transaction ids 1..3; G_BUFS / G_INFLIGHT: deeper).
#ifdef G_BUFS
    constexpr uint32_t num_buffers = G_BUFS;  // ring columns (union.py DFLASH_G_BUFS)
    constexpr uint32_t extra_in_flight = G_INFLIGHT;
#else
    constexpr uint32_t num_buffers = 3;
    constexpr uint32_t extra_in_flight = 1;
#endif
    cb_reserve_back(cb_w, kt * (extra_in_flight + 1));
    uint32_t l1_write = get_write_ptr(cb_w);
    const uint32_t cb_base = cb_start;
    const uint32_t cb_end = cb_base + num_buffers * col_bytes;
    uint32_t free_blocks = num_buffers;
    uint32_t trid = 1;
    uint32_t trid_to_wait = 1;
    uint32_t issued = 0;
    // one task: the `cols` columns of one expert starting at src (bytes in this bank)
    auto issue = [&](uint32_t src) __attribute__((always_inline)) {
        for (uint32_t c = 0; c < cols; ++c) {
            noc_async_read_set_trid(trid);
            for (uint32_t p = 0; p < full_pkts; ++p) {
                noc_async_read_one_packet_with_state_with_trid(w_base, src, l1_write, trid);
                src += pkt;
                l1_write += pkt;
            }
            if constexpr (tail > 0) {
                noc_async_read_one_packet_set_state<true>(w_base, tail, vc);
                noc_async_read_one_packet_with_state_with_trid(w_base, src, l1_write, trid);
                noc_async_read_one_packet_set_state<true>(w_base, pkt, vc);
                src += tail;
                l1_write += tail;
            }
            ++issued;
            if (free_blocks == num_buffers - extra_in_flight) {
                noc_async_read_barrier_with_trid(trid_to_wait);
                cb_push_back(cb_w, kt);
#ifdef MOE_FUSED
                if (issued == 2) {
                    DeviceTimestampedData("MOE_T", 401);
                }
#endif
                trid_to_wait = trid_to_wait == num_buffers ? 1 : trid_to_wait + 1;
                cb_reserve_back(cb_w, kt * (extra_in_flight + 1));
            } else {
                free_blocks -= 1;
            }
            trid = trid == num_buffers ? 1 : trid + 1;
            if (l1_write >= cb_end) {
                l1_write = cb_base;
            }
        }
    };

#ifdef MOE_STEAL
    // Work stealing among the gate|up cores of this bank (dflash/moe_fused.py DFLASH_MOE_STEAL). The DRAM-streamed
    // experts are tasks t = 0 .. n - 1 (processing index e0 + t) of every core's own column pair. A core publishes
    // its next task in program semaphore 7 and stops at task n - max(claims) (semaphores 8 + r: the claim of the
    // bank-mate with reader index r, written by that mate). A core done with its own tasks claims the next one from
    // the end of the mate with the most unstarted, unclaimed tasks (claim m = that mate's largest claim + 1, task
    // n - m). Races only duplicate a task (its outputs and done flags are idempotent), never drop one: every claim
    // value 1 .. max was set by some core. Tasks go to the writer as [processing index, owner reader index] (cb 8).
    // runtime args 8..: [my reader index, mates, (noc x, noc y, reader index) per mate]
    constexpr uint32_t cb_task = 8;
    const uint32_t my_r = get_arg_val<uint32_t>(8);
    const uint32_t n_mates = get_arg_val<uint32_t>(9);
    const uint32_t n = num_sel - e0;
    volatile tt_l1_ptr uint32_t* progress = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(7));
    auto push_task = [&](uint32_t idx, uint32_t owner) __attribute__((always_inline)) {
        cb_reserve_back(cb_task, 1);
        volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_task));
        d[0] = idx;
        d[1] = owner;
        cb_push_back(cb_task, 1);
    };
    auto expert_src = [&](uint32_t idx, uint32_t r) __attribute__((always_inline)) -> uint32_t {
        return static_cast<uint32_t>(uni[1 + order[idx]]) * expert_stride + r * cols * col_bytes;
    };
    for (uint32_t t = 0; t < n; ++t) {
        uint32_t maxk = 0;
        for (uint32_t r = 0; r < G_READERS; ++r) {
            const uint32_t k = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(8 + r))[0];
            maxk = k > maxk ? k : maxk;
        }
        if (t + maxk >= n) {
            break;
        }
        progress[0] = t + 1;
        push_task(e0 + t, my_r);
        issue(expert_src(e0 + t, my_r));
    }
    progress[0] = n;
    // steal
    const uint32_t scratch = get_write_ptr(cb_idx) + 160;  // a mate's progress and claims, 16 B apart
    while (n_mates > 0) {
        uint32_t best = n_mates, best_rem = 1, best_max = 0;
        for (uint32_t j = 0; j < n_mates; ++j) {
            const uint32_t x = get_arg_val<uint32_t>(10 + 3 * j), y = get_arg_val<uint32_t>(11 + 3 * j);
            noc_async_read(get_noc_addr(x, y, get_semaphore(7)), scratch, 4);
            for (uint32_t r = 0; r < G_READERS; ++r) {
                noc_async_read(get_noc_addr(x, y, get_semaphore(8 + r)), scratch + 16 * (1 + r), 4);
            }
            noc_async_read_barrier();
            invalidate_l1_cache();
            volatile tt_l1_ptr uint32_t* v = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
            uint32_t mk = 0;
            for (uint32_t r = 0; r < G_READERS; ++r) {
                mk = v[4 * (1 + r)] > mk ? v[4 * (1 + r)] : mk;
            }
            const uint32_t started = v[0];
            const uint32_t end = n - mk;
            const uint32_t rem = end > started ? end - started : 0;
            if (rem > best_rem) {  // >= 2 unstarted: the mate's next task stays its own
                best = j;
                best_rem = rem;
                best_max = mk;
            }
        }
        if (best == n_mates) {
            break;
        }
        const uint32_t x = get_arg_val<uint32_t>(10 + 3 * best), y = get_arg_val<uint32_t>(11 + 3 * best);
        const uint32_t owner = get_arg_val<uint32_t>(12 + 3 * best);
        const uint32_t m = best_max + 1;
        noc_inline_dw_write(get_noc_addr(x, y, get_semaphore(8 + my_r)), m);
        noc_async_write_barrier();
        // the probes above used the read command buffer: restore the weight reads' one-packet state
        noc_async_read_one_packet_set_state<true>(w_base, pkt, vc);
        push_task(e0 + n - m, owner);
        issue(expert_src(e0 + n - m, owner));
    }
}
push_task(0xFFFF, 0);  // end of tasks
#else
    for (uint32_t e = e0; e < num_sel; ++e) {
        issue(static_cast<uint32_t>(uni[1 + order[e]]) * expert_stride + reader_offset);
    }
#endif
// Columns issued but not yet pushed (fewer than extra_in_flight + 1 when the union is tiny).
const uint32_t pending = issued < extra_in_flight ? issued : extra_in_flight;
for (uint32_t i = 0; i < pending; ++i) {
    noc_async_read_barrier_with_trid(trid_to_wait);
    cb_push_back(cb_w, kt);
    trid_to_wait = trid_to_wait == num_buffers ? 1 : trid_to_wait + 1;
}
}
