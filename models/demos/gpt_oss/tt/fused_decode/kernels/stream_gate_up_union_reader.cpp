// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Union-of-experts gate|up weight streamer (NCRISC, NOC0) for the multi-row decode MoE (DFlash verify;
// fused_decode/union.py ExpertGateUpUnion). As kernels/stream_gate_up_reader.cpp, but the experts are the runtime
// union of the rows' routed experts: the router's union buffer holds [U, id_0 .. id_{U-1}] (UINT16).
//
// runtime args: [w_addr, union_addr, bank_id, vc, reader_offset_bytes]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

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

    cb_reserve_back(cb_idx, 1);
    const uint32_t idx_l1 = get_write_ptr(cb_idx);
    const auto s_uni = TensorAccessor(uni_args, uni_addr, 128);
    noc_async_read(s_uni.get_noc_addr(0), idx_l1, 128);
    noc_async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr uint16_t* uni = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(idx_l1);
    const uint32_t num_sel = uni[0];
    uint32_t offsets[32];
    for (uint32_t e = 0; e < num_sel; ++e) {
        offsets[e] = static_cast<uint32_t>(uni[1 + e]) * expert_stride + reader_offset;
    }

    const uint64_t w_base = get_noc_addr_from_bank_id<true>(bank_id, w_addr);
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    noc_async_read_one_packet_set_state<true>(w_base, pkt, vc);

    // Triple-buffered columns, one extra column in flight (transaction ids 1..3).
    constexpr uint32_t num_buffers = 3;
    constexpr uint32_t extra_in_flight = 1;
    cb_reserve_back(cb_w, kt * (extra_in_flight + 1));
    uint32_t l1_write = get_write_ptr(cb_w);
    const uint32_t cb_base = l1_write;
    const uint32_t cb_end = cb_base + num_buffers * col_bytes;
    uint32_t free_blocks = num_buffers;
    uint32_t trid = 1;
    uint32_t trid_to_wait = 1;
    uint32_t issued = 0;

    for (uint32_t e = 0; e < num_sel; ++e) {
        uint32_t src = offsets[e];
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
    }
    // Columns issued but not yet pushed (fewer than extra_in_flight + 1 when the union is tiny).
    const uint32_t pending = issued < extra_in_flight ? issued : extra_in_flight;
    for (uint32_t i = 0; i < pending; ++i) {
        noc_async_read_barrier_with_trid(trid_to_wait);
        cb_push_back(cb_w, kt);
        trid_to_wait = trid_to_wait == num_buffers ? 1 : trid_to_wait + 1;
    }
}
