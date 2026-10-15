// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Union-of-experts down-projection weight streamer (NCRISC, NOC0) for the multi-row decode MoE (DFlash verify;
// fused_decode/union.py ExpertDownUnion). As kernels/stream_down_reader.cpp, for the U union experts (router union
// buffer [U, ids..]); each output column's U segments are read in blocks of `chunk` experts (the last block may hold
// fewer; a block always occupies chunk segments of the ring), which compute accumulates in DST.
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
    constexpr uint32_t seg_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t cols = get_compile_time_arg_val(3);
    constexpr uint32_t chunk = get_compile_time_arg_val(4);
    constexpr uint32_t expert_stride = get_compile_time_arg_val(5);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(6);
    constexpr auto uni_args = TensorAccessorArgs<7>();

    constexpr uint32_t seg_bytes = seg_tiles * tile_bytes;
    constexpr uint32_t block_tiles = chunk * seg_tiles;
    constexpr uint32_t block_bytes = chunk * seg_bytes;
    static_assert(seg_bytes <= NOC_MAX_BURST_SIZE, "one expert segment must fit one NOC packet");

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
    const uint32_t blocks = (num_sel + chunk - 1) / chunk;

    const uint64_t w_base = get_noc_addr_from_bank_id<true>(bank_id, w_addr);
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    noc_async_read_one_packet_set_state<true>(w_base, seg_bytes, vc);

    constexpr uint32_t num_buffers = 3;
    constexpr uint32_t extra_in_flight = 1;
    cb_reserve_back(cb_w, block_tiles * (extra_in_flight + 1));
    uint32_t l1_write = get_write_ptr(cb_w);
    const uint32_t cb_base = l1_write;
    const uint32_t cb_end = cb_base + num_buffers * block_bytes;
    uint32_t free_blocks = num_buffers;
    uint32_t trid = 1;
    uint32_t trid_to_wait = 1;
    uint32_t issued = 0;

    for (uint32_t c = 0; c < cols; ++c) {
        for (uint32_t b = 0; b < blocks; ++b) {
            const uint32_t e0 = b * chunk;
            const uint32_t n = num_sel - e0 < chunk ? num_sel - e0 : chunk;
            noc_async_read_set_trid(trid);
            for (uint32_t e = 0; e < n; ++e) {
                noc_async_read_one_packet_with_state_with_trid(
                    w_base, offsets[e0 + e] + c * seg_bytes, l1_write + e * seg_bytes, trid);
            }
            l1_write += block_bytes;
            ++issued;
            if (free_blocks == num_buffers - extra_in_flight) {
                noc_async_read_barrier_with_trid(trid_to_wait);
                cb_push_back(cb_w, block_tiles);
                trid_to_wait = trid_to_wait == num_buffers ? 1 : trid_to_wait + 1;
                cb_reserve_back(cb_w, block_tiles * (extra_in_flight + 1));
            } else {
                free_blocks -= 1;
            }
            trid = trid == num_buffers ? 1 : trid + 1;
            if (l1_write >= cb_end) {
                l1_write = cb_base;
            }
        }
    }
    const uint32_t pending = issued < extra_in_flight ? issued : extra_in_flight;
    for (uint32_t i = 0; i < pending; ++i) {
        noc_async_read_barrier_with_trid(trid_to_wait);
        cb_push_back(cb_w, block_tiles);
        trid_to_wait = trid_to_wait == num_buffers ? 1 : trid_to_wait + 1;
    }
}
