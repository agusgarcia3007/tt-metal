// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Multi-core decode boundary, helper writer (BRISC): (1) this helper's row sums of squares (one FP32 tile from
// compute) -> slot `index` of cb_xchg on every helper (itself included), then program semaphore XCHG_SEM on each;
// (2) its normed pages -> pages [page0, page0 + valid) of the normed output on the boundary core, then program
// semaphore RELAY_SEM there (the boundary core relays to the consumers once every helper is done).
//
// runtime args: [x_addr (boundary core), boundary noc x, y, page0, valid pages, index, (noc x, y) x helpers]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t bx = get_arg_val<uint32_t>(1);
    const uint32_t by = get_arg_val<uint32_t>(2);
    const uint32_t page0 = get_arg_val<uint32_t>(3);
    const uint32_t valid = get_arg_val<uint32_t>(4);
    const uint32_t index = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_v = get_compile_time_arg_val(0);
    constexpr uint32_t cb_xchg = get_compile_time_arg_val(1);
    constexpr uint32_t cb_x = get_compile_time_arg_val(2);
    constexpr uint32_t PH = get_compile_time_arg_val(3);
    constexpr uint32_t helpers = get_compile_time_arg_val(4);
    constexpr uint32_t xchg_sem = get_compile_time_arg_val(5);
    constexpr uint32_t relay_sem = get_compile_time_arg_val(6);
    // 1: also multicast the normed pages into the same offset of the x mirror on every worker core (runtime args
    // 6 + 2 helpers ..: mirror address, noc x0, y0, x1, y1 of the worker grid, cores), so consumers read x locally.
    constexpr uint32_t mirror = get_compile_time_arg_val(7);
    // 1: x stays in this helper's shard of the normed output (cb_x backed by it); consumers read it from here
    constexpr uint32_t x_on_helpers = get_compile_time_arg_val(8);
    constexpr uint32_t v_bytes = 4096;  // FP32 tile
    constexpr uint32_t page = 2048;

    cb_wait_front(cb_v, 1);
    const uint32_t v = get_read_ptr(cb_v);
    const uint32_t slot = get_write_ptr(cb_xchg) + index * v_bytes;  // same address on every helper
    for (uint32_t h = 0; h < helpers; ++h) {
        noc_async_write(
            v, get_noc_addr(get_arg_val<uint32_t>(6 + 2 * h), get_arg_val<uint32_t>(7 + 2 * h), slot), v_bytes);
    }
    noc_async_write_barrier();
    const uint32_t sem = get_semaphore(xchg_sem);
    for (uint32_t h = 0; h < helpers; ++h) {
        noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(6 + 2 * h), get_arg_val<uint32_t>(7 + 2 * h), sem), 1);
    }
    cb_pop_front(cb_v, 1);

    cb_wait_front(cb_x, PH);
    if constexpr (!x_on_helpers) {
        noc_async_write(get_read_ptr(cb_x), get_noc_addr(bx, by, x_addr + page0 * page), valid * page);
    }
    if constexpr (mirror) {
        constexpr uint32_t m = 6 + 2 * helpers;
        const uint32_t mirror_addr = get_arg_val<uint32_t>(m);
        const uint64_t dst = get_noc_multicast_addr(
            get_arg_val<uint32_t>(m + 1),
            get_arg_val<uint32_t>(m + 2),
            get_arg_val<uint32_t>(m + 3),
            get_arg_val<uint32_t>(m + 4),
            mirror_addr + page0 * page);
        noc_async_write_multicast_loopback_src(get_read_ptr(cb_x), dst, valid * page, get_arg_val<uint32_t>(m + 5));
    }
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(bx, by, get_semaphore(relay_sem)), 1);
    noc_async_atomic_barrier();
    cb_pop_front(cb_x, PH);
}
