// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Multi-core decode boundary, boundary-core relay: once every helper wrote its normed pages here (program semaphore
// RELAY_SEM reaches `helpers`), increment program semaphore NOTIFY_SEM on every consumer core.
//
// runtime args: [(noc x, y) x consumers]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t helpers = get_compile_time_arg_val(0);
    constexpr uint32_t relay_sem = get_compile_time_arg_val(1);
    constexpr uint32_t consumers = get_compile_time_arg_val(2);
    constexpr uint32_t notify_sem = get_compile_time_arg_val(3);
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(relay_sem)), helpers);
    const uint32_t sem = get_semaphore(notify_sem);
    for (uint32_t i = 0; i < consumers; ++i) {
        noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(2 * i), get_arg_val<uint32_t>(2 * i + 1), sem), 1);
    }
    noc_async_atomic_barrier();
}
