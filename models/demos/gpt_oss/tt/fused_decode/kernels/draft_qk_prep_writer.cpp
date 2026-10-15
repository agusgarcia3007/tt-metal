// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// DFlash drafter q / k prep, writer (BRISC) (dflash/qk_prep.py): this core's normed + rotated Q and K tiles -> its
// shards of the head-sharded Q / K outputs (same core, same L1 address on every core).
//
// runtime args: [q_out, k_out (, kv write core noc x, y with PREP_NOTIFY)]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t q_out = get_arg_val<uint32_t>(0);
    const uint32_t k_out = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_qo = get_compile_time_arg_val(0);
    constexpr uint32_t cb_ko = get_compile_time_arg_val(1);
    constexpr uint32_t tile = 2048;
    cb_wait_front(cb_qo, 2);
    noc_async_write(get_read_ptr(cb_qo), get_noc_addr(q_out), 2 * tile);
    cb_wait_front(cb_ko, 2);
    noc_async_write(get_read_ptr(cb_ko), get_noc_addr(k_out), 2 * tile);
    noc_async_write_barrier();
#ifdef PREP_NOTIFY
    // K shard written: tell the fused KV write (semaphore 0 on its core)
    noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(3), get_semaphore(0)), 1);
#endif
    cb_pop_front(cb_qo, 2);
    cb_pop_front(cb_ko, 2);
}
