// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Early expert prefetch (NCRISC, NOC0) on a gate|up or down core of the fused verify MoE (dflash/moe_fused.py
// MoEFused.early_parts), run inside the program before the MoE (the sliding-window layers' QKV + row attention, on
// cores of their own): this core's columns of the layer's first E predicted experts are read into CB `cb`, the only
// circular buffer of this core in that program and so at the CB base address, where the MoE program's prediction CB
// (also first on this core, MOE_EARLY) finds them. The reads start once the QKV readers have landed their weights
// (program semaphore EARLY_GATE_SEM on the gate core counts them), so they do not slow QKV.
//
// runtime args: [w_addr, bank_id, reader_offset, read_bytes, expert_stride, hub_x, hub_y, pred_addr, gate_x, gate_y,
//                gate_n]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t bank_id = get_arg_val<uint32_t>(1);
    const uint32_t reader_offset = get_arg_val<uint32_t>(2);
    const uint32_t read_bytes = get_arg_val<uint32_t>(3);
    const uint32_t expert_stride = get_arg_val<uint32_t>(4);
    const uint32_t hub_x = get_arg_val<uint32_t>(5);
    const uint32_t hub_y = get_arg_val<uint32_t>(6);
    const uint32_t pred_addr = get_arg_val<uint32_t>(7);
    const uint32_t gate_x = get_arg_val<uint32_t>(8);
    const uint32_t gate_y = get_arg_val<uint32_t>(9);
    const uint32_t gate_n = get_arg_val<uint32_t>(10);

    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(1);
    constexpr uint32_t slot_bytes = get_compile_time_arg_val(2);  // the MoE prediction CB's stride per expert
    constexpr uint32_t slot_pages = get_compile_time_arg_val(3);
    constexpr uint32_t E = get_compile_time_arg_val(4);

    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);
    noc_async_read(get_noc_addr(hub_x, hub_y, pred_addr), scr, 32);  // the router's predictions (32 B per layer)
    noc_async_read_barrier();
    invalidate_l1_cache();
    uint32_t pred[E];
    for (uint32_t k = 0; k < E; ++k) {
        pred[k] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr)[k];
    }
    cb_reserve_back(cb, E * slot_pages);
    const uint32_t dst = get_write_ptr(cb);
    {
        volatile tt_l1_ptr uint32_t* g = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + 32);
        const uint64_t gate = get_noc_addr(gate_x, gate_y, get_semaphore(EARLY_GATE_SEM));
        while (true) {
            noc_async_read(gate, scr + 32, 4);
            noc_async_read_barrier();
            invalidate_l1_cache();
            if (g[0] >= gate_n) {
                break;
            }
            riscv_wait(1000);  // ~0.7 us between polls: dozens of cores poll this one
        }
    }
    DeviceTimestampedData("MOE_E", 1);
    for (uint32_t k = 0; k < E; ++k) {
        noc_async_read(
            get_noc_addr_from_bank_id<true>(bank_id, w_addr + pred[k] * expert_stride + reader_offset),
            dst + k * slot_bytes,
            read_bytes);
    }
    noc_async_read_barrier();
    DeviceTimestampedData("MOE_E", 2);
}
