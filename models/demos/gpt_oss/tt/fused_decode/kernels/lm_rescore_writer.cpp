// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Two-stage DFlash LM head, stage 2 writer (BRISC) (dflash/lm_rescore.py): the candidate's BF8 logit (lane of the 1x32
// output tile) as a BF16 order key + its column -> entry (list j, row r) of the stage-2 lists ([key, index] per row
// at byte 8 (rows j + r), the argmax merge's per-core list format, dflash/argmax.py).
//
// runtime args: [lists2_addr, r, j]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t lists_addr = get_arg_val<uint32_t>(0);
    const uint32_t r = get_arg_val<uint32_t>(1);
    const uint32_t j = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_meta = get_compile_time_arg_val(1);
    constexpr uint32_t rows = get_compile_time_arg_val(2);
    constexpr uint32_t lists_bytes = get_compile_time_arg_val(3);
    constexpr auto l_args = TensorAccessorArgs<4>();
    const auto s_l = TensorAccessor(l_args, lists_addr, lists_bytes);

    cb_wait_front(cb_meta, 1);
    volatile tt_l1_ptr uint32_t* m = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_meta));
    const uint32_t lane = m[0], c = m[1];
    cb_wait_front(cb_out, 1);
    volatile tt_l1_ptr uint16_t* o = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(cb_out));
    int32_t s = static_cast<int32_t>(static_cast<int16_t>(o[(lane / 16) * 16 + lane % 16]));
    s ^= (s >> 31) & 0x7FFF;
    const uint64_t dst = s_l.get_noc_addr(0) + 8 * (rows * j + r);
    noc_inline_dw_write(dst, static_cast<uint32_t>(s));
    noc_inline_dw_write(dst + 4, c);
    noc_async_write_barrier();
    cb_pop_front(cb_out, 1);
    cb_pop_front(cb_meta, 1);
}
