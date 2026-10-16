// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Constant BF16 tiles of the decode boundary readers (boundary_reader.cpp, bnd_mc_reader.cpp): the page is zeroed by
// NOC reads of the core's zero region, then only the nonzero entries are stored (element by element, the P matrix
// took ~9 us of the boundary's critical path).

#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace bnd_tiles {

inline void zero(uint32_t addr, uint32_t bytes) {
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t z = 0; z < bytes; z += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, addr + z, bytes - z < MEM_ZEROS_SIZE ? bytes - z : MEM_ZEROS_SIZE);
    }
    noc_async_read_barrier();
}

// reduce scaler: 1.0 in the first row of each face (scalar sum of all 1024 values)
inline void scaler(uint32_t addr) {
    zero(addr, 2048);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    for (uint32_t f = 0; f < 4; ++f) {
        for (uint32_t i = 0; i < 8; ++i) {
            s[f * 128 + i] = 0x3F803F80;
        }
    }
}

// P[R][C] = 1 iff R = C (mod rows), faces of 16 x 16 row-major
template <uint32_t rows>
inline void row_sum_matrix(uint32_t addr) {
    zero(addr, 2048);
    volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(addr);
    for (uint32_t R = 0; R < 32; ++R) {
        for (uint32_t C = R % rows; C < 32; C += rows) {
            p[((R >> 4) * 2 + (C >> 4)) * 256 + (R & 15) * 16 + (C & 15)] = 0x3F80;
        }
    }
}

}  // namespace bnd_tiles
