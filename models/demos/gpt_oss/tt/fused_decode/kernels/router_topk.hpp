// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Router top-k + softmax on a data-movement core (no FPU): top-k by order-preserving integer keys of the BF16
// logits (ties to the lower id), softmax over the selected logits in Q16 fixed point (exp2 = integer shift x
// degree-5 polynomial, ~1e-5 relative), rounded to BF16. Shared by the decode router stream kernels.

#pragma once

#include <stdint.h>

namespace router_topk {
// BF16 bits -> Q16.16 fixed point (|x| < 2^14).
inline int32_t bf16_to_q16(uint16_t b) {
    const int32_t e = (b >> 7) & 0xFF;
    if (e == 0) {
        return 0;
    }
    const int32_t m = (b & 0x7F) | 0x80;
    const int32_t shift = e - 118;  // value = m * 2^(e - 134); in Q16: m << (e - 118)
    int32_t v = shift >= 0 ? (shift > 22 ? (m << 22) : (m << shift)) : (shift < -31 ? 0 : (m >> -shift));
    return (b & 0x8000) ? -v : v;
}

// exp(d) in Q16 for d <= 0 in Q16.
inline uint32_t exp_q16(int32_t d) {
    const int32_t t = static_cast<int32_t>((static_cast<int64_t>(d) * 94548) >> 16);  // d * log2(e), Q16
    const int32_t n = t >> 16;                                                        // floor
    const uint32_t f = static_cast<uint32_t>(t) & 0xFFFF;                             // [0, 1) in Q16
    uint32_t p = 87;                                                                  // 2^f, Horner, Q16
    p = 630 + ((p * f) >> 16);
    p = 3638 + ((p * f) >> 16);
    p = 15743 + ((p * f) >> 16);
    p = 45426 + ((p * f) >> 16);
    p = 65536 + ((p * f) >> 16);
    return n <= -31 ? 0 : p >> -n;
}

// Q16 in [0, 1] -> BF16 bits, round to nearest.
inline uint16_t q16_to_bf16(uint32_t w) {
    if (w == 0) {
        return 0;
    }
    int32_t msb = 31 - __builtin_clz(w);
    int32_t shift = msb - 7;
    uint32_t m = shift > 0 ? (w + (1u << (shift - 1))) >> shift : w << -shift;
    if (m >= 0x100) {  // rounding carried into a new bit
        m >>= 1;
        msb += 1;
    }
    return static_cast<uint16_t>(((127 + msb - 16) << 7) | (m & 0x7F));
}
}  // namespace router_topk

namespace router_topk {
// Writes 8 UINT16 ids then 8 BF16 scores (first topk valid, rest zero) to `stage` (32 bytes).
template <uint32_t topk, uint32_t n_logits>
inline void topk_softmax(volatile tt_l1_ptr uint16_t* logits, volatile tt_l1_ptr uint16_t* stage) {
    uint32_t keys[n_logits];
    for (uint32_t i = 0; i < n_logits; ++i) {
        const uint32_t b = logits[i];
        keys[i] = (b & 0x8000) ? (~b & 0xFFFF) : (b | 0x8000);
    }
    uint32_t ids[topk];
    int32_t vals[topk];
    for (uint32_t j = 0; j < topk; ++j) {
        uint32_t best = 0;
        for (uint32_t i = 1; i < n_logits; ++i) {
            best = keys[i] > keys[best] ? i : best;
        }
        ids[j] = best;
        vals[j] = bf16_to_q16(logits[best]);
        keys[best] = 0;
    }
    uint32_t e[topk];
    uint32_t sum = 0;
    for (uint32_t j = 0; j < topk; ++j) {
        e[j] = exp_q16(vals[j] - vals[0]);
        sum += e[j];
    }
    for (uint32_t i = 0; i < 16; ++i) {
        stage[i] = 0;
    }
    for (uint32_t j = 0; j < topk; ++j) {
        stage[j] = static_cast<uint16_t>(ids[j]);
        stage[8 + j] = q16_to_bf16((e[j] << 15) / (sum >> 1));
    }
}
}  // namespace router_topk

namespace router_topk {
// Local-array variant (no volatile L1 traffic): logits[n_logits] BF16 bits -> ids[topk], weights[topk] (BF16 bits).
template <uint32_t topk, uint32_t n_logits>
inline void topk_softmax_local(const uint16_t* logits, uint16_t* ids_out, uint16_t* w_out) {
    uint32_t keys[n_logits];
    for (uint32_t i = 0; i < n_logits; ++i) {
        const uint32_t b = logits[i];
        keys[i] = (b & 0x8000) ? (~b & 0xFFFF) : (b | 0x8000);
    }
    int32_t vals[topk];
    for (uint32_t j = 0; j < topk; ++j) {
        uint32_t best = 0;
        uint32_t bk = keys[0];
        for (uint32_t i = 1; i < n_logits; ++i) {
            if (keys[i] > bk) {
                bk = keys[i];
                best = i;
            }
        }
        ids_out[j] = static_cast<uint16_t>(best);
        vals[j] = bf16_to_q16(logits[best]);
        keys[best] = 0;
    }
    uint32_t e[topk];
    uint32_t sum = 0;
    for (uint32_t j = 0; j < topk; ++j) {
        e[j] = exp_q16(vals[j] - vals[0]);
        sum += e[j];
    }
    for (uint32_t j = 0; j < topk; ++j) {
        w_out[j] = q16_to_bf16((e[j] << 15) / (sum >> 1));
    }
}
}  // namespace router_topk

namespace router_topk {
// Multi-row router tail (DFlash verify): logits = the [rows, 32] BF16 output tile (row r: 16 values in face 0 at
// word 8 r, 16 in face 1 at word 8 (rows + r)). Per row top-k (one pass, sorted insertion, ties to the lower id) and
// softmax over the k (one division per row); then the union (ascending ids) and the [rows, 32] weight matrix:
// uni[0] = U, uni[1 + i] = union id i; wts[32 r + i] = row r's weight of union expert i (0 if not routed).
template <uint32_t topk, uint32_t rows>
inline void rows_union(const uint32_t* words, volatile tt_l1_ptr uint32_t* uni_w, volatile tt_l1_ptr uint32_t* wts_w) {
    uint32_t ids[rows][topk];
    uint32_t wq[rows][topk];
    uint32_t mask = 0;
    for (uint32_t r = 0; r < rows; ++r) {
        uint32_t bk[topk];
        uint32_t bi[topk];
        for (uint32_t j = 0; j < topk; ++j) {
            bk[j] = 0;
            bi[j] = 0;
        }
        for (uint32_t f = 0; f < 2; ++f) {
            const uint32_t* w = words + 8 * (f * rows + r);
            for (uint32_t q = 0; q < 8; ++q) {
                const uint32_t v = w[q];
                for (uint32_t h = 0; h < 2; ++h) {
                    const uint32_t b = h ? (v >> 16) : (v & 0xFFFF);
                    // order key: monotonic in the BF16 value, > 0 for every real value (0 = empty slot)
                    const uint32_t key = (b & 0x8000) ? (~b & 0xFFFF) : (b | 0x8000);
                    if (key > bk[topk - 1]) {  // strict: an equal value at a higher id stays out (lower id wins)
                        uint32_t p = topk - 1;
                        while (p > 0 && key > bk[p - 1]) {
                            bk[p] = bk[p - 1];
                            bi[p] = bi[p - 1];
                            --p;
                        }
                        bk[p] = key;
                        bi[p] = f * 16 + 2 * q + h;
                    }
                }
            }
        }
        int32_t v0 = 0;
        uint32_t e[topk];
        uint32_t sum = 0;
        for (uint32_t j = 0; j < topk; ++j) {
            const uint32_t k = bk[j];
            const uint16_t b = (k & 0x8000) ? static_cast<uint16_t>(k & 0x7FFF) : static_cast<uint16_t>(~k & 0xFFFF);
            const int32_t v = bf16_to_q16(b);
            if (j == 0) {
                v0 = v;
            }
            e[j] = exp_q16(v - v0);
            sum += e[j];
        }
        for (uint32_t j = 0; j < topk; ++j) {
            ids[r][j] = bi[j];
            wq[r][j] = (e[j] << 15) / (sum >> 1);  // Q16 weight (as topk_softmax)
            mask |= 1u << bi[j];
        }
    }

    uint32_t slot[32];
    uint32_t u = 0;
    uint16_t list[64];
    for (uint32_t i = 0; i < 64; ++i) {
        list[i] = 0;
    }
    for (uint32_t m = mask; m; m &= m - 1) {
        const uint32_t id = __builtin_ctz(m);
        slot[id] = u;
        list[1 + u] = id;
        ++u;
    }
    list[0] = u;
    for (uint32_t i = 0; i < 32; ++i) {
        uni_w[i] = list[2 * i] | (static_cast<uint32_t>(list[2 * i + 1]) << 16);
    }
    uint16_t wt[32];
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t i = 0; i < 32; ++i) {
            wt[i] = 0;
        }
        for (uint32_t j = 0; j < topk; ++j) {
            wt[slot[ids[r][j]]] = q16_to_bf16(wq[r][j]);
        }
        for (uint32_t i = 0; i < 16; ++i) {
            wts_w[16 * r + i] = wt[2 * i] | (static_cast<uint32_t>(wt[2 * i + 1]) << 16);
        }
    }
}
}  // namespace router_topk

namespace router_topk {
// rows_union in two parts, so two RISCs can split the rows: rows_topk writes rows [r0, r1)'s top-k ids and Q16
// weights to ids[r * topk + j] / wq[r * topk + j]; rows_finish builds the union and the weight matrix from all rows.
template <uint32_t topk, uint32_t rows>
inline void rows_topk(
    const uint32_t* words,
    uint32_t r0,
    uint32_t r1,
    volatile tt_l1_ptr uint32_t* ids,
    volatile tt_l1_ptr uint32_t* wq) {
    static_assert(topk == 4, "rows_topk: register top-4");
    for (uint32_t r = r0; r < r1; ++r) {
        // sorted top-4 in registers (b0 >= b1 >= b2 >= b3; strict > keeps the lower id first on ties)
        uint32_t b0 = 0, b1 = 0, b2 = 0, b3 = 0, i0 = 0, i1 = 0, i2 = 0, i3 = 0;
        auto ins = [&](uint32_t key, uint32_t id) __attribute__((always_inline)) {
            if (key > b3) {
                if (key > b2) {
                    b3 = b2;
                    i3 = i2;
                    if (key > b1) {
                        b2 = b1;
                        i2 = i1;
                        if (key > b0) {
                            b1 = b0;
                            i1 = i0;
                            b0 = key;
                            i0 = id;
                        } else {
                            b1 = key;
                            i1 = id;
                        }
                    } else {
                        b2 = key;
                        i2 = id;
                    }
                } else {
                    b3 = key;
                    i3 = id;
                }
            }
        };
#pragma GCC unroll 2
        for (uint32_t f = 0; f < 2; ++f) {
            const uint32_t* w = words + 8 * (f * rows + r);
#pragma GCC unroll 8
            for (uint32_t q = 0; q < 8; ++q) {
                const uint32_t v = w[q];
                const uint32_t lo = v & 0xFFFF, hi = v >> 16;
                ins((lo & 0x8000) ? (~lo & 0xFFFF) : (lo | 0x8000), f * 16 + 2 * q);
                ins((hi & 0x8000) ? (~hi & 0xFFFF) : (hi | 0x8000), f * 16 + 2 * q + 1);
            }
        }
        const uint32_t bk[4] = {b0, b1, b2, b3};
        const uint32_t bi[4] = {i0, i1, i2, i3};
        int32_t v0 = 0;
        uint32_t e[topk];
        uint32_t sum = 0;
        for (uint32_t j = 0; j < topk; ++j) {
            const uint32_t k = bk[j];
            const uint16_t b = (k & 0x8000) ? static_cast<uint16_t>(k & 0x7FFF) : static_cast<uint16_t>(~k & 0xFFFF);
            const int32_t v = bf16_to_q16(b);
            if (j == 0) {
                v0 = v;
            }
            e[j] = exp_q16(v - v0);
            sum += e[j];
        }
        for (uint32_t j = 0; j < topk; ++j) {
            ids[r * topk + j] = bi[j];
            wq[r * topk + j] = (e[j] << 15) / (sum >> 1);
        }
    }
}

template <uint32_t topk, uint32_t rows>
inline void rows_finish(
    volatile tt_l1_ptr uint32_t* ids,
    volatile tt_l1_ptr uint32_t* wq,
    volatile tt_l1_ptr uint32_t* uni_w,
    volatile tt_l1_ptr uint32_t* wts_w) {
    uint32_t mask = 0;
    for (uint32_t i = 0; i < rows * topk; ++i) {
        mask |= 1u << ids[i];
    }
    uint32_t slot[32];
    uint32_t u = 0;
    uint16_t list[64];
    for (uint32_t i = 0; i < 64; ++i) {
        list[i] = 0;
    }
    for (uint32_t m = mask; m; m &= m - 1) {
        const uint32_t id = __builtin_ctz(m);
        slot[id] = u;
        list[1 + u] = id;
        ++u;
    }
    list[0] = u;
    for (uint32_t i = 0; i < 32; ++i) {
        uni_w[i] = list[2 * i] | (static_cast<uint32_t>(list[2 * i + 1]) << 16);
    }
    uint16_t wt[32];
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t i = 0; i < 32; ++i) {
            wt[i] = 0;
        }
        for (uint32_t j = 0; j < topk; ++j) {
            wt[slot[ids[r * topk + j]]] = q16_to_bf16(wq[r * topk + j]);
        }
        for (uint32_t i = 0; i < 16; ++i) {
            wts_w[16 * r + i] = wt[2 * i] | (static_cast<uint32_t>(wt[2 * i + 1]) << 16);
        }
    }
}
}  // namespace router_topk
