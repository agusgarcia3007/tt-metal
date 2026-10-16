// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Block attention, reader (NCRISC) (dflash/block_attention.py): one core per (kv head g, history slice s), fused into
// the QKV program. The M block rows' queries of kv head g are Tt packed tiles (tile t, rows 8 i .. 8 i + 7 = block row
// 4 t + i's query heads 8 g .. 8 g + 7), each [32 rows, 64 dims] = 2 BF16 tiles: every K / V chunk of the slice is read
// once for all M rows. Pushes:
//   cb_aux: [scaler (1.0 in row 0 of each face), the causal masks of the block's chunks c_block + e (tile 1 + 2 t + e:
//            row group i keeps the columns at positions <= pos[4 t + i]), the sink-column mask (keeps column 0), the
//            sink tile (slice 0: column 0 of row group i = kv head g's 8 sink logits / scale), a zero tile];
//   cb_k / cb_v: the slice's chunks in groups of G ([32 positions, 32 dims] x 2 per chunk); the groups entirely before
//                the block's first position (already in the cache) while QKV runs, as far as the CBs hold them;
//   cb_q: the Tt x 2 packed Q tiles, after the QKV writers signal (BA_WAIT_N increments of program semaphore
//         BA_WAIT_SEM: Q and the block's K / V rows are written);
//   cb_recv: per tree round r < recv_rounds, the child's partial (written into slot r by its writer; semaphore
//            BA_TREE_SEM bit r).
//
// common runtime args: [q_addr, k_addr, v_addr, pos_addr, pt_addr, sink_addr, mask_lib_addr, core index table]
// BA_SLIDE: causal with a sliding window: row r attends [pos[r] + 1 - BA_SLIDE, pos[r]] (lower masks per row group).
// BA_UNIFORM: every row attends [0, pos[M - 1]] (the drafter's block; [pos[M - 1] + 1 - BA_WINDOW, ..] with a window),
// no sinks.
// BA_TPC: packed Q tiles per core (t0 .. t0 + BA_TPC - 1; BA_TPC < Tt: the slices of each tile group are their own
// tree)

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ba_common.hpp"

namespace {
constexpr uint32_t rows8(uint32_t i, uint32_t f) { return ((i >> 1) * 2 + f) * 512 + (i & 1) * 256; }
}  // namespace

void kernel_main() {
    const uint32_t q_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t pos_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t pt_addr = get_common_arg_val<uint32_t>(4);
    const uint32_t sink_addr = get_common_arg_val<uint32_t>(5);
    const uint32_t lib_addr = get_common_arg_val<uint32_t>(6);
    const uint32_t ci = ba_core_index(7);  // (kv head g, tile group, slice s) = ((g BA_TQ + q) S + s)
    const uint32_t g = ci / get_compile_time_arg_val(12) / BA_TQ;
    const uint32_t s = ci % get_compile_time_arg_val(12);
    const uint32_t t0 = (ci / get_compile_time_arg_val(12)) % BA_TQ * BA_TPC;  // this core's first packed Q tile

    constexpr uint32_t cb_q = get_compile_time_arg_val(0);
    constexpr uint32_t cb_k = get_compile_time_arg_val(1);
    constexpr uint32_t cb_v = get_compile_time_arg_val(2);
    constexpr uint32_t cb_aux = get_compile_time_arg_val(3);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(4);
    constexpr uint32_t cb_recv = get_compile_time_arg_val(5);
    constexpr uint32_t M = get_compile_time_arg_val(6);
    constexpr uint32_t nkv = get_compile_time_arg_val(7);
    constexpr uint32_t block = get_compile_time_arg_val(8);  // positions per page-table block
    constexpr uint32_t kv_tile_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t pt_page_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t pos_page_bytes = get_compile_time_arg_val(11);
    constexpr uint32_t S = get_compile_time_arg_val(12);           // slice cores per kv head
    constexpr uint32_t min_chunks = get_compile_time_arg_val(13);  // fewest chunks per active slice
    constexpr uint32_t G = get_compile_time_arg_val(14);           // chunks per group
    constexpr uint32_t cap_groups = get_compile_time_arg_val(15);  // groups cb_k / cb_v hold
    constexpr uint32_t partial_tiles = get_compile_time_arg_val(16);
    constexpr auto q_args = TensorAccessorArgs<17>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto pos_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto pt_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
    constexpr auto sink_args = TensorAccessorArgs<pt_args.next_compile_time_args_offset()>();
    constexpr auto lib_args = TensorAccessorArgs<sink_args.next_compile_time_args_offset()>();

    constexpr uint32_t tile = 2048;  // BF16 tile
    constexpr uint32_t Tt = M / 4;   // packed Q tiles per kv head
    constexpr uint32_t TC = BA_TPC;  // this core's: tiles t0 .. t0 + TC - 1
    constexpr uint32_t tpb = block / 32;
    const auto s_q = TensorAccessor(q_args, q_addr, tile);
    const auto s_k = TensorAccessor(k_args, k_addr, kv_tile_bytes);
    const auto s_v = TensorAccessor(v_args, v_addr, kv_tile_bytes);
    const auto s_pos = TensorAccessor(pos_args, pos_addr, pos_page_bytes);
    const auto s_pt = TensorAccessor(pt_args, pt_addr, pt_page_bytes);
    const auto s_sink = TensorAccessor(sink_args, sink_addr, tile);
    const auto s_lib = TensorAccessor(lib_args, lib_addr, 33 * 33 * 2 * 256);

    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);  // [pos page | page-table row]
    noc_async_read(s_pos.get_noc_addr(0), scr, pos_page_bytes);
    noc_async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr int32_t* pos = reinterpret_cast<volatile tt_l1_ptr int32_t*>(scr);
    {
        // the positions to this core's writer (BA_CB_POS): one DRAM read of them per core, not two
        cb_reserve_back(BA_CB_POS, 1);
        volatile tt_l1_ptr int32_t* wp = reinterpret_cast<volatile tt_l1_ptr int32_t*>(get_write_ptr(BA_CB_POS));
        for (uint32_t r = 0; r < M; ++r) {
            wp[r] = pos[r];
        }
        cb_push_back(BA_CB_POS, 1);
    }
    uint32_t p0 = static_cast<uint32_t>(pos[0]);  // the block's first position (its K / V rows written by QKV)
#ifdef BA_UNIFORM
    // every row attends [lo, hi = pos[M - 1]] (the drafter; lo = 0, or hi + 1 - BA_WINDOW with a window): the last
    // chunk masked past hi (mask e = 0), with a window the first chunk before lo (mask e = 1, slice 0's chunk 0), no
    // sinks
    const uint32_t hi = static_cast<uint32_t>(pos[M - 1]);
    // pos holds hi for every row, but the block's rows (written this round) are hi + 1 - M .. hi: a group ending
    // before hi may still hold some of them (the block crossing a chunk boundary), so it is not old before they land
#ifndef BA_P0_HI
    p0 = hi + 1 >= M ? hi + 1 - M : 0;
#endif
#ifdef BA_WINDOW
    const uint32_t lo = hi + 1 >= BA_WINDOW ? hi + 1 - BA_WINDOW : 0;
#else
    const uint32_t lo = 0;
#endif
    const BaSlice sl = ba_slice(hi, hi, s, S, min_chunks, lo / 32);
    constexpr bool has_sink = false;
#elif defined(BA_SLIDE)
    // row r attends [lo_r, pos[r]], lo_r = pos[r] + 1 - BA_SLIDE (>= 0): the slices cover from the first row's lo; its
    // first chunks are cut per row group (lower masks e = 2, 3: slice 0's chunks 0, 1), the block's chunks causally
    auto lo_of = [&](uint32_t r) -> uint32_t {
        const uint32_t p = static_cast<uint32_t>(pos[r]);
        return p + 1 >= BA_SLIDE ? p + 1 - BA_SLIDE : 0;
    };
    const uint32_t lo_min = lo_of(0);
    const BaSlice sl = ba_slice(p0, static_cast<uint32_t>(pos[M - 1]), s, S, min_chunks, lo_min / 32);
    constexpr bool has_sink = true;
#else
    const BaSlice sl = ba_slice(p0, static_cast<uint32_t>(pos[M - 1]), s, S, min_chunks);
    constexpr bool has_sink = true;
#endif
#ifdef BA_SLIDE
    constexpr uint32_t NM = 4;  // mask tiles per Q tile: causal e = 0, 1; window starts e = 2, 3
#else
    constexpr uint32_t NM = 2;
#endif
    volatile tt_l1_ptr uint32_t* qkv_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(BA_WAIT_SEM));
    auto kv_ready = [&]() {
        invalidate_l1_cache();
        return *qkv_sem >= BA_WAIT_N;
    };
#ifdef BA_PATCH
    // BA_PATCH (the drafter, fused after its K / V cache write, dflash/kv_write.py notify_early): the block's chunks
    // (holding rows hi + 1 - M .. hi) are read from DRAM with the old ones (their other rows are old), and the block's
    // rows are patched in from the head-sharded K / V (row r: user core r's shard, kv head g = tile row g) once the
    // KV write signals that those are written (BA_WAIT_N): the cache write is off this op's path.
    // common args BA_PATCH..: [k_src, v_src, noc (x << 8 | y) per row (2 per word)]
#endif
    if (sl.c_hi == sl.c_lo) {
        // idle slice: done (the QKV writers' increments may still land here: harmless, the next program on this core
        // starts after this one completes, i.e. after those writers' atomic barriers)
        return;
    }
    // only this slice's page-table entries (every core reading the whole row would hit one DRAM bank with cores x
    // the row): entries [e0, e1), read from the 64-byte boundary at or below e0
    const uint32_t e0 = sl.c_lo * 32 / block, e1 = (sl.c_hi * 32 - 1) / block + 1;
    const uint32_t a0 = (e0 * 4) & ~63u;
    const uint32_t a1 = (e1 * 4 + 63) & ~63u;
    noc_async_read(s_pt.get_noc_addr(0) + a0, scr + 256, (a1 < pt_page_bytes ? a1 : pt_page_bytes) - a0);
    noc_async_read_barrier();
    const uint32_t pt_first = a0 / 4;  // the first entry read (pt[] below indexes entries from it)
    const uint32_t nc = sl.c_hi - sl.c_lo;
    const uint32_t n_groups = (nc + G - 1) / G;

    // aux: scaler, causal masks, sink-column mask, sink tile, zero tile
    constexpr uint32_t n_aux = NM * TC + 4;
    cb_reserve_back(cb_aux, n_aux);
    const uint32_t aux = get_write_ptr(cb_aux);
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t z = 0; z < tile; z += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, aux + z, MEM_ZEROS_SIZE);
        noc_async_read(zeros, aux + (NM * TC + 3) * tile + z, MEM_ZEROS_SIZE);
    }
    if (has_sink &&
        s == 0) {  // the sink tile (aux 2 TC + 2): kv head g's rows of the heads' sink tile, in every row group
        const uint64_t src = s_sink.get_noc_addr(0);
        for (uint32_t i = 0; i < 4; ++i) {
            for (uint32_t f = 0; f < 2; ++f) {
                noc_async_read(src + rows8(g, f), aux + (NM * TC + 2) * tile + rows8(i, f), 256);
            }
        }
    }
    // mask tile: row group i keeps columns [from, keep[i]) (BF16 -1e30 elsewhere): its 8 segments (8 rows x 16
    // columns) copied from the mask library (segment (a, b, f) = half f of a row group keeping columns [a, b), a, b in
    // 0 .. 32)
    const uint64_t lib = s_lib.get_noc_addr(0);
    auto mask = [&](uint32_t t, const uint32_t* keep, uint32_t from = 0) {
        for (uint32_t i = 0; i < 4; ++i) {
            for (uint32_t f = 0; f < 2; ++f) {
                noc_async_read(lib + ((from * 33 + keep[i]) * 2 + f) * 256, aux + t * tile + rows8(i, f), 256);
            }
        }
    };
    // only the causal masks of the block's chunks this slice holds (chunk c_block + e)
    for (uint32_t e = 0; e < 2; ++e) {
        const uint32_t c = sl.c_block + e;
        if (c < sl.c_lo || c >= sl.c_hi) {
            continue;
        }
        for (uint32_t t = 0; t < TC; ++t) {
            uint32_t keep[4];
            const int32_t c0 = static_cast<int32_t>(c * 32);
            for (uint32_t i = 0; i < 4; ++i) {
#ifdef BA_UNIFORM
                const int32_t k = pos[M - 1] - c0 + 1;
#else
                const int32_t k = pos[4 * (t0 + t) + i] - c0 + 1;
#endif
                keep[i] = k < 0 ? 0 : (k > 32 ? 32 : static_cast<uint32_t>(k));
            }
#ifdef BA_UNIFORM
            // the window's first chunk is this one too: keep [lo % 32, ..)
            mask(1 + NM * t + e, keep, c == lo / 32 ? lo % 32 : 0);
#else
            mask(1 + NM * t + e, keep);
#endif
        }
    }
#ifdef BA_SLIDE
    // the window starts: chunks lo_min / 32 + e (slice 0's chunks e), row group i keeps [lo_r - 32 c, 32)
    for (uint32_t e = 0; e < 2; ++e) {
        const uint32_t c = lo_min / 32 + e;
        if (sl.c_lo != lo_min / 32 || c >= sl.c_hi || c >= sl.c_block) {  // (the window's first slice)
            continue;
        }
        for (uint32_t t = 0; t < TC; ++t) {
            for (uint32_t i = 0; i < 4; ++i) {
                const uint32_t lo = lo_of(4 * (t0 + t) + i);
                const uint32_t a = lo <= c * 32 ? 0 : (lo - c * 32 > 32 ? 32 : lo - c * 32);
                for (uint32_t f = 0; f < 2; ++f) {
                    noc_async_read(
                        lib + ((a * 33 + 32) * 2 + f) * 256, aux + (1 + NM * t + 2 + e) * tile + rows8(i, f), 256);
                }
            }
        }
    }
#endif
#ifdef BA_UNIFORM
    // the window's first chunk (slice 0's chunk 0) when it is not the last: keep [lo % 32, 32)
    if (sl.c_lo == lo / 32 && (lo % 32) && lo / 32 != hi / 32) {
        const uint32_t keep32[4] = {32, 32, 32, 32};
        for (uint32_t t = 0; t < TC; ++t) {
            mask(1 + NM * t + 1, keep32, lo % 32);
        }
    }
#endif
    if (has_sink && s == 0) {
        const uint32_t keep1[4] = {1, 1, 1, 1};
        mask(1 + NM * TC, keep1);
    }
    noc_async_read_barrier();  // the zero fill / sink / masks
    // scaler: 1.0 in row 0 of each face (BF16)
    volatile tt_l1_ptr uint16_t* sc = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(aux);
    for (uint32_t f = 0; f < 4; ++f) {
        for (uint32_t c = 0; c < 16; ++c) {
            sc[f * 256 + c] = 0x3F80;
        }
    }
    cb_push_back(cb_aux, n_aux);

    // K / V groups pipelined (up to cap_groups in flight, NOC transaction id 1 + slot), pushed in order as each
    // lands; Q as soon as the QKV writers have signalled (a group holding the block's rows waits for it too)
    volatile tt_l1_ptr int32_t* pt = reinterpret_cast<volatile tt_l1_ptr int32_t*>(scr + 256);
    const uint32_t k_base = get_write_ptr(cb_k), v_base = get_write_ptr(cb_v);  // fresh CBs: slot 0
    constexpr uint32_t group_pages = 2 * G;
    auto issue_group = [&](uint32_t gi) {
        const uint32_t slot = gi % cap_groups;
        noc_async_read_set_trid(1 + slot);
        const uint32_t kl1 = k_base + slot * group_pages * kv_tile_bytes,
                       vl1 = v_base + slot * group_pages * kv_tile_bytes;
        const uint32_t j0 = gi * G, j1 = j0 + G < nc ? j0 + G : nc;
        for (uint32_t j = j0; j < j1; ++j) {
            const uint32_t p = (sl.c_lo + j) * 32;
            const uint32_t blk = static_cast<uint32_t>(pt[p / block - pt_first]);
            const uint32_t t0 = ((blk * nkv + g) * tpb + (p % block) / 32) * 2;
            for (uint32_t d = 0; d < 2; ++d) {
                noc_async_read(s_k.get_noc_addr(t0 + d), kl1 + (2 * (j - j0) + d) * kv_tile_bytes, kv_tile_bytes);
                noc_async_read(s_v.get_noc_addr(t0 + d), vl1 + (2 * (j - j0) + d) * kv_tile_bytes, kv_tile_bytes);
            }
        }
        noc_async_read_set_trid(0);
    };
#ifdef BA_PATCH
    const uint32_t kv_c0 = (hi + 1 - M) / 32;  // the block's first chunk
    auto block_chunks = [&](uint32_t gi) {     // group gi holds a block chunk
        const uint32_t c0 = sl.c_lo + gi * G, c1 = sl.c_lo + ((gi + 1) * G < nc ? (gi + 1) * G : nc);
        return c1 > kv_c0 && c0 <= hi / 32;
    };
#endif
    // a group is old when its last chunk ends before position p0 (the QKV writes start there)
    auto old_group = [&](uint32_t gi) {
#ifdef BA_PATCH
        return true;  // (the block's rows are patched in after the read)
#endif
        const uint32_t j_last = (gi + 1) * G < nc ? (gi + 1) * G - 1 : nc - 1;
        return (sl.c_lo + j_last) * 32 + 31 < p0;
    };
    bool q_done = false;
    auto read_q = [&]() {
        cb_reserve_back(cb_q, 2 * TC);
        const uint32_t ql1 = get_write_ptr(cb_q);
        noc_async_read_set_trid(15);
#ifdef BA_Q_ROWS
        // Q in the rows layout [1, M, heads, 64] (row r's tiles 2 r + d, heads in tile rows): gather packed tile (t, d)
        // rows 8 i .. 8 i + 7 = row 4 (t0 + t) + i's heads 8 g .. 8 g + 7 (256-byte segments per 16-column half f)
        for (uint32_t t = 0; t < TC; ++t) {
            for (uint32_t d = 0; d < 2; ++d) {
                for (uint32_t i = 0; i < 4; ++i) {
                    const uint64_t src = s_q.get_noc_addr(2 * (4 * (t0 + t) + i) + d);
                    for (uint32_t f = 0; f < 2; ++f) {
                        noc_async_read(src + rows8(g, f), ql1 + (2 * t + d) * tile + rows8(i, f), 256);
                    }
                }
            }
        }
#else
        for (uint32_t t = 0; t < 2 * TC; ++t) {
            noc_async_read(s_q.get_noc_addr(g * 2 * Tt + 2 * t0 + t), ql1 + t * tile, tile);
        }
#endif
        noc_async_read_set_trid(0);
        noc_async_read_barrier_with_trid(15);
        cb_push_back(cb_q, 2 * TC);
        q_done = true;
        DeviceTimestampedData("BA_R", 1);  // (profiling: Q read)
    };
#ifdef BA_PATCH
    // the block's rows of group gi's chunks (landed in its slot): row r (position hi + 1 - M + r) = head row g of user
    // core r's K / V shard (2 tiles d, 32 bytes per face f) -> row pos % 32 of the chunk's tiles
    auto patch = [&](uint32_t gi) {
        const uint32_t slot = gi % cap_groups;
        const uint32_t kl1 = k_base + slot * group_pages * kv_tile_bytes,
                       vl1 = v_base + slot * group_pages * kv_tile_bytes;
        const uint32_t k_src = get_common_arg_val<uint32_t>(BA_PATCH),
                       v_src = get_common_arg_val<uint32_t>(BA_PATCH + 1);
        const uint32_t src_off = (g < 16 ? 0 : 1024) + (g % 16) * 32;
        const uint32_t c0 = sl.c_lo + gi * G;
        for (uint32_t r = 0; r < M; ++r) {
            const uint32_t p = hi + 1 - M + r, c = p / 32;
            if (c < c0 || c >= c0 + G || c >= sl.c_hi) {
                continue;
            }
            const uint32_t j = c - c0;
            const uint32_t nw = get_common_arg_val<uint32_t>(BA_PATCH + 2 + r / 2) >> (16 * (r % 2));
            const uint32_t x = (nw >> 8) & 0xFF, y = nw & 0xFF;
            const uint32_t rr = p % 32;
            const uint32_t dst_off = (rr < 16 ? 0 : 1024) + (rr % 16) * 32;
            for (uint32_t d = 0; d < 2; ++d) {
                for (uint32_t f = 0; f < 2; ++f) {
                    const uint32_t so = d * tile + src_off + f * 512, dof = (2 * j + d) * tile + dst_off + f * 512;
                    noc_async_read(get_noc_addr(x, y, k_src + so), kl1 + dof, 32);
                    noc_async_read(get_noc_addr(x, y, v_src + so), vl1 + dof, 32);
                }
            }
        }
        noc_async_read_barrier();
        DeviceTimestampedData("BA_R", 2);  // (profiling: block rows patched)
    };
#endif
    uint32_t pushed = 0;
    auto q_ready = [&]() {
        invalidate_l1_cache();
#ifdef BA_Q_AFTER
        // (experiment) Q only once this core's first BA_Q_AFTER groups are in (or all of them)
        if (pushed < BA_Q_AFTER && pushed < n_groups) {
            return false;
        }
#endif
        return kv_ready();
    };
    uint32_t issued = 0;
#ifdef BA_SHARE
    // BA_SHARE (2 tile groups, 1 Q tile per core): the tile-group-0 core of each (kv head, slice) reads K / V from DRAM
    // (the leader); its tile-group-1 partner copies each landed group from the leader's L1 (2 NOC reads per group, the
    // same CB addresses) instead: one DRAM read for both Q tiles. The groups holding the block's rows (only readable
    // after QKV, on the critical path) the partner reads from DRAM itself. Program semaphores: BA_LAND_SEM on the
    // partner (groups landed in the leader), BA_COPY_SEM on the leader (groups the partner has copied: a slot is reused
    // after).
    const bool leader = (ci / S) % BA_TQ == 0;
    const uint32_t partner = leader ? ci + S : ci - S;
    const uint32_t pnw = get_common_arg_val<uint32_t>(7 + BA_NIDX + partner / 2) >> (16 * (partner % 2));
    const uint32_t pnx = (pnw >> 8) & 0xFF, pny = pnw & 0xFF;
    volatile tt_l1_ptr uint32_t* land_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(BA_LAND_SEM));
    volatile tt_l1_ptr uint32_t* copy_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(BA_COPY_SEM));
    auto slot_free = [&](uint32_t gi) {  // the partner has copied what this slot held before
        invalidate_l1_cache();
        return gi < cap_groups || *copy_sem >= gi - cap_groups + 1;
    };
    if (!leader) {
        while (pushed < n_groups) {
            invalidate_l1_cache();
            if (!q_done && q_ready()) {
                read_q();
                continue;
            }
            if (!cb_pages_reservable_at_back(cb_k, group_pages) || !cb_pages_reservable_at_back(cb_v, group_pages)) {
                continue;
            }
            if (old_group(pushed)) {
                // prefetched by the leader before QKV ends: copy it from the leader's L1
                if (*land_sem < pushed + 1) {
                    continue;
                }
                const uint32_t off = (pushed % cap_groups) * group_pages * kv_tile_bytes;
                noc_async_read_set_trid(1);
                noc_async_read(get_noc_addr(pnx, pny, k_base + off), k_base + off, group_pages * kv_tile_bytes);
                noc_async_read(get_noc_addr(pnx, pny, v_base + off), v_base + off, group_pages * kv_tile_bytes);
                noc_async_read_set_trid(0);
            } else {
                // holds the block's rows (written by QKV): read straight from DRAM after Q, no hop via the leader
                if (!q_done) {
                    continue;
                }
                issue_group(pushed);
                noc_async_read_barrier_with_trid(1 + pushed % cap_groups);
            }
            noc_async_read_barrier_with_trid(1);
            cb_push_back(cb_k, group_pages);
            cb_push_back(cb_v, group_pages);
            ++pushed;
            DeviceTimestampedData("BA_R", 200 + pushed);  // (profiling: partner group pushed)
            noc_semaphore_inc(get_noc_addr(pnx, pny, get_semaphore(BA_COPY_SEM)), 1);
        }
        issued = n_groups;
    }
#endif
    while (pushed < n_groups) {
#ifdef BA_CAP0
        // before Q: at most BA_CAP0 groups in flight (the QKV weights and every core's first group land sooner)
        while (issued < n_groups && issued - pushed < (q_done ? cap_groups : BA_CAP0) &&
               (q_done || old_group(issued)) &&
#else
        while (issued < n_groups && issued - pushed < cap_groups && (q_done || old_group(issued)) &&
#endif
#ifdef BA_SHARE
               slot_free(issued) &&
#endif
               cb_pages_reservable_at_back(cb_k, (issued - pushed + 1) * group_pages) &&
               cb_pages_reservable_at_back(cb_v, (issued - pushed + 1) * group_pages)) {  // V frees later than K
            issue_group(issued++);
        }
        if (!q_done && q_ready()) {
            read_q();
            continue;
        }
        if (issued > pushed) {
            noc_async_read_barrier_with_trid(1 + pushed % cap_groups);
#ifdef BA_PATCH
            if (block_chunks(pushed)) {
                if (!q_done) {
                    while (!kv_ready()) {  // the source shards (and Q) written
                    }
                    read_q();
                }
                patch(pushed);
            }
#endif
            cb_push_back(cb_k, group_pages);
            cb_push_back(cb_v, group_pages);
            ++pushed;
            DeviceTimestampedData("BA_R", 100 + pushed);  // (profiling: group pushed)
#ifdef BA_SHARE
            noc_semaphore_inc(get_noc_addr(pnx, pny, get_semaphore(BA_LAND_SEM)), 1);  // the partner may copy it
#endif
        } else if (!q_done) {
            while (!kv_ready()) {  // nothing in flight: the next group needs the block's rows
            }
            read_q();
        }
        // else: the CBs are full (compute frees a slot)
    }
    if (!q_done) {
        while (!kv_ready()) {
        }
        read_q();
    }
#ifdef BA_SHARE
    if (leader) {  // the slots stay readable until the partner has copied the last groups
        while (true) {
            invalidate_l1_cache();
            if (*copy_sem >= n_groups) {
                break;
            }
        }
    }
    noc_async_atomic_barrier();
#endif

    // the children's partials (their writers fill slot r of cb_recv and set bit r of the tree semaphore)
    volatile tt_l1_ptr uint32_t* tree_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(BA_TREE_SEM));
    for (uint32_t i = 0; i < sl.recv_rounds; ++i) {
#ifdef BA_PERM
        const uint32_t r = sl.recv_rounds - 1 - i;  // last round first (slot i: ba_writer.cpp)
#else
        const uint32_t r = i;
#endif
        cb_reserve_back(cb_recv, partial_tiles);
        while ((*tree_sem & (1u << r)) == 0) {
            invalidate_l1_cache();
        }
        cb_push_back(cb_recv, partial_tiles);
    }
    if (sl.recv_rounds) {
        noc_semaphore_set(tree_sem, 0);
    }
}
