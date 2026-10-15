# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The z-lab/gpt-oss-20b-DFlash drafter on a 1 x D mesh (dflash/reference.py describes the model).

Tensor parallel like a dense target layer: per device the drafter holds 64 / D query heads and 8 / D KV heads (q / k /
v column-parallel, o_proj row-parallel), 7680 / D MLP columns (gate / up column-parallel, down row-parallel) and
14400 / D input rows of fc; every row-parallel product is all-reduced (BF16), so the residual is replicated. The norms
are replicated. RoPE is the target's own (identical YaRN parameters), so q / k weights, biases and q_norm / k_norm
weights are permuted to the interleaved layout of ttnn's rotary_embedding_llama, as the target's.

Each activation is one 32-row tile, one row per "user" of ttnn's decode attention ops: the context and noise rows of a
round are written into a paged draft KV cache (one page table, repeated per row) at their own positions, then the
noise rows attend (paged SDPA decode, each with cur_pos = the block's last position, so the block is not causal).

Two calls per round (DFlashDraft):
  add_context(features [n, 5 * hidden], positions): fc + hidden_norm, then every layer's context K / V into the cache
      (the context does not pass through the layers: each layer projects the same normed context);
  draft(last_token, start): the noise block [last_token, mask x 7] at positions start .. start + 7 through the 8 layers,
      the final norm and the target's LM head; returns the 7 draft tokens (rows 1..7).
"""

import torch

import ttnn
from models.tt_transformers.tt.load_checkpoints import reverse_permute, reverse_permute_1d

from ..fused_decode.config import residual_memory_config, width_sharded_memory_config
from .reference import draft_snapshot, load_draft_config

TILE = ttnn.TILE_SIZE
ROWS = 32  # one tile of rows: up to 32 context rows per add_context call, 8 noise rows per draft


class _AllReduce:
    """One-op BF16 all-reduce of a [1, 1, 32, hidden] partial over the mesh columns (batched_decode.DecodeAllReduce in
    BF16: the drafter's fc reduces 14400 products)."""

    def __init__(self, mesh_device, hidden):
        from ..fused_decode.config import RESIDUAL_CORES

        self.mesh_device = mesh_device
        n = mesh_device.shape[1]
        self.memory_config = residual_memory_config(mesh_device, hidden)
        grid = mesh_device.compute_with_storage_grid_size()
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        self.slots = [
            (
                ttnn.empty(
                    [1, 1, ROWS, hidden * n],
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    memory_config=width_sharded_memory_config(mesh_device, hidden * n, RESIDUAL_CORES),
                ),
                ttnn.create_global_semaphore(mesh_device, cores, 0),
            )
            for _ in range(2)
        ]
        self.i = 0

    def __call__(self, partial):
        moved = ttnn.to_memory_config(partial, self.memory_config, dtype=ttnn.bfloat16)
        partial.deallocate(True)
        scratch, sem = self.slots[self.i]
        self.i ^= 1
        out = ttnn.experimental.all_reduce_async(
            moved,
            scratch,
            cluster_axis=1,
            mesh_device=self.mesh_device,
            multi_device_global_semaphore=sem,
            memory_config=self.memory_config,
            dtype=ttnn.bfloat16,
            topology=ttnn.Topology.Ring,
            num_links=1,
        )
        moved.deallocate(True)
        flat = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        out.deallocate(True)
        return flat


class DFlashDraft:
    def __init__(
        self, mesh_device, target, max_positions, path=None, weight_dtype=ttnn.bfloat16, block_size=64, users=1
    ):
        """target: the gpt-oss Model (embedding table, LM head). max_positions: the longest sequence drafted for.
        users: sequences with their own draft cache (batched DFlash); a draft call takes up to 32 / 8 = 4 of them."""
        from safetensors.torch import load_file

        self.mesh_device = mesh_device
        self.target = target
        self.path = path or draft_snapshot()
        c = load_draft_config(self.path)
        self.cfg = c
        D = mesh_device.shape[1]
        self.D = D
        self.hidden = c["hidden_size"]
        self.hd = c["head_dim"]
        self.nq, self.nkv = c["num_attention_heads"] // D, c["num_key_value_heads"] // D
        self.inter = c["intermediate_size"] // D
        self.layers = c["num_hidden_layers"]
        self.eps = c["rms_norm_eps"]
        self.block = c["block_size"]
        self.mask_token_id = c["dflash_config"]["mask_token_id"]
        self.target_layer_ids = c["dflash_config"]["target_layer_ids"]
        self.ctx_in = len(self.target_layer_ids) * self.hidden
        w = load_file(str(self.path / "model.safetensors"))
        mesh = mesh_device
        rep = ttnn.ReplicateTensorToMesh(mesh)

        def put(t, mapper=rep, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                t.contiguous(),
                dtype=dtype,
                layout=layout,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )

        def norm_w(t):
            return put(t.float().reshape(1, 1, -1, TILE), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        cols = ttnn.ShardTensorToMesh(mesh, dim=-1)
        rows = ttnn.ShardTensorToMesh(mesh, dim=-2)
        hd, nq, nkv = self.hd, self.nq, self.nkv
        self.fc = put(w["fc.weight"].T, rows)  # [14400, hidden], rows split (decode rounds: all-reduced)
        self.fc_full = put(w["fc.weight"].T)  # replicated (prompt context: no all-reduce over thousands of rows)
        self.hidden_norm = norm_w(w["hidden_norm.weight"])
        self.norm = norm_w(w["norm.weight"])
        self.L = []
        for i in range(self.layers):
            p = f"layers.{i}."
            a = p + "self_attn."
            q = reverse_permute(w[a + "q_proj.weight"], D * nq, D * nq * hd, self.hidden)
            k = reverse_permute(w[a + "k_proj.weight"], D * nkv, D * nkv * hd, self.hidden)
            v = w[a + "v_proj.weight"]
            qb = reverse_permute(w[a + "q_proj.bias"][:, None], D * nq, D * nq * hd, 1)[:, 0]
            kb = reverse_permute(w[a + "k_proj.bias"][:, None], D * nkv, D * nkv * hd, 1)[:, 0]
            vb = w[a + "v_proj.bias"]
            # Per device: [q heads | k heads | v heads] columns (nlp_create_qkv_heads_decode's layout).
            qkv = torch.cat(
                [
                    torch.cat(
                        [
                            q[d * nq * hd : (d + 1) * nq * hd],
                            k[d * nkv * hd : (d + 1) * nkv * hd],
                            v[d * nkv * hd : (d + 1) * nkv * hd],
                        ]
                    )
                    for d in range(D)
                ]
            )
            qkvb = torch.cat(
                [
                    torch.cat(
                        [
                            qb[d * nq * hd : (d + 1) * nq * hd],
                            kb[d * nkv * hd : (d + 1) * nkv * hd],
                            vb[d * nkv * hd : (d + 1) * nkv * hd],
                        ]
                    )
                    for d in range(D)
                ]
            )
            m = p + "mlp."
            g, u = w[m + "gate_proj.weight"], w[m + "up_proj.weight"]
            I = self.inter
            gu = torch.cat([torch.cat([g[d * I : (d + 1) * I], u[d * I : (d + 1) * I]]) for d in range(D)])
            kv = torch.cat(
                [
                    torch.cat([k[d * nkv * hd : (d + 1) * nkv * hd], v[d * nkv * hd : (d + 1) * nkv * hd]])
                    for d in range(D)
                ]
            )
            kvb = torch.cat(
                [
                    torch.cat([kb[d * nkv * hd : (d + 1) * nkv * hd], vb[d * nkv * hd : (d + 1) * nkv * hd]])
                    for d in range(D)
                ]
            )
            self.L.append(
                dict(
                    kv=put(kv.T, cols),
                    kv_b=put(kvb.reshape(1, -1), cols, dtype=ttnn.bfloat16),
                    ln1=norm_w(w[p + "input_layernorm.weight"]),
                    ln2=norm_w(w[p + "post_attention_layernorm.weight"]),
                    qkv=put(qkv.T, cols),
                    qkv_b=put(qkvb.reshape(1, -1), cols, dtype=ttnn.bfloat16),
                    q_norm=norm_w(reverse_permute_1d(w[a + "q_norm.weight"])),
                    k_norm=norm_w(reverse_permute_1d(w[a + "k_norm.weight"])),
                    o=put(w[a + "o_proj.weight"].T, rows),
                    o_b=put(w[a + "o_proj.bias"].reshape(1, -1), dtype=ttnn.bfloat16),
                    gu=put(gu.T, cols),
                    down=put(w[m + "down_proj.weight"].T, rows),
                )
            )
        # Paged draft KV cache, one page table row per "user" row (all identical: the rows are one sequence).
        self.block_size = block_size
        self.num_blocks = -(-(max_positions + self.block) // block_size)  # per user
        self.users = users
        self.user_tables = torch.arange(users * self.num_blocks, dtype=torch.int32).reshape(users, self.num_blocks)
        self.caches = []
        heads_split = ttnn.ShardTensorToMesh(mesh, dim=1)
        for _ in range(self.layers):
            z = torch.zeros(users * self.num_blocks, D * nkv, block_size, hd)
            self.caches.append(
                tuple(
                    ttnn.from_torch(
                        z,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=mesh,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=heads_split,
                    )
                    for _ in range(2)
                )
            )
        self.page_table = self._page_table([0] * ROWS)
        from ..model import create_rope_setup

        self.rope = create_rope_setup(mesh, target.hf_config, max_local_batch_size=ROWS, datatype=ttnn.bfloat16)
        self.trans_mat = self.rope.get_both_trans_mats()["decode"]
        self.all_reduce = _AllReduce(mesh, self.hidden)
        self.head_mc = ttnn.create_sharded_memory_config(
            shape=(TILE, hd),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.sdpa_pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8), q_chunk_size=0, k_chunk_size=128, exp_approx_mode=False
        )
        self.ck = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        self.mm_ck = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )

    # -- helpers -------------------------------------------------------------------------------------------------
    def _page_table(self, row_users):
        """[32, blocks] device page table: row r = the table of user row_users[r] (rows past the list: user 0)."""
        rows = list(row_users) + [0] * (ROWS - len(row_users))
        return ttnn.from_torch(
            self.user_tables[rows].contiguous(),
            dtype=ttnn.int32,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _rows(self, t, mapper=None):
        """Host [n <= 32, w] -> device [1, 1, 32, w] BF16 (zero rows after n)."""
        x = torch.zeros(1, 1, ROWS, t.shape[-1])
        x[0, 0, : t.shape[0]] = t.float()
        return ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper or ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _pos(self, positions):
        """Per-row positions (-1: row skipped) -> INT32 [32] device tensor and RoPE rows."""
        p = torch.full((ROWS,), -1, dtype=torch.int32)
        p[: len(positions)] = torch.as_tensor(list(positions), dtype=torch.int32)
        tt = ttnn.from_torch(
            p, dtype=ttnn.int32, device=self.mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device)
        )
        rot = self.rope.get_rot_mats(torch.clamp(p, min=0))
        return tt, rot

    def _heads(self, x, layer):
        """[1, 1, 32, hidden] rows -> rotated / normed q [1, 32, nq, hd], k, v [1, 32, nkv, hd] (height-sharded)."""
        L = self.L[layer]
        xqkv = ttnn.linear(
            x,
            L["qkv"],
            bias=L["qkv_b"],
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.mm_ck,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv, num_heads=self.nq, num_kv_heads=self.nkv, memory_config=self.head_mc
        )
        xqkv.deallocate(True)
        return q, k, v

    def _norm_rope(self, t, w, rot):
        flat = ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)
        t.deallocate(True)
        normed = ttnn.rms_norm(flat, epsilon=self.eps, weight=w, compute_kernel_config=self.ck)
        flat.deallocate(True)
        sharded = ttnn.to_memory_config(normed, self.head_mc)
        normed.deallocate(True)
        out = ttnn.experimental.rotary_embedding_llama(sharded, rot[0], rot[1], self.trans_mat, is_decode_mode=True)
        sharded.deallocate(True)
        return out

    def _write_kv(self, layer, k, v, positions, page_table):
        """One paged_update_cache per row: the rows are one sequence, and the op's per-row read-modify-write of a cache
        tile races when several rows of a call share a tile."""
        kc, vc = self.caches[layer]
        for r, p in enumerate(positions):
            idx = torch.full((ROWS,), -1, dtype=torch.int32)
            idx[r] = p
            tt = ttnn.from_torch(
                idx, dtype=ttnn.int32, device=self.mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device)
            )
            ttnn.experimental.paged_update_cache(kc, k, update_idxs_tensor=tt, page_table=page_table)
            ttnn.experimental.paged_update_cache(vc, v, update_idxs_tensor=tt, page_table=page_table)
            tt.deallocate(True)

    # -- the two calls of a round --------------------------------------------------------------------------------
    def add_prompt_context(self, features, chunk=2048, user=0):
        """Context of prompt positions 0 .. n - 1 (prefill-style: [1, heads, rows, head_dim] K / V, paged_fill_cache of
        block-aligned chunks; positions past n in the last chunk are rewritten before they are read)."""
        n = features.shape[0]
        mesh = self.mesh_device
        pt_host = self.user_tables[user][None]
        for s0 in range(0, n, chunk):
            rows = min(chunk, -(-(n - s0) // self.block_size) * self.block_size)
            x = torch.zeros(1, 1, rows, self.ctx_in)
            x[0, 0, : min(rows, n - s0)] = features[s0 : s0 + rows].float()
            tx = ttnn.from_torch(
                x,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            ctx = ttnn.linear(tx, self.fc_full, dtype=ttnn.bfloat16, compute_kernel_config=self.mm_ck)
            tx.deallocate(True)
            ctx = ttnn.rms_norm(ctx, epsilon=self.eps, weight=self.hidden_norm, compute_kernel_config=self.ck)
            cos = self.rope.cos_matrix_prefill[:, :, s0 : s0 + rows, :]
            sin = self.rope.sin_matrix_prefill[:, :, s0 : s0 + rows, :]
            trans = self.rope.get_both_trans_mats()["prefill"]
            b0 = s0 // self.block_size
            chunk_pt = ttnn.from_torch(
                pt_host[:, b0 : b0 + rows // self.block_size].contiguous(),
                dtype=ttnn.int32,
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            for i in range(self.layers):
                L = self.L[i]
                kv = ttnn.linear(ctx, L["kv"], bias=L["kv_b"], dtype=ttnn.bfloat16, compute_kernel_config=self.mm_ck)
                width = self.nkv * self.hd
                for j, cache in enumerate(self.caches[i]):
                    t = ttnn.slice(kv, (0, 0, 0, j * width), (1, 1, rows, (j + 1) * width))
                    t = ttnn.transpose(ttnn.reshape(t, [1, rows, self.nkv, self.hd]), 1, 2)  # [1, heads, rows, hd]
                    if j == 0:
                        t = ttnn.rms_norm(t, epsilon=self.eps, weight=L["k_norm"], compute_kernel_config=self.ck)
                        t = ttnn.experimental.rotary_embedding_llama(t, cos, sin, trans, is_decode_mode=False)
                    ttnn.experimental.paged_fill_cache(cache, t, chunk_pt, batch_idx=0)
                    t.deallocate(True)
                kv.deallocate(True)
            chunk_pt.deallocate(True)
            ctx.deallocate(True)

    def add_context(self, features, positions, user=0):
        """features: host [n, 5 * hidden] target hidden states (target_layer_ids order) of positions `positions`."""
        n = features.shape[0]
        for s in range(0, n, ROWS):
            chunk, pos_list = features[s : s + ROWS], list(positions)[s : s + ROWS]
            x = self._rows(chunk, ttnn.ShardTensorToMesh(self.mesh_device, dim=-1))
            partial = ttnn.linear(x, self.fc, dtype=ttnn.bfloat16, compute_kernel_config=self.mm_ck)
            x.deallocate(True)
            summed = self.all_reduce(partial)
            ctx = ttnn.rms_norm(summed, epsilon=self.eps, weight=self.hidden_norm, compute_kernel_config=self.ck)
            summed.deallocate(True)
            pos, rot = self._pos(pos_list)
            table = self._page_table([user] * len(pos_list))
            for i in range(self.layers):
                q, k, v = self._heads(ctx, i)
                q.deallocate(True)
                k = self._norm_rope(k, self.L[i]["k_norm"], rot)
                self._write_kv(i, k, v, pos_list, table)
                k.deallocate(True)
                v.deallocate(True)
            ctx.deallocate(True)

    def draft(self, last_token, start, user=0):
        """The 7 draft tokens after `last_token` (at position `start`); the context up to start - 1 is in the cache."""
        return self.draft_users([last_token], [start], [user])[0]

    def draft_users(self, last_tokens, starts, users):
        """Up to 32 / 8 users at once: user j's block is rows 8 j .. 8 j + 7. Returns each user's 7 draft tokens."""
        mesh = self.mesh_device
        B = self.block
        n = len(users)
        assert n * B <= ROWS
        toks = torch.full((1, ROWS), self.mask_token_id, dtype=torch.int32)
        for j, t in enumerate(last_tokens):
            toks[0, j * B] = t
        tt_toks = ttnn.from_torch(toks, dtype=ttnn.uint32, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        emb = ttnn.embedding(tt_toks, self.target.embedding_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        # FP32 residual: the drafter's residual reaches ~5e3 with outlier channels, where BF16 sums lose the rest.
        h = ttnn.typecast(ttnn.reshape(emb, [1, 1, ROWS, self.hidden]), ttnn.float32)
        positions = [s0 + i for s0 in starts for i in range(B)]
        pos, rot = self._pos(positions)
        table = self._page_table([u for u in users for _ in range(B)])
        # Attention bound: every noise row sees every position up to its block's last one (not causal).
        bound = torch.full((ROWS,), -1, dtype=torch.int32)
        for j, s0 in enumerate(starts):
            bound[j * B : (j + 1) * B] = s0 + B - 1
        tt_bound = ttnn.from_torch(bound, dtype=ttnn.int32, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        for i in range(self.layers):
            L = self.L[i]
            x = ttnn.rms_norm(h, epsilon=self.eps, weight=L["ln1"], compute_kernel_config=self.ck)
            q, k, v = self._heads(x, i)
            x.deallocate(True)
            q = self._norm_rope(q, L["q_norm"], rot)
            k = self._norm_rope(k, L["k_norm"], rot)
            self._write_kv(i, k, v, positions, table)
            k.deallocate(True)
            v.deallocate(True)
            kc, vc = self.caches[i]
            attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q,
                kc,
                vc,
                cur_pos_tensor=tt_bound,
                page_table_tensor=table,
                scale=self.hd**-0.5,
                program_config=self.sdpa_pc,
                compute_kernel_config=self.ck,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            q.deallocate(True)
            attn_s = ttnn.to_memory_config(attn, self.head_mc)
            attn.deallocate(True)
            heads = ttnn.experimental.nlp_concat_heads_decode(attn_s, num_heads=self.nq)
            attn_s.deallocate(True)
            partial = ttnn.linear(heads, L["o"], dtype=ttnn.bfloat16, compute_kernel_config=self.mm_ck)
            heads.deallocate(True)
            o = self.all_reduce(partial)
            o = ttnn.add(o, L["o_b"], output_tensor=o)
            h2 = ttnn.add(h, o, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.float32)
            h.deallocate(True)
            o.deallocate(True)
            h = h2
            x = ttnn.rms_norm(h, epsilon=self.eps, weight=L["ln2"], compute_kernel_config=self.ck)
            gu = ttnn.linear(x, L["gu"], dtype=ttnn.bfloat16, compute_kernel_config=self.mm_ck)
            x.deallocate(True)
            I = self.inter
            g = ttnn.slice(gu, (0, 0, 0, 0), (1, 1, ROWS, I))
            u = ttnn.slice(gu, (0, 0, 0, I), (1, 1, ROWS, 2 * I))
            gu.deallocate(True)
            act = ttnn.multiply(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
            g.deallocate(True)
            u.deallocate(True)
            partial = ttnn.linear(act, L["down"], dtype=ttnn.bfloat16, compute_kernel_config=self.mm_ck)
            act.deallocate(True)
            mlp = self.all_reduce(partial)
            h2 = ttnn.add(h, mlp, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.float32)
            h.deallocate(True)
            mlp.deallocate(True)
            h = h2
            if getattr(self, "debug_layers", None) is not None:
                self.debug_layers.append(ttnn.to_torch(ttnn.get_device_tensors(h)[0]).float()[0, 0, : self.block])
        x = ttnn.rms_norm(h, epsilon=self.eps, weight=self.norm, compute_kernel_config=self.ck)
        h.deallocate(True)
        self.last_hidden = x  # [1, 1, 32, hidden] (rows 0..7), for tests
        logits = ttnn.linear(x, self.target.lm_head_weight, dtype=ttnn.bfloat16, compute_kernel_config=self.mm_ck)
        full = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1))[0, 0].float()
        logits.deallocate(True)
        V = self.target.vocab_size
        ids = full[:, :V].argmax(-1)
        return [ids[j * B + 1 : (j + 1) * B].tolist() for j in range(n)]
