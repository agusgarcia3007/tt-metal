# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""On-device merge of the multi-row LM head's per-core argmax lists (LinearStream out_mode 6) across cores and devices
(fused_decode/kernels/argmax_merge.cpp): per device a merge into one UINT32 tile, an all-gather of the 4 tiles, and a
final merge on every device that writes either the verify block [last token, 7 drafts] (kind 0: drafts go straight
from the drafter to the verify pass) or [block, 8 target argmaxes] (kind 1: the host's single read per round)."""

import torch

import ttnn

from ..fused_decode.boundary import KERNEL_DIR

CORE = ttnn.CoreCoord(0, 0)


class ArgmaxGather:
    def __init__(self, mesh_device, rows, n_lists, vocab_per_device, kind):
        self.mesh_device = mesh_device
        self.rows, self.n_lists, self.vd, self.kind = rows, n_lists, vocab_per_device, kind
        rep = ttnn.ReplicateTensorToMesh(mesh_device)
        self.dev_tile = ttnn.from_torch(
            torch.zeros(1, 1, 32, 32, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        self.gathered = ttnn.from_torch(
            torch.zeros(1, 1, 32 * mesh_device.get_num_devices(), 32, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(CORE, CORE)])
        self.coords = [ttnn.MeshCoordinate(0, c) for c in range(mesh_device.shape[1])]

    def _kernel(self, ct, rt_args, scratch, defines=None):
        rt = ttnn.RuntimeArgs()
        rt[CORE.x][CORE.y] = rt_args
        k = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "argmax_merge.cpp"),
            core_ranges=self.cores,
            compile_time_args=ct,
            runtime_args=rt,
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0),
            defines=list(defines or []),
        )
        cb = ttnn.CBDescriptor(
            total_size=scratch,
            core_ranges=self.cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=scratch)],
        )
        return ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[cb])

    def __call__(self, lists, tokens_src, out, accept=None, rows_cap=0):
        """lists: the LM head's [1, n_lists * rows * 2] UINT32 lists; tokens_src: kind 0 the drafter's token input
        (element 0 = the block's first token), kind 1 the verify block tokens; out: kind 0 the verify token input
        [1, rows] UINT32, kind 1 a [1, 2 * rows] UINT32 result buffer."""
        rows = self.rows
        lists_page = self.n_lists * rows * 8
        ct0 = [0, rows, self.n_lists, self.kind, 0, lists_page, 4096, 64]
        ct0 += ttnn.TensorAccessorArgs(lists).get_compile_time_args()
        ct0 += ttnn.TensorAccessorArgs(self.dev_tile).get_compile_time_args()
        ct0 += ttnn.TensorAccessorArgs(self.dev_tile).get_compile_time_args()
        mesh_program = ttnn.MeshProgramDescriptor()
        for coord in self.coords:
            d = coord[1]
            prog = self._kernel(
                ct0, [lists.buffer_address(), self.dev_tile.buffer_address(), d * self.vd], max(lists_page, 4096)
            )
            mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = prog
        ttnn.generic_op([lists, self.dev_tile], mesh_program)
        ttnn.all_gather(
            self.dev_tile, dim=2, cluster_axis=1, output_tensor=self.gathered, topology=ttnn.Topology.Linear
        )
        n_dev = self.mesh_device.shape[1]
        src_page = tokens_src.shape[-1] * 4
        out_page = out.shape[-1] * 4
        ct1 = [1, rows, n_dev, self.kind, 0, 4096, out_page, src_page]
        for t in (self.gathered, out, tokens_src):
            ct1 += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        rt1 = [self.gathered.buffer_address(), out.buffer_address(), tokens_src.buffer_address()]
        io = [self.gathered, tokens_src, out]
        defines = None
        if accept is not None:  # (mask id, [vpos, t_toks, t_pos, t_rot, t_bound, t_cpos, t_crot]): device rounds
            mask_id, ts = accept
            for t in ts:
                ct1 += ttnn.TensorAccessorArgs(t).get_compile_time_args()
            rt1 += [t.buffer_address() for t in ts]
            io += list(ts)
            defines = [("ACCEPT", "1"), ("ACCEPT_MASK_ID", str(mask_id))]
            if rows_cap:  # only rows [0, rows_cap) were verified: accepted drafts <= rows_cap - 1
                defines.append(("ACCEPT_ROWS", str(rows_cap)))
        prog = self._kernel(ct1, rt1, n_dev * 4096 + max(src_page, 64) + 1024, defines)
        ttnn.generic_op(io, prog)
