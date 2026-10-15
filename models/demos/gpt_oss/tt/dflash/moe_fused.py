# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The verify pass's MoE as one program per device (MOE_FUSED in the kernels): the post-attention boundary, the router,
the union gate|up and the union down run on disjoint cores and hand over through semaphores instead of op boundaries,
and down starts on union slot i while gate|up is still computing the later slots.

    boundary core B   reader: all-reduce + residual + RMSNorm (attn site) -> x; increments semaphore 0 on R and G
                      sender (BRISC): the moe-site fabric send of the down partial once every D core incremented
                      semaphore 2
    router core R     waits for x (semaphore 0), writes the union + weights, zeroes the per-slot counters (in its
                      L1), increments semaphore 1 on every G and D core
    gate|up cores G   read x after semaphore 0 and the union after semaphore 1 (their weight readers too); after slot i
                      is written, increment counter i on R
    down cores D      weight readers start at semaphore 1 (streaming while gate|up runs); the writers poll counter i
                      until it equals |G|, then hand slot i to compute; at the end they increment semaphore 2 on B

Separate ops would serialize router -> gate|up -> down, each with its own launch and DRAM pipeline ramp."""

import torch

import ttnn

from ..fused_decode.union import ExpertDownUnion, ExpertGateUpUnion

MOE = [("MOE_FUSED", "1")]


def _noc(mesh_device, cores):
    return [mesh_device.worker_core_from_logical_core(c) for c in ttnn.corerange_to_cores(cores, row_wise=True)]


class MoEFused:
    def __init__(
        self,
        mesh_device,
        boundary,
        router,
        hidden,
        inter_pad,
        swiglu_limit,
        alpha,
        rows,
        gate_up_readers=3,
        down_readers=3,
    ):
        """router: the verify pass's router LinearStream (out_mode 4, one core)."""
        self.mesh = mesh_device
        self.boundary, self.router = boundary, router
        assert len(router.assignments) == 1
        rc = router.assignments[0][0]
        taken = [(rc.x, rc.y), (boundary.core.x, boundary.core.y)]
        self.gate_up = ExpertGateUpUnion(
            mesh_device, hidden, inter_pad, swiglu_limit, alpha, readers=gate_up_readers, rows=rows, exclude=taken
        )
        taken += [(c.x, c.y) for c, *_ in self.gate_up.assignments]
        self.down = ExpertDownUnion(
            mesh_device, hidden, inter_pad, readers=down_readers, rows=rows, expert_outer=True, exclude=taken
        )
        self.R = router.cores
        self.G, self.D = self.gate_up.cores, self.down.cores
        self.router_noc = mesh_device.worker_core_from_logical_core(rc)
        # Per-slot counters: 32 x 16 B in the router core's L1 (one shard).
        shard = ttnn.ShardSpec(self.R, [32, 4], ttnn.ShardOrientation.ROW_MAJOR)
        self.counters = ttnn.from_torch(
            torch.zeros(32, 4, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard),
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        print(
            f"MoEFused: router {rc}, boundary {boundary.core}, gate|up {[(c.x, c.y) for c, *_ in self.gate_up.assignments]}"
            f", down {[(c.x, c.y) for c, *_ in self.down.assignments]}",
            flush=True,
        )

    def __call__(self, pending, router_ws, union, weights, gate_up_ws, act, down_ws, partial):
        """pending: the attn-site PendingBoundary (its partial sent by o_proj); runs the boundary, router, gate|up and
        down and sends the down partial to the moe site. Sets pending.residual_out / pending.x."""
        b, mesh = self.boundary, self.mesh
        ctr = self.counters.buffer_address()
        rx, ry = self.router_noc.x, self.router_noc.y
        gd = _noc(mesh, self.G) + _noc(mesh, self.D)
        router_rt = [ctr, len(gd)] + [v for p in gd for v in (p.x, p.y)]
        rprog, rio = self.router(
            pending,
            router_ws,
            (union, weights),
            build_only=True,
            consumer_cores=self.R.merge(self.G),
            extra_defines=MOE,
            extra_writer_rt=router_rt,
        )
        x = pending.x
        n_g = self.G.num_cores()
        gprog, gio = self.gate_up(
            x, union, weights, gate_up_ws, act, build_only=True, extra_defines=MOE, extra_writer_rt=[ctr, rx, ry]
        )
        dprog, dio = self.down(
            act,
            union,
            weights,
            down_ws,
            partial,
            build_only=True,
            extra_defines=MOE,
            extra_writer_rt=[ctr, rx, ry, n_g],
            notify_args=[1, b.noc_xy[0], b.noc_xy[1], 2],
        )
        assert len(rprog.semaphores) == 1  # semaphore 0: x ready (R, G, B)
        sems = list(rprog.semaphores) + [
            ttnn.SemaphoreDescriptor(1, ttnn.CoreType.WORKER, self.G.merge(self.D), 0),
            ttnn.SemaphoreDescriptor(2, ttnn.CoreType.WORKER, self.D.merge(b.cores), 0),
        ]
        mesh_program = ttnn.MeshProgramDescriptor()
        n_d = self.D.num_cores()
        for coord in b.coords:
            program = ttnn.ProgramDescriptor(
                kernels=list(rprog.kernels) + list(gprog.kernels) + list(dprog.kernels),
                semaphores=list(sems),
                cbs=list(rprog.cbs) + list(gprog.cbs) + list(dprog.cbs),
            )
            sender = b._sender_kernel(coord, partial, "moe", producers=n_d, sem_id=2)
            b._add_sender(coord, program, sender)
            mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = program
        io = []
        for t in [*rio, *gio, *dio, self.counters]:
            if all(t is not u for u in io):
                io.append(t)
        ttnn.generic_op(io, mesh_program)
