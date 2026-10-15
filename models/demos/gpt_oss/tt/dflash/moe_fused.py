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

import os

import torch

import ttnn

from ..fused_decode.union import ExpertDownUnion, ExpertGateUpUnion

MOE = [("MOE_FUSED", "1")]


def _wait_n(b):
    """Consumer x-ready count with the multi-core boundary's direct notify (one increment per helper)."""
    return [("BND_WAIT_N", str(b.H))] if getattr(b, "mc", False) and b.direct_notify else []


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
        real_inter=None,
    ):
        """router: the verify pass's router LinearStream (out_mode 4, one core)."""
        self.mesh = mesh_device
        self.boundary, self.router = boundary, router
        assert len(router.assignments) == 1
        rc = router.assignments[0][0]
        taken = [(rc.x, rc.y), (boundary.core.x, boundary.core.y)]
        taken += [(h.x, h.y) for h in getattr(boundary, "helpers", [])] if getattr(boundary, "mc", False) else []
        self.gate_up = ExpertGateUpUnion(
            mesh_device,
            hidden,
            inter_pad,
            swiglu_limit,
            alpha,
            readers=gate_up_readers,
            rows=rows,
            exclude=taken,
            real_inter=real_inter,
        )
        taken += [(c.x, c.y) for c, *_ in self.gate_up.assignments]
        self.down = ExpertDownUnion(
            mesh_device, hidden, inter_pad, readers=down_readers, rows=rows, expert_outer=True, exclude=taken
        )
        self.R = router.cores
        # the router's per-row top-k split over BRISC and NCRISC (kernels/stream_linear_{writer,reader}.cpp)
        self.split_defines = (
            [
                ("MOE_SPLIT_ROWS", "1"),
                ("MOE_SPLIT_SEM", "3"),
                ("MOE_SPLIT_CB_OUT", "16"),
                ("MOE_SPLIT_CB_SCR", "3"),
                ("MOE_SPLIT_TOPK", str(router.heads[0])),
            ]
            if os.environ.get("DFLASH_ROUTER_SPLIT", "1") == "1"
            else []
        )
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
        # Expert prediction (MOE_PRED): per layer, the previous round's P most-routed experts (16 B per layer in the
        # router core's L1), prefetched by gate|up during the boundary + router and processed first if routed again.
        self.pred_p = int(os.environ.get("DFLASH_MOE_PRED", "3"))
        self.pred = ttnn.from_torch(
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

    def __call__(self, pending, router_ws, union, weights, gate_up_ws, act, down_ws, partial, layer=0):
        """pending: the attn-site PendingBoundary (its partial sent by o_proj); runs the boundary, router, gate|up and
        down and sends the down partial to the moe site. Sets pending.residual_out / pending.x."""
        b, mesh = self.boundary, self.mesh
        ctr = self.counters.buffer_address()
        rx, ry = self.router_noc.x, self.router_noc.y
        gd = _noc(mesh, self.G) + _noc(mesh, self.D)
        router_rt = [ctr, len(gd)] + [v for p in gd for v in (p.x, p.y)]
        P = self.pred_p
        pred_addr = self.pred.buffer_address() + 16 * layer
        pdef = [("MOE_PRED", str(P)), ("MOE_PRED_SEM", "4"), ("MOE_PRED_CB", "6")] if P else []
        if P:
            router_rt += [pred_addr, 2 * (self.G.num_cores() + self.D.num_cores())]
        rprog, rio = self.router(
            pending,
            router_ws,
            (union, weights),
            build_only=True,
            consumer_cores=self.R.merge(self.G),
            extra_defines=MOE + self.split_defines + pdef,
            extra_writer_rt=router_rt,
        )
        x = pending.x
        n_g = self.G.num_cores()
        gprog, gio = self.gate_up(
            x,
            union,
            weights,
            gate_up_ws,
            act,
            build_only=True,
            extra_defines=MOE + pdef + _wait_n(b),
            extra_writer_rt=[ctr, rx, ry] + ([pred_addr] if P else []),
            extra_reader_rt=[rx, ry, pred_addr] if P else None,
            pred_p=P,
            x_local_addr=b.x_mirror.buffer_address() if getattr(b, "x_mirror", None) is not None else None,
        )
        dprog, dio = self.down(
            act,
            union,
            weights,
            down_ws,
            partial,
            build_only=True,
            extra_defines=MOE + pdef,
            extra_writer_rt=[ctr, rx, ry, n_g] + ([pred_addr] if P else []),
            notify_args=[1, b.noc_xy[0], b.noc_xy[1], 2],
            extra_reader_rt=[rx, ry, pred_addr] if P else None,
            pred_p=P,
        )
        # semaphore 0: x ready (R, G, B) (+ the multi-core boundary's 5 / 6)
        assert {sd.id for sd in rprog.semaphores} <= {0, 5, 6}
        sems = list(rprog.semaphores) + [
            ttnn.SemaphoreDescriptor(1, ttnn.CoreType.WORKER, self.G.merge(self.D), 0),
            ttnn.SemaphoreDescriptor(2, ttnn.CoreType.WORKER, self.D.merge(b.cores), 0),
        ]
        if self.split_defines:
            sems.append(ttnn.SemaphoreDescriptor(3, ttnn.CoreType.WORKER, self.R, 0))
        if P:
            sems.append(ttnn.SemaphoreDescriptor(4, ttnn.CoreType.WORKER, self.R, 0))
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
        for t in [*rio, *gio, *dio, self.counters, self.pred]:
            if all(t is not u for u in io):
                io.append(t)
        ttnn.generic_op(io, mesh_program)


class DenseMLPFused:
    """The drafter's dense SwiGLU MLP as one program per device: the post-attention boundary (B), gate|up (G) and down
    (D) on disjoint cores. The MLP is split into `slices` fixed "experts" of I_pad / slices columns (union = all of
    them, weight 1), so down starts on slice i once every G core wrote it; with no router both weight streams start
    at launch, during the boundary. Slice counters: program semaphores 3 .. 3 + slices - 1 of B (reset per launch)."""

    CTR_SEM = 3

    def __init__(
        self, mesh_device, boundary, hidden, slice_pad, slices, rows, weight_dtype, gate_up_readers=4, down_readers=4
    ):
        self.mesh, self.boundary, self.slices = mesh_device, boundary, slices
        taken = [(boundary.core.x, boundary.core.y)]
        taken += [(h.x, h.y) for h in getattr(boundary, "helpers", [])] if getattr(boundary, "mc", False) else []
        self.gate_up = ExpertGateUpUnion(
            mesh_device,
            hidden,
            slice_pad,
            0.0,
            1.0,
            readers=gate_up_readers,
            rows=rows,
            dense_silu=True,
            max_experts=slices,
            weight_dtype=weight_dtype,
            exclude=taken,
        )
        taken += [(c.x, c.y) for c, *_ in self.gate_up.assignments]
        self.down = ExpertDownUnion(
            mesh_device,
            hidden,
            slice_pad,
            readers=down_readers,
            rows=rows,
            expert_outer=True,
            max_experts=slices,
            weight_dtype=weight_dtype,
            num_buffers=2,
            exclude=taken,
        )
        self.G, self.D = self.gate_up.cores, self.down.cores
        self.defines = MOE + [("MOE_NO_ROUTER", "1"), ("MOE_CTR_SEM", str(self.CTR_SEM))]

    def __call__(self, pending, union, weights, gate_up_ws, act, down_ws, partial):
        b = self.boundary
        hx, hy = b.noc_xy
        bk, bcbs, bsems, bio = b.consumer_parts(pending, self.G, notify_in_reader=True)
        x = pending.x
        gprog, gio = self.gate_up(
            x,
            union,
            weights,
            gate_up_ws,
            act,
            build_only=True,
            extra_defines=self.defines + _wait_n(b),
            extra_writer_rt=[0, hx, hy],
            x_local_addr=b.x_mirror.buffer_address() if getattr(b, "x_mirror", None) is not None else None,
        )
        dprog, dio = self.down(
            act,
            union,
            weights,
            down_ws,
            partial,
            build_only=True,
            extra_defines=self.defines,
            extra_writer_rt=[0, hx, hy, self.G.num_cores()],
            notify_args=[1, hx, hy, 2],
        )
        sems = list(bsems) + [ttnn.SemaphoreDescriptor(2, ttnn.CoreType.WORKER, self.D.merge(b.cores), 0)]
        sems += [
            ttnn.SemaphoreDescriptor(self.CTR_SEM + s, ttnn.CoreType.WORKER, b.cores, 0) for s in range(self.slices)
        ]
        mesh_program = ttnn.MeshProgramDescriptor()
        n_d = self.D.num_cores()
        for coord in b.coords:
            program = ttnn.ProgramDescriptor(
                kernels=list(bk) + list(gprog.kernels) + list(dprog.kernels),
                semaphores=list(sems),
                cbs=list(bcbs) + list(gprog.cbs) + list(dprog.cbs),
            )
            b._add_sender(coord, program, b._sender_kernel(coord, partial, "moe", producers=n_d, sem_id=2))
            mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = program
        io = []
        for t in [*bio, *gio, *dio]:
            if all(t is not u for u in io):
                io.append(t)
        ttnn.generic_op(io, mesh_program)
