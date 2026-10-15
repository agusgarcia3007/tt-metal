# Scratch: L1 headroom of the DFlash pass programs: with an extra lockstep L1 buffer of X KB per core allocated, do the
# eager draft / verify / context passes still fit (no circular buffer / L1 buffer clash)?
import os

import pytest
import torch

import ttnn
from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric


@pytest.mark.timeout(3600)
@parametrize_mesh_with_fabric([(1, 4)])
def test_l1_headroom(mesh_device, device_params):
    import models.demos.gpt_oss.demo.text_demo as text_demo
    from models.demos.gpt_oss.config import MeshConfig, ModeConfig
    from models.demos.gpt_oss.tt.dflash.draft import DFlashDraft
    from models.demos.gpt_oss.tt.dflash.runner import FastDFlashRunner
    from models.tt_transformers.tt.generator import Generator

    ctx = 4096
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=4, ep=1))
    model_args, model, page_table, kv_cache, tokenizer, processor, _ = text_demo.prepare_gpt_oss_generator_args(
        num_devices=4,
        data_parallel=1,
        mesh_device=mesh_device,
        global_batch_size=1,
        optimizations=None,
        max_seq_len=ctx,
        page_params={"page_block_size": 64, "page_max_num_blocks_per_dp": ctx // 64},
        paged_attention=True,
        mesh_config=mesh_config,
        state_dict=None,
    )
    generator = Generator(model, model_args, mesh_device, processor=processor, tokenizer=tokenizer)
    m = model[0]
    draft = DFlashDraft(mesh_device, m, max_positions=ctx)
    fast = FastDFlashRunner(generator, m, page_table, kv_cache, draft)
    e = tokenizer.apply_chat_template([{"role": "user", "content": "What is 2+2?"}], add_generation_prompt=True)
    ids = [int(t) for t in (e["input_ids"] if isinstance(e, dict) or hasattr(e, "input_ids") else e)]
    fast.generate(ids, 16)
    if os.environ.get("L1_LIST"):
        bufs = [
            b for b in ttnn._ttnn.reports.get_buffers(mesh_device) if "L1" in str(b.buffer_type) and b.device_id == 0
        ]
        bufs.sort(key=lambda b: b.address)
        for b in bufs:
            print(f"L1BUF addr {b.address} size {b.max_size_per_bank} layout {b.buffer_layout}", flush=True)
        return
    grid = mesh_device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    n_cores = grid.x * grid.y
    for kb in [int(x) for x in os.environ.get("L1_KB", "64,128,256,384,512").split(",")]:
        tiles = kb // 2
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(cores, (32, 32 * tiles), ttnn.ShardOrientation.ROW_MAJOR),
        )
        try:
            t = ttnn.from_torch(
                torch.zeros(1, 1, 32 * n_cores, 32 * tiles),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=mc,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
        except Exception as ex:
            print(f"L1HR {kb} KB: allocation failed: {str(ex)[:200]}", flush=True)
            break
        try:
            fast.generate(ids, 16)
            ttnn.synchronize_device(mesh_device)
            print(f"L1HR {kb} KB per core (buffer at {t.buffer_address()}): passes OK", flush=True)
        except Exception as ex:
            print(f"L1HR {kb} KB per core (buffer at {t.buffer_address()}): FAILED {str(ex)[:300]}", flush=True)
            t.deallocate(True)
            break
        t.deallocate(True)
