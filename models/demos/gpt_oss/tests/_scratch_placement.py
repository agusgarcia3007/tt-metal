import ttnn
from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric


@parametrize_mesh_with_fabric([(1, 4)])
def test_placement(mesh_device, device_params):
    print("PLACE grid", mesh_device.compute_with_storage_grid_size())
    for i, c in enumerate(mesh_device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)):
        print("PLACE bank", i, c.x, c.y)


@parametrize_mesh_with_fabric([(1, 4)])
def test_l1_info(mesh_device, device_params):
    info = ttnn._ttnn.reports.get_device_info(mesh_device)
    print(
        "PLACE",
        info.address_at_first_l1_cb_buffer,
        info.address_at_first_l1_bank,
        info.total_l1_for_tensors,
        info.l1_bank_size,
        info.cb_limit,
    )


@parametrize_mesh_with_fabric([(1, 4)])
def test_placement_per_device(mesh_device, device_params):
    for c in range(mesh_device.shape[1]):
        a = mesh_device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0, ttnn.MeshCoordinate(0, c))
        print("PLACEDEV", c, [(b, a[b].x, a[b].y) for b in sorted(a)], flush=True)
