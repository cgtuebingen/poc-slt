import trimesh
from tqdm import tqdm
import torch
import os
from typing import Tuple, Any


def _read_meshes_abc():
    obj_dir = "/graphics/scratch/datasets/ABC/obj/"
    # res_dir = "/graphics/scratch2/staff/zakeri/ABC_dataset_processing"
    data_path = "/graphics/scratch/datasets/ABC/abc_v00_files"

    resolution = 128
    value_range = 1
    num_obj_files_to_process = 5000

    # read the data:
    obj_list = []
    with open(data_path, "r") as file:
        for line in file:
            obj_list.append(line.rstrip("\n"))
    # we used the first 100K for train
    # we use 100K-105K for val
    obj_list_val = obj_list[100000 : 100000 + num_obj_files_to_process]
    assert obj_list_val[0] == obj_list[100000]
    assert obj_list_val[1] == obj_list[100000 + 1]
    assert obj_list_val[-1] == obj_list[100000 + num_obj_files_to_process - 1]
    bbx_val_dict: dict = {}
    for i in tqdm(range(len(obj_list_val)), desc="ABC files test"):
        obj_current = obj_list_val[i]

        mesh_current = trimesh.load(os.path.join(obj_dir, obj_current), force="mesh")
        if isinstance(mesh_current, list):
            if len(mesh_current) == 0:
                continue
        bbx_current = mesh_current.bounds
        bbx_val_dict[obj_current] = bbx_current
    out_name = (
        "/graphics/scratch2/staff/zakeri/LMDBs/ABC_128cube_5KLMDB_Test_with_nonOptimizedLatentCodes/"
        + "mesh_file_names_to_bbx.pkl"
    )
    torch.save(bbx_val_dict, out_name)
    bbx_val_dict_raed = torch.load(out_name)
    assert len(bbx_val_dict_raed) == len(bbx_val_dict)


def normalize_abc_compatible_with_shapenet(
    mesh_file_name_to_bbx: dict, obj_file_name: str
) -> Tuple[Any, Any]:

    obj_bbx = mesh_file_name_to_bbx.get(obj_file_name)
    if obj_bbx is None:
        return None
    else:
        # normalize:
        extents = obj_bbx[1] - obj_bbx[0]
        extents = torch.from_numpy(extents).to(dtype=torch.float32)
        diag_norm = torch.linalg.vector_norm(extents)
        largest_extent = torch.max(extents)
        # scale from side length 2 back to original largest extent
        # then, scale to diagonal length 1

        hausdorff_scale = largest_extent / (2.0 * diag_norm)
        chamfer_scale = hausdorff_scale * hausdorff_scale
        return (hausdorff_scale, chamfer_scale)


# if __name__ == "__main__":
#     # _read_meshes_ABC()
#     out_name = "/graphics/scratch2/staff/zakeri/LMDBs/ABC_128cube_5KLMDB_Test_with_nonOptimizedLatentCodes/" + "mesh_file_names_to_bbx.pkl"
#     bbx_val_dict_raed = torch.load(out_name)
#     breakpoint()
