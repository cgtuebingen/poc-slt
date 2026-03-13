from typing import Tuple, Any

import numpy as np


def extract_keys_from_mesh_file_name(mesh_file_name):
    prefix = "/ceph/ruppert/ShapeNetCorev2_remeshed_0.008/ShapeNetCore.v2/"
    suffix = "/models/model_normalized.obj"

    sub_name = mesh_file_name.replace(prefix, "").replace(suffix, "")
    return sub_name


def extract_bbx_with_mesh_file_name(
    mesh_file_name: str, bbx_list: list
) -> Tuple[Any, Any]:

    key_info = extract_keys_from_mesh_file_name(mesh_file_name)
    bbox_info = [y for y in bbx_list if key_info in y["mesh_file_name"]]
    assert len(bbox_info) == 1

    aabb = bbox_info[0]["bbx"]

    return (key_info, aabb)


def calculate_scales_for_metrics(aabb) -> Tuple[Any, Any]:
    extents = aabb[1] - aabb[0]
    d = np.max(extents)

    hausdorff_scale = d / 2

    chamfer_scale = d * d / 4

    # fscore1% -> threshold  0.02 ( our side length is 2 and not 1!)

    return (hausdorff_scale, chamfer_scale)


def calculate_metric_scale_for_mesh_file_name(
    mesh_file_name: str, bbx_list: list
) -> Tuple[Tuple[Any, Any], Tuple[Any, Any]]:

    key_info, aabb = extract_bbx_with_mesh_file_name(mesh_file_name, bbx_list)

    hausdorff_scale, chamfer_scale = calculate_scales_for_metrics(aabb)

    return ((key_info, aabb), (hausdorff_scale, chamfer_scale))


# if __name__ == "__main__":
#
#     bbx_path = "/graphics/scratch2/staff/zakeri/all_mesh_file_names_shapenetCorev1_55/all_mesh_file_bbx.pkl"
#     bbx_list = torch.load(bbx_path)
#
#     mesh_file_name = "/ceph/ruppert/ShapeNetCorev2_remeshed_0.008/ShapeNetCore.v2/03001627/c98e1a3e61caec6a67d783b4714d4324/models/model_normalized.obj"
#     key_info, aabb = extract_bbx_with_mesh_file_name(mesh_file_name, bbx_list)
#
#     calculate_scales_for_metrics(aabb)
