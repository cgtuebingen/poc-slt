import os
import mcubes
import numpy as np


def make_mcubes_from_voxels(
    voxel: np.ndarray,
    current_epoch: int,
    current_global_step: int,
    current_batch,
    file_name: str,
    result_dir: str,
):
    voxel__copy = np.array(voxel, copy=True)
    vertices_, triangles_ = mcubes.marching_cubes(voxel__copy, 0)

    if voxel__copy.shape == (128, 128, 128):
        vertices_ = (
            vertices_ / 64.0 - 1.0
        )  # assuming the resolution is 128 , bbx of the voxel changes by mcubes, we need to turn it back
        export_file_name_obj = (
            result_dir
            + str(file_name)
            + "-"
            + "epoch"
            + "="
            + str(current_epoch)
            + "-"
            + "global_step"
            + "="
            + str(current_global_step)
            + ".obj"
        )
        mcubes.export_obj(vertices_, triangles_, export_file_name_obj)
    else:
        export_file_name_dae = (
            result_dir
            + str(file_name)
            + "-"
            + "epoch"
            + "="
            + str(current_epoch)
            + "-"
            + "global_step"
            + "="
            + str(current_global_step)
            + ".dae"
        )
        mcubes.export_mesh(vertices_, triangles_, export_file_name_dae, str(file_name))


def make_mcubes_from_voxels_obj(
    voxel: np.ndarray,
    current_epoch: int,
    current_global_step: int,
    current_batch,
    file_name: str,
    result_dir: str,
):
    voxel__copy = np.array(voxel, copy=True)
    vertices_, triangles_ = mcubes.marching_cubes(voxel__copy, 0)

    vertices_ = vertices_ / (vertices_.shape[0] * 0.5) - 1.0

    export_file_name_obj = (
        result_dir
        + str(file_name)
        + "-"
        + "epoch"
        + "="
        + str(current_epoch)
        + "-"
        + "global_step"
        + "="
        + str(current_global_step)
        + ".obj"
    )
    mcubes.export_obj(vertices_, triangles_, export_file_name_obj)


def make_mcubes_from_voxels_obj_for_eval(
    voxel: np.ndarray, object_index: int, file_name: str, result_dir: str
) -> str:
    voxel__copy = np.array(voxel, copy=True)
    vertices_, triangles_ = mcubes.marching_cubes(voxel__copy, 0)

    if voxel__copy.shape == (128, 128, 128):
        vertices_ = (
            vertices_ / 64.0 - 1.0
        )  # assuming the resolution is 128 , bbx of the voxel changes by mcubes, we need to turn it back

    export_file_name_obj = (
        result_dir + str(file_name) + "-" + "ObjID=" + str(object_index) + "_" + ".obj"
    )
    mcubes.export_obj(vertices_, triangles_, export_file_name_obj)
    return export_file_name_obj


def make_mcubes_from_voxels_obj_for_pad(
    voxel: np.ndarray, object_index: int, file_name: str, result_dir: str
) -> str:
    voxel__copy = np.array(voxel, copy=True)
    voxel__copy = np.pad(voxel__copy, pad_width=1, constant_values=-1)
    vertices_, triangles_ = mcubes.marching_cubes(voxel__copy, 0)

    if voxel__copy.shape == (128 + 2, 128 + 2, 128 + 2):
        # add +0.5 for center
        # subtract 1 for padding
        # divide by half res ==> extents of 2
        # subtract 1 ==> center to 0, bbox [-1, 1]
        vertices_ = (
            vertices_ - 0.5
        ) / 64.0 - 1.0  # assuming the resolution is 128 , bbx of the voxel changes by mcubes, we need to turn it back
    elif voxel__copy.shape == (32 + 2, 32 + 2, 32 + 2):
        # add +0.5 for center
        # subtract 1 for padding
        # divide by half res ==> extents of 2
        # subtract 1 ==> center to 0, bbox [-1, 1]
        vertices_ = (
            vertices_ - 0.5
        ) / 16.0 - 1.0  # assuming the resolution is 32 , bbx of the voxel changes by mcubes, we need to turn it back
    else:
        raise Exception("weird shape")

    export_file_name_obj = os.path.join(
        result_dir, str(file_name) + "-" + "ObjID=" + str(object_index) + "_" + ".obj"
    )
    mcubes.export_obj(vertices_, triangles_, export_file_name_obj)
    return export_file_name_obj


def make_mcubes_from_voxels_dae(
    voxel: np.ndarray,
    current_epoch: int,
    current_global_step: int,
    current_batch,
    file_name: str,
    result_dir: str,
):
    voxel__copy = np.array(voxel, copy=True)
    vertices_, triangles_ = mcubes.marching_cubes(voxel__copy, 0)

    export_file_name_dae = (
        result_dir
        + str(file_name)
        + "-"
        + "epoch"
        + "="
        + str(current_epoch)
        + "-"
        + "global_step"
        + "="
        + str(current_global_step)
        + ".dae"
    )
    mcubes.export_mesh(vertices_, triangles_, export_file_name_dae, str(file_name))


def marche_the_cube(
    tensor_cube: np.ndarray,
    epoch,
    global_step: int,
    marching_cube_result_dir: str,
    title: str,
    selected_mesh_index: int,
):
    # marching cube
    collected_voxel_decoded_name = title + str(selected_mesh_index) + "-"
    make_mcubes_from_voxels(
        tensor_cube,
        epoch,
        global_step,
        0,
        collected_voxel_decoded_name,
        marching_cube_result_dir,
    )
