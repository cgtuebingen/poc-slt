import sys
sys.path.append("/home/zakeri/Documents/Codes/MyCodes/Proposal2/SDF_VAE/")
import trimesh
import numpy as np
import GTSDF_python
from typing import Tuple, Any
import mcubes
from Visualization.m_cube_fns import make_mcubes_from_voxels_obj_for_pad, make_mcubes_from_voxels_obj_for_eval
import os
from tqdm import tqdm

def make_voxel_points(resolution: int, value_range: int) -> Tuple[np.ndarray, np.ndarray]:
    x_ = np.linspace(-value_range, value_range, resolution, False, dtype=np.float64)
    y_ = np.linspace(-value_range, value_range, resolution, False, dtype=np.float64)
    z_ = np.linspace(-value_range, value_range, resolution, False, dtype=np.float64)

    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    # shift by half a voxel (i.e., 2*resolution)
    voxel = np.stack((x, y, z), axis=3) + (value_range - -value_range) / (2 * resolution)
    points = voxel.reshape([-1, 3])
    return (points, voxel)

def calculate_gt_sdf_from_mesh_my_way(mesh_data: Tuple[np.ndarray, np.ndarray], resolution: int, value_range: int) -> np.ndarray:
    faces, vertices = mesh_data
    points, voxels = make_voxel_points(resolution, value_range)

    GTSDF_fn = GTSDF_python.GTSDF_python(faces, vertices)
    gt_sdf = np.array(GTSDF_fn.signed_distance_v(points))
    # print("\n gt_sdf :", gt_sdf.shape)

    gt_sdf_voxel = gt_sdf.reshape([resolution, resolution, resolution])  # ground truth for the input

    return gt_sdf_voxel

def normalize_mesh(mesh_data: Tuple[np.ndarray, np.ndarray, np.ndarray]):
    # keep the max component of the extracted mesh
    faces, vertices, bbx = mesh_data
    # normalize mesh
    bbmin = bbx.min(0)
    bbmax = bbx.max(0)
    center = (bbmin + bbmax) * 0.5

    scale = 2.0 * 1 / (bbmax - bbmin).max()
    vertices = (vertices - center) * scale

    normalized_mesh_data = (scale, center)
    mesh_data = (faces, vertices)
    mesh_stuff = (mesh_data, normalized_mesh_data)
    return mesh_stuff

def extarct_obj_name_from_ABC(obj_file_name: str):
    sub_name = obj_file_name.replace(".obj")
    return sub_name

def generate_sdf_from_ABC(obj_dir: str, res_dir: str, resolution: int = 128):
    obj_list = []
    watertight_list = []
    with open("/graphics/scratch/datasets/ABC/abc_v00_files", 'r') as file:
        for line in file:
            obj_list.append(line.rstrip('\n'))

    progress = tqdm(range(len(obj_list)), desc="Obj Files")
    for f in progress:
        obj_file_current = obj_list[f]

        if (obj_file_current.endswith(".obj")):
            obj_file_current_path = os.path.join(obj_dir, obj_file_current)
            mesh_current = trimesh.load_mesh(obj_file_current_path, force='mesh')
            if(isinstance(mesh_current, list)):
                if (len(mesh_current) == 0):
                    print("\trimesh retuned empty list")
                    continue
            if (mesh_current.is_watertight):
                watertight_list.append(obj_file_current)
                # mesh_data_current = (np.asarray(mesh_current.faces), np.asarray(mesh_current.vertices), mesh_current.bounds)
                # normalized_Mesh = normalize_mesh(mesh_data_current, resolution=resolution)
                # gt_sdf_voxel = calculate_gt_sdf_from_mesh_my_way((normalized_Mesh.faces, normalized_Mesh.vertices), resolution=128, value_range=1)
                # gt_sdf_voxel_mesh_name = "/gt_sdf_"
                # object_index = extarct_obj_name_from_ABC(obj_file_current)
                # make_mcubes_from_voxels_obj_for_eval(gt_sdf_voxel, object_index, gt_sdf_voxel_mesh_name + "_mcube", res_dir)
                # # open LMDB and put it into lmdb
                # example = {"obj_file_name": obj_file_current, "gt_sdf_voxel": gt_sdf_voxel}

        progress.set_postfix({"watertight_num": len(watertight_list)})
        if (f %1000 == 0):
            print("\n f: ", f, "watertight_num:", len(watertight_list))
            out_name = os.path.join("/graphics/scratch/datasets/ABC/", "watertight_obj_file_names.txt")
            with open(out_name, "w") as out:
                # Iterating over each element of the list
                for line in watertight_list:
                    out.write(line)  # Adding the line to the text.txt
                    out.write('\n')  # Adding a new line character


def setup():
    res_dir = "/graphics/scratch2/staff/zakeri/ABC_dataset_processing"
    mesh1_path = "/graphics/scratch/datasets/ABC/obj/00360043/00360043_5840853c00e03110ce81b9e4_trimesh_001.obj"
    mesh2_path = "/graphics/scratch/datasets/ABC/obj/00097321/00097321_43cf1d4eb5d9b8eb86f0581f_trimesh_000.obj"
    mesh3_path = "/graphics/scratch/datasets/ABC/obj/00482067/00482067_540f94ed50edf67a8cabba32_trimesh_006.obj"
    mesh4_path = "/graphics/scratch/datasets/ABC/obj/00173520/00173520_575b4f40e4b03d4addc6dbfb_trimesh_000.obj"
    mesh5_path = "/graphics/scratch/datasets/ABC/obj/00104072/00104072_8af298acdd9664e54b748411_trimesh_000.obj"
    mesh6_path = "/graphics/scratch/datasets/ABC/obj/00211059/00211059_459bcc30ab511aeb290e0eaf_trimesh_001.obj"

    mesh1 = trimesh.load_mesh(mesh6_path, force='mesh')
    is_watertight = mesh1.is_watertight
    print("\nis_watertight", is_watertight)
    vertices = np.asarray(mesh1.vertices)
    faces = np.asarray(mesh1.faces)
    bbx = np.asarray(mesh1.bounds)
    # print("\n bbx: ", bbx, ", min: ", bbx.min, ", max: ", bbx.max)

    mesh_stuff = normalize_mesh((faces, vertices, bbx), resolution=128)
    (mesh_data, normalized_mesh_data) = mesh_stuff
    faces, vertices = mesh_data
    points, voxel = make_voxel_points(resolution=128, value_range=1)
    gt_sdf_voxel = calculate_gt_sdf_from_mesh_my_way((faces, vertices), resolution=128, value_range=1)
    gt_sdf_voxel_mesh_name = "/sdf2Mesh_"
    object_index = "mesh6_path"
    make_mcubes_from_voxels_obj_for_pad(gt_sdf_voxel, object_index, gt_sdf_voxel_mesh_name + "_mcube", res_dir)
    print()

# if __name__ == "__main__":
#     # obj_dir = "/graphics/scratch/datasets/ABC/obj/"
#     # res_dir = "/graphics/scratch2/staff/zakeri/ABC_dataset_processing"
#     # generate_sdf_from_ABC(obj_dir, res_dir, resolution=128)
#     # setup()
#     # print()

