import sys
sys.path.append("../")
import os
# disable CUDA
import numpy as np
import trimesh
from typing import List, Iterable
import msgpack_numpy as m
m.patch()
from tqdm.auto import tqdm
import typing
import open3d as o3d
import  random

def mapShapeNetCorev2(label):

    synsteIds_categories_mapping = {
        'airplane': 0, 'trash can': 1, 'bag': 2,
        'basket': 3, 'bathtub': 4, 'bed': 5,
        'bench': 6, 'birdhouse': 7, 'bookshelf': 8,
        'bottle': 9, 'bowl': 10, 'bus': 11,
        'cabinet': 12, 'camera': 13, 'can': 14,
        'cap': 15, 'car': 16, 'cell phone': 17,
        'chair': 18, 'clock': 19, 'keyboard': 20,
        'dishwasher': 21, 'display': 22, 'headphone': 23,
        'faucet': 24, 'file': 25, 'guitar': 26,
        'helmet': 27, 'jar': 28, 'knife': 29,
        'lamp': 30, 'laptop': 31, 'speaker': 32,
        'mailbox': 33, 'microphone': 34, 'microwave': 35,
        'bike': 36, 'mug': 37, 'piano': 38,
        'pillow': 39, 'pistol': 40, 'pot': 41,
        'printer': 42, 'remote': 43, 'file': 44,
        'rocket': 45, 'skateboard': 46, 'sofa': 47,
        'stove': 48, 'table': 49, 'telephone': 50,
        'tower': 51, 'training': 52, 'watercraft': 53,
        'washing machine': 54}

    class_id = synsteIds_categories_mapping.get(label)
    return class_id
def as_mesh(scene_or_mesh: typing.Union[trimesh.Scene, trimesh.Trimesh]) -> typing.Optional[trimesh.Trimesh]:
    if isinstance(scene_or_mesh, trimesh.Scene):
        if (len(scene_or_mesh.geometry) == 0 or scene_or_mesh ==[] ):
            return None  # empty scene ==> empty mesh
        # mesh = trimesh.util.concatenate([trimesh.Trimesh(vertices=m.vertices, faces=m.faces) for m in scene_or_mesh.geometry.values()])
        return trimesh.util.concatenate(scene_or_mesh.geometry.values())
    elif isinstance(scene_or_mesh, trimesh.Trimesh):
        return scene_or_mesh
    # else:
    #     raise "unexpected input given to as_mesh: " #+ str(type(scene_or_mesh))
def PrepareShapeNetCorev2(mesh_path: str):
    synsteIds_categories = {
        '02691156': 'airplane', '02747177': 'trash can', '02773838': 'bag',
        '02801938': 'basket', '02808440': 'bathtub', '02818832': 'bed',
        '02828884': 'bench', '02843684': 'birdhouse', '02871439': 'bookshelf',
        '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
        '02933112': 'cabinet', '02942699': 'camera', '02946921': 'can',
        '02954340': 'cap', '02958343': 'car', '02992529': 'cell phone',
        '03001627': 'chair', '03046257': 'clock', '03085013': 'keyboard',
        '03207941': 'dishwasher', '03211117': 'display', '03261776': 'headphone',
        '03325088': 'faucet', '03337140': 'file', '03467517': 'guitar',
        '03513137': 'helmet', '03593526': 'jar', '03624134': 'knife',
        '03636649': 'lamp', '03642806': 'laptop', '03691459': 'speaker',
        '03710193': 'mailbox', '03759954': 'microphone', '03761084': 'microwave',
        '03790512': 'bike', '03797390': 'mug', '03928116': 'piano',
        '03938244': 'pillow', '03948459': 'pistol', '03991062': 'pot',
        '04004475': 'printer', '04074963': 'remote', '04090263': 'file',
        '04099429': 'rocket', '04225987': 'skateboard', '04256520': 'sofa',
        '04330267': 'stove', '04379243': 'table', '04401088': 'telephone',
        '04460130': 'tower', '04468005': 'training', '04530566': 'watercraft',
        '04554684': 'washing machine'}
    dataset_train = []
    dataset_val = []

    data = dict()
    data_train = dict()
    data_val = dict()

    train_split = 0.80
    val_split = 0.20
    dataset_index = 0
    train_data_index = 0
    val_data_index = 0
    # read all the dataset and separate training, val, and test and write their indices in a file with synsetId, label and mesh_file_path
    for folder in tqdm(os.listdir(mesh_path), desc="Processing Folders"):
        # print("\n folder:", folder, ", type:", type(folder))
        folder_path = os.path.join(mesh_path, folder)
        sub_folder_data = dict()
        category_train = []
        category_val = []
        category_index_data = []
        sub_folder_index = 0
        for sub_folder in tqdm(os.listdir(folder_path), desc="Processing Sub-Folders"):
            sub_folder_path = os.path.join(folder_path, sub_folder)
            file_name = os.path.join(sub_folder_path, "models/model_normalized.obj.obj")
            if file_name.endswith('.obj'):
                # print("\n name of the mesh:", file_name)
                label = synsteIds_categories.get(str(folder))
                if label == None:
                    print("\n The synsetId/folder does not exist")
                current_sub_folder_data = {dataset_index: [folder, sub_folder_index, label]}
                data[dataset_index] = [folder, sub_folder_path,  file_name, sub_folder_index, label]
                sub_folder_data[sub_folder_index] = dataset_index  # adding new element to the dict
                category_index_data.append(sub_folder_index)
                dataset_index += 1
                sub_folder_index += 1
        # divide training, val and test indices per each folder
        # folder_train_len = int((sub_folder_index * train_split))
        # folder_val_len = int((sub_folder_index * val_split))
        random.shuffle(category_index_data)
        category_train = category_index_data[:int((sub_folder_index + 1) * train_split)]  # Remaining 80% to training set
        category_val = category_index_data[int((sub_folder_index + 1) * train_split):]  # Splits 20% data to validation set

        # train_data_index = 0
        for i in range(len(category_train)):
            sub_folder_index_i = category_train[i]
            dataset_index_i = sub_folder_data.get(sub_folder_index_i)
            dataset_train.append(dataset_index_i)
            folder_t, sub_folder_path_t, file_name_t, sub_folder_index_t, label_t = data.get(dataset_index_i)
            train_example_list = [dataset_index_i, folder_t, sub_folder_path_t, file_name_t, sub_folder_index_t, label_t]
            data_train[train_data_index] = train_example_list
            train_data_index += 1
        for j in range(len(category_val)):
            sub_folder_index_j = category_val[j]
            dataset_index_j = sub_folder_data.get(sub_folder_index_j)
            dataset_val.append(dataset_index_j)
            folder_v, sub_folder_path_v, file_name_v, sub_folder_index_v, label_v = data.get(dataset_index_j)
            val_example_list = [dataset_index_j, folder_v, sub_folder_path_v, file_name_v, sub_folder_index_v, label_v]
            data_val[val_data_index] = val_example_list
            val_data_index += 1
    return data_train, data_val

def point_cloud_from_mesh(mesh:trimesh.Trimesh, points_to_sample: int) -> np.array:

    points, face_index = trimesh.sample.sample_surface(mesh, points_to_sample)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    point_cloud = np.asarray(pcd.points).astype(np.float32)

    return point_cloud

def generate_random_distance_and_offsets_for_this_point(point: np.array) -> List:
    x, y, z = point
    # ---pick a d as a random value between [0, 1]----------------------------------------------------------------------------------------
    # from uniform the possibility of having small d values and large values are equal
    d = np.random.uniform(0.1, 1, 1)  # FIXME previous lower value was 0.00001
    # print("\n d value is:", d)
    # ---Generate a 3 random values between [0, d] as x_offset, y_offset, z_offset---------------------------------------------------------
    # with normal distribution we are more likely to do not go crazy far away from the point, yet we do not want the point necessarily be the center of the voxel,
    # yet we want the center of the voxel be close to the point since there we have more surface
    x_offset = np.random.normal(0, d/3.0, 1) # since we have the standard deviation of the normal is to 1. but it goes from -3 to 3 we divide d/3
    # print("\n d x_offset is:", x_offset)

    y_offset = np.random.normal(0, d/3.0, 1)
    # print("\n y_offset is:", y_offset)

    z_offset = np.random.normal(0, d/3.0, 1)
    # print("\n z_offset is:", z_offset)

    # ---make corners of the spanning bounding box around this point such tha this point is not necessarily in the center------------------
    left = x - d/2.0 - x_offset
    top = y - d/2.0 - y_offset
    front = z - d/2.0 - z_offset
    return [d, left, x_offset, top, y_offset, front, z_offset]

def span_voxel_with_distance_and_corners(distance_and_corners_list: List, target_resolution: int) -> List:
    assert (len(distance_and_corners_list) == 7)
    d, left, x_offset, top, y_offset, front, z_offset = distance_and_corners_list

    # --- make voxels already from the corners with target resolution many points----------------------------------------------------------
    x_s = np.linspace(left, left + d, target_resolution, endpoint=False, dtype=np.float32)
    y_s = np.linspace(top, top + d, target_resolution, endpoint=False, dtype=np.float32)
    z_s = np.linspace(front, front + d, target_resolution, endpoint=False, dtype=np.float32)

    x_m, y_m, z_m = np.meshgrid(x_s, y_s, z_s, indexing='ij')
    # --- stack the x_s, y_s, z_s to generate a voxel shape
    # d / (2 * self.target_resolution) is the offset to go from corners to the center of grids
    voxel = np.stack((x_m, y_m, z_m), axis=3) + d / (2 * target_resolution)
    return [voxel, d, left, x_offset, top, y_offset, front, z_offset]

def extract_point_positions_from_distances_and_corners(distance_and_corners_list: List, target_resolution: int) -> np.ndarray:
    assert (len(distance_and_corners_list) == 7)
    d, left, x_offset, top, y_offset, front, z_offset = distance_and_corners_list
    # --- make voxels already from the corners with target resolution many points----------------------------------------------------------
    x_s = np.linspace(left, left + d, target_resolution, endpoint=False, dtype=np.float32)
    y_s = np.linspace(top, top + d, target_resolution, endpoint=False, dtype=np.float32)
    z_s = np.linspace(front, front + d, target_resolution, endpoint=False, dtype=np.float32)

    x_m, y_m, z_m = np.meshgrid(x_s, y_s, z_s, indexing='ij')
    # --- stack the x_s, y_s, z_s to generate a voxel shape
    # d / (2 * self.target_resolution) is the offset to go from corners to the center of grids
    voxel = np.stack((x_m, y_m, z_m), axis=3) + d / (2 * target_resolution)
    points = voxel.reshape([-1, 3])
    return points

def span_voxel_for_this_point(point: np.array, target_resolution: int):
    distance_and_corners_list = generate_random_distance_and_offsets_for_this_point(point)
    current_point_data = span_voxel_with_distance_and_corners(distance_and_corners_list, target_resolution)
    return current_point_data

def mapp_chunk_number_to_index_range_f(data: dict, chunk_number: int, desired_index_range: int = 1000) -> List:

    number_of_chunks = len(data) // desired_index_range
    left_out_chunk = len(data) % desired_index_range

    if ((chunk_number < number_of_chunks) and (chunk_number != -1)):
        index_start = chunk_number * desired_index_range
        index_end = index_start + desired_index_range
        processing_index_list = list(range(index_start, index_end))
        assert len(processing_index_list) == desired_index_range
    elif (chunk_number == -1):
        index_start = number_of_chunks * desired_index_range
        index_end = index_start + left_out_chunk
        assert (index_end) == len(data)

        processing_index_list = list(range(index_start, index_end))
        assert len(processing_index_list) == left_out_chunk
    else:
        raise "the chunk number is invalid relative to desired_index_range!"

    return processing_index_list
def mapp_chunk_number_to_index_range_l(data_size: int, chunk_number: int, desired_index_range: int = 1000) -> Iterable:

    number_of_chunks = (data_size + desired_index_range-1) // desired_index_range

    if ((chunk_number < number_of_chunks)):
        index_start = chunk_number * desired_index_range
        index_end = min(index_start + desired_index_range, data_size)
        processing_index_list = range(index_start, index_end)
    else:
        print("\n chunk_number: ", chunk_number, "desired_index_range:", desired_index_range, "data_size:", data_size, "number_of_chunks: ", number_of_chunks)
        raise RuntimeError("the chunk number is invalid relative to desired_index_range!")

    return processing_index_list


# file_name ="/backup_graphics/backup_scratch3-2022_10_26/datasets/ShapeNetCorev2_remeshed_0.008/ShapeNetCore.v2/03928116/9578f81207bacd5ddd7dc0810367760d/models/model_normalized.obj.obj"
# #file_name="/home/zakeri/Documents/Datasets/3DData/simple 3D obj/armadillo.obj"
# # file_name="/home/zakeri/Documents/Datasets/3DData/simple 3D obj/bunny.obj"
# import GTSDF_python
# import sys
# sys.path.append("../")
# from LMDB_Distributed_Generator import helper_functions as hf
#
# scene_or_mesh = trimesh.load(file_name, force='mesh')
# # print("\n mesh file name:", file_name)
# mesh = hf.as_mesh(scene_or_mesh)
# if (mesh is not None):
#     # print("\n mesh vertices:", mesh.vertices.shape, ", mesh faces:", mesh.faces.shape)
#
#     vertices = np.asarray(mesh.vertices)
#     faces = np.asarray(mesh.faces)
#     GTSDF_fn = GTSDF_python.GTSDF_python(faces, vertices)
#     point_cloud = hf.point_cloud_from_mesh(mesh, 64)
#     for p in range(20):
#         point = point_cloud[p, :]
#         voxel_data = hf.span_voxel_around_point(point, 32)
#         voxel_tmp = voxel_data[0]
#         d = voxel_data[1]
#         # print("\n d:", d)
#         voxel_to_points = voxel_tmp.reshape([-1, 3])
#         point = voxel_to_points[p, :]
#         point = point.reshape([1, 3])
#         # print("\n point:", point)
#         # ---GT SDF---------------------------------------------------------------------------------------------------------------------------
#         gt_sdf = np.array(GTSDF_fn.signed_distance_v(voxel_to_points), dtype=np.float32)
#         print("\n min sdf :", np.min(voxel_tmp), "max sdf: ", np.max(voxel_tmp))
#     #
#     print("\n-----------------------------------------------------------------------------")
#     # # generate coordinates:
#     value_range =1
#     resolution =128
#     x_ = np.linspace(-value_range, value_range, resolution, False, dtype=np.float64)
#     y_ = np.linspace(-value_range, value_range, resolution, False, dtype=np.float64)
#     z_ = np.linspace(-value_range, value_range, resolution, False, dtype=np.float64)
#
#     x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
#     voxels = np.stack((x, y, z), axis=3) + (value_range - -value_range) / (2 * resolution)
#     points = voxels.reshape([-1, 3])
#     print("\n points[0, :]:", points[0, :])
#     p0 = np.array(points[0, :], copy=True, dtype=np.float32)
#     p0 = p0.reshape([1, 3])
#     gt_sdf = np.array(GTSDF_fn.signed_distance_v(p0), dtype=np.float32)
#     print("\n min sdf[0]:", np.min(gt_sdf), "max sdf[0]: ", np.max(gt_sdf))
#
#     print("\n points[1, :]:", points[1, :])
#     p1 = np.array(points[1, :], copy=True, dtype=np.float32)
#     p1 = p0.reshape([1, 3])
#     gt_sdf = np.array(GTSDF_fn.signed_distance_v(p1), dtype=np.float32)
#     print("\n min sdf[1] :", np.min(gt_sdf), "max sdf[1]: ", np.max(gt_sdf))
#
#     print("\n points[2, :]:", points[2, :])
#     p2 = np.array(points[2, :], copy=True, dtype=np.float32)
#     p2 = p0.reshape([1, 3])
#     gt_sdf = np.array(GTSDF_fn.signed_distance_v(p2), dtype=np.float32)
#     print("\n min sdf[2] :", np.min(gt_sdf), "max sdf[2]: ", np.max(gt_sdf))
#
#
#
