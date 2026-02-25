import sys
sys.path.append("../")
import numpy as np
import mcubes
import trimesh
from Augmentation import Augmentation_Visualization as av
from Mesh_Preparation import mesh_preparation_functions as mp
from mayavi import mlab
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
print("torch.cuda.device_count()", torch.cuda.device_count())
from vae_main_oldLMDB import SDFtoSDF
import argparse
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# mesh_file_name = "/graphics/scratch2/datasets/ShapeNetCorev2_remeshed_0.008/ShapeNetCore.v2/02933112/1ad4c572e0fd6a576e1e9a13188ab4bb/models/model_normalized.obj.obj"  # obj1
# mesh_file_name = "/graphics/scratch/datasets/ShapeNetCorev2/ShapeNetCore.v2/02933112/1ad4c572e0fd6a576e1e9a13188ab4bb/models/model_normalized.obj"  # obj1 original

# mesh_file_name = "/home/zakeri/Documents/Datasets/3DData/simple 3D obj/multi_difficult_obj/armadillo.obj"  # obj2
# mesh_file_name = "/graphics/scratch/datasets/ShapeNetCorev2_remeshed/ShapeNetCore.v2/02876657/3a6fae97b5fa5cbb3db8babb13da5441/models/model_normalized.obj.obj"  # obj3

# mesh_file_name = "/graphics/scratch/datasets/ShapeNetCorev2_remeshed/ShapeNetCore.v2/02808440/1acfdd86f8d48bb19750530d0552b23/models/model_normalized.obj.obj"  # obj4
# mesh_file_name = "/graphics/scratch/datasets/ShapeNetCorev2/ShapeNetCore.v2/02808440/1acfdd86f8d48bb19750530d0552b23/models/model_normalized.obj" # obj4_original
mesh_file_name = "/home/zakeri/Documents/Datasets/3DData/simple 3D obj/multi_difficult_obj/DoubleBed.obj"
scene_or_mesh = trimesh.load(mesh_file_name)
mesh = mp.as_mesh(scene_or_mesh)

resolution = 32
# create a voxel grid
voxel_grid = av.make_voxel_from(resolution)
# transform/augment the voxel grid
# voxel_grid_transformed = transform_voxel(voxel_grid, resolution)
# generate gt sdf for transformed voxel grid
gt_sdf_voxel = av.generate_sdf_from_voxel(mesh, voxel_grid.reshape([-1, 3]), resolution)
vertices, triangles = mcubes.marching_cubes(gt_sdf_voxel, 0)
print("\n vertices: ", vertices.shape)
print("\n triangles: ", triangles.shape)
# Export the result to sphere.dae
mcubes.export_mesh(vertices, triangles, "gt_sdf_voxel.dae", "gt_sdf_voxel")
try:
    print("Plotting mesh...")
    from mayavi import mlab
    mlab.triangular_mesh(
        vertices[:, 0], vertices[:, 1], vertices[:, 2],
        triangles)
    print("Done.")
    mlab.show()
except ImportError:
    print("Could not import mayavi. Interactive demo not available.")
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--latent_dim', default=512, type=int)  # 512
parser.add_argument('--resolution', default=128, type=int)
parser.add_argument('--target_resolution', default=32, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--train_val_dict_path', default="/graphics/scratch2/datasets/ShapeNetCorev2_remeshed_0.008/train_val_dictionaries/", type=str)
# parser.add_argument('--lmdb_path', default="/scratch_shared/zakeri/", type=str)
parser.add_argument('--lmdb_path', default="/graphics/scratch2/datasets/ShapeNetCoreV2_remeshed_0.008_GTSDF_Voxel128_LMDB/",
                    type=str)  # FIXME!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!To Originl LMDB!!!!!!!!!!!!!!!!!!!!!!
parser.add_argument('--mesh_path', default="/graphics/scratch2/datasets/ShapeNetCorev2_remeshed_0.008/ShapeNetCore.v2/", type=str)
parser.add_argument('--length', default=10000, type=int)
parser.add_argument('--points_to_sample', default=1024, type=int)
parser.add_argument('--query_number', default=1000, type=int)
parser.add_argument('--value_range', default=1, type=int)
parser.add_argument('--examples_per_epoch', default=1000, type=int)

args = parser.parse_args()

model = SDFtoSDF(
    latent_dim=args.latent_dim,
    resolution=args.resolution,
    target_resolution=args.target_resolution,
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    train_val_dict_path=args.train_val_dict_path,
    lmdb_path=args.lmdb_path,
    mesh_path=args.mesh_path,
    length=args.length,

    points_to_sample=args.points_to_sample,
    query_number=args.query_number,
    value_range=args.value_range,
    examples_per_epoch=args.examples_per_epoch,

)
checkpoint = torch.load("/graphics/scratch2/train_logs/zakeri/VAE/integration_test/lightning_logs/version_29/checkpoints/used_for_vis1/checkpoint-epoch=1730-loss=0.000.ckpt", map_location=torch.device('cpu'))  # exp5

model.load_state_dict(checkpoint["state_dict"])
model.eval()
gt_sdf_voxel_copy = np.array(gt_sdf_voxel, copy=True)
print("\n gt_sdf_voxel: ", gt_sdf_voxel.shape)

gt_sdf_voxel_copy = torch.from_numpy(gt_sdf_voxel_copy)
gt_sdf_voxel_copy = gt_sdf_voxel_copy.unsqueeze(0)
print("\n gt_sdf_voxel_copy: ", gt_sdf_voxel_copy.shape)
gt_sdf_voxel_copy = gt_sdf_voxel_copy.unsqueeze(0)
print("\n gt_sdf_voxel_copy: ", gt_sdf_voxel_copy.shape)
with torch.no_grad():
    reconstructed_gt_sdf_voxel = model.forward(gt_sdf_voxel_copy.to(torch.float32))
# y = self.forward(gt_sdf_voxels_transformed_reshaped.to(torch.float32))
reconstructed_gt_sdf_voxel_ = reconstructed_gt_sdf_voxel.detach().numpy()
print("\n reconstructed_gt_sdf_voxel_: ", reconstructed_gt_sdf_voxel_.shape)
reconstructed_gt_sdf_voxel_ = reconstructed_gt_sdf_voxel_.squeeze(0)
print("\n reconstructed_gt_sdf_voxel_: ", reconstructed_gt_sdf_voxel_.shape)
reconstructed_gt_sdf_voxel_ = reconstructed_gt_sdf_voxel_.squeeze(0)
print("\n reconstructed_gt_sdf_voxel_: ", reconstructed_gt_sdf_voxel_.shape)

reconstructed_gt_sdf_voxel_copy = np.array(reconstructed_gt_sdf_voxel_, copy=True)
vertices_, triangles_ = mcubes.marching_cubes(reconstructed_gt_sdf_voxel_copy, 0)
print("\n vertices: ", vertices_.shape)
print("\n triangles_: ", triangles_.shape)
# Export the result to sphere.dae
mcubes.export_mesh(vertices_, triangles_, "reconstructed_gt_sdf_voxel.dae", "reconstructed_gt_sdf_voxel")

try:
    print("Plotting mesh...")
    from mayavi import mlab
    mlab.triangular_mesh(
        vertices_[:, 0], vertices_[:, 1], vertices_[:, 2],
        triangles_)
    print("Done.")
    mlab.show()
except ImportError:
    print("Could not import mayavi. Interactive demo not available.")


