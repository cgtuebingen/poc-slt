#!/usr/bin/env python3
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import trimesh
sys.path.append("../Mesh_Preparation")
from  Mesh_Preparation.mesh_preparation_functions import as_mesh
import GTSDF_python
import os
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image
import torch
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()

def visualize_sdf_voxel(resolution, gt_sdf):

    min_sdf, max_sdf = np.min(gt_sdf), np.max(gt_sdf)
    value_limit = min(np.abs(min_sdf), np.abs(max_sdf))
    gt_sdf_r = gt_sdf[:resolution*resolution]
    gt_sdf_r= gt_sdf_r.reshape([resolution, resolution])
    fig = Figure(figsize=(resolution / 40, resolution / 50))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    # ax.set_title(str(rotation_quaternion))
    ax.set_axis_off()
    # plot = ax.imshow(sdf_values_for_vis___[o_id], cmap="seismic", interpolation="none")
    plot = ax.imshow(gt_sdf_r, cmap="seismic", interpolation="none")
    # ax.imshow(sdf_values_for_vis___[o_id], interpolation="none")
    plot.set_clim(-value_limit, value_limit)
    fig.colorbar(plot, ax=ax)
    fig.tight_layout()
    canvas.draw()
    image = np.array(canvas.renderer.buffer_rgba())
    image = image[:, :, :3]
    im = Image.fromarray(image)
    im.save("/home/zakeri/Documents/Codes/MyCodes/AutoEncoder_SDF_and_Z_Learning/results//img.png")
    fig.savefig('/home/zakeri/Documents/Codes/MyCodes/AutoEncoder_SDF_and_Z_Learning/results//fig.png')
    # print("image:", image.shape)
    return image
def visualize_sdf(resolution, voxel, label, root_path):
    fig = plt.figure(figsize=[10, 10])
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # the original voxel grid with coordinate value range [-1, 1]
    ax.scatter(
        voxel[:, 0],
        voxel[:, 1],
        voxel[:, 2],
        marker="^",
        label=label,
    )

    ax.legend()
    # root_path = "/home/zakeri/Documents/Codes/MyCodes/AutoEncoder_SDF_and_Z_Learning/results/"
    name = label + '.png'
    root_path_name = os.path.join(root_path, name)
    # root_path_name = str(root_path + label + '.png')
    fig.savefig(root_path_name)

def make_voxel_from(resolution: int, value_range: int=1) -> np.array:
    # create voxel grid
    xs = np.linspace(-value_range, value_range, resolution, dtype=np.float32)
    ys = xs.copy()
    zs = xs.copy()
    xs, ys, zs = np.meshgrid(xs, ys, zs)
    voxel_grid = np.stack([xs, ys, zs], -1)
    return voxel_grid
def transform_voxel(voxel_grid, resolution):

    voxel_grid = voxel_grid.reshape([-1, 3])
    noise_factor = 0.3  # the higher the more extreme augmentations you'll get

    # create random augmentation
    transformation = noise_factor * np.random.randn(3, 3)  # standard normal distribution np.eye(3) +
    # Compute the required minimum scale of the grid under the transformation
    # For that we transform all 8 corners and then check the coordinates
    xs = np.array([-1.0, 1.0])
    ys = xs.copy()
    zs = xs.copy()
    xs, ys, zs = np.meshgrid(xs, ys, zs)
    corners = np.stack([xs, ys, zs], -1)
    corners = corners.reshape([-1, 3])

    # apply transformation
    corners_transformed = np.einsum("ij,nj->ni", transformation, corners)

    # what is the maximum that any coordinate component has changed
    corner_differences = np.abs(corners_transformed)
    max_difference = np.max(corner_differences)
    # We get the necessary scaling factor by making sure that a point that started with
    # a coordinate component with value 1 ends up at 1 + this maximum difference.
    minimum_scale_factor = 1 / max_difference
    # print("\n Minimum scale factor", minimum_scale_factor)

    # We can of course still make the scale larger - thus the object smaller
    # for example like this. This would make the objects smaller, which could
    # cause problems with resolution at some point.
    scale_up_factor = 1  # (1 + np.random.rand())
    # print("\n additional scale up factor:", scale_up_factor)
    scale_factor = minimum_scale_factor * scale_up_factor

    # The transformation is linear, so it does not matter if we scale before or
    # after computing the transformation.
    voxel_grid_transformed = np.einsum("ij,nj->ni", transformation, voxel_grid)
    voxel_grid_transformed *= scale_factor  # apply scaling
    # print("\n voxel_grid_transformed range:", np.max(voxel_grid_transformed), "--", np.min(voxel_grid_transformed))
    # print("\n voxel_grid range:", np.max(voxel_grid), "--", np.min(voxel_grid))
    return voxel_grid_transformed
def generate_sdf_from_voxel(mesh, voxel, resolution):

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    GTSDF_fn = GTSDF_python.GTSDF_python(faces, vertices)
    # sdf original----------------------------------------------------------------------------------------------------------------------------
    gt_sdf_voxel = np.array(GTSDF_fn.signed_distance_v(voxel))
    gt_sdf_voxel = gt_sdf_voxel.reshape([resolution, resolution, resolution])  # ground truth for the input

    return gt_sdf_voxel
def voxel_slice_plot(gt_sdf_voxel, number_of_slices, resolution, title):
    min_sdf, max_sdf = np.min(gt_sdf_voxel), np.max(gt_sdf_voxel)
    # print("\n min_sdf: ", min_sdf, " , max_sdf: ", max_sdf)
    value_limit = min(np.abs(min_sdf), np.abs(max_sdf))
    xy_range = 1
    # if (value_limit < xy_range / 10):
    #     value_limit = max(np.abs(min_sdf), np.abs(max_sdf))
    image_list = []
    title_list = []
    for dim in range(3):
        for i in range(number_of_slices):
            fig = Figure(figsize=(resolution / 30, resolution / 40))
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)

            if dim == 0:
                # print("\n visualization:", gt_sdf_voxel.shape, ", type:", type(gt_sdf_voxel))
                gt_sdf_voxel_slice = gt_sdf_voxel[int(i), :, :]
                title_dim = title + '_' + 'dim' + str(dim) + '_slice' + str(i)
            elif dim == 1:
                # print("\n visualization:", gt_sdf_voxel.shape, ", type:", type(gt_sdf_voxel))
                gt_sdf_voxel_slice = gt_sdf_voxel[:, int(i), :]
                title_dim = title + '_' + 'dim' + str(dim) + '_slice' + str(i)
            else:
                # print("\n visualization:", gt_sdf_voxel.shape, ", type:", type(gt_sdf_voxel))
                gt_sdf_voxel_slice = gt_sdf_voxel[:, :, int(i)]
                title_dim = title + '_' + 'dim' + str(dim) + '_slice' + str(i)

            ax.set_title(title_dim, fontsize=8)
            ax.set_axis_off()
            plot = ax.imshow(gt_sdf_voxel_slice, origin="lower", cmap="seismic", interpolation="none")
            plot.set_clim(-value_limit, value_limit)
            fig.colorbar(plot, ax=ax)
            fig.tight_layout()
            canvas.draw()
            image = np.array(canvas.renderer.buffer_rgba())
            image_list.append(image)
            title_list.append(title_dim)
    return image_list, title_list

# @typechecked
def voxel_plot(gt_sdf_voxel: np.array, number_of_slices: int, resolution: int, title: str, plot_range: float):
    min_sdf = np.min(gt_sdf_voxel)
    max_sdf = np.max(gt_sdf_voxel)
    # print("\n min_sdf: ", min_sdf, " , max_sdf: ", max_sdf)
    value_limit = max(np.abs(min_sdf), np.abs(max_sdf))
    # print("\n value_limit: ", -value_limit, " , value_limit: ", value_limit)

    # if (value_limit < xy_range / 10):
    #     value_limit = max(np.abs(min_sdf), np.abs(max_sdf))
    images = []
    titles = []
    for i in range(0, number_of_slices, 1):
        # res = 128 // 2 = 64
        # dpi = 100
        # figsize_x = 64/10 * dpi = 640
        # figsize_y = 64/20 * dpi = 320
        #fig = Figure(figsize=(resolution/10, resolution/20))

        fig = Figure(figsize=(4.8, 3.2))  # 480x320 px
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        #center = int(resolution/2)
        center = int(gt_sdf_voxel.shape[0] / 2)
        #if (number_of_slices + center < resolution):
        if (number_of_slices + center < gt_sdf_voxel.shape[0]):
            gt_sdf_voxel_slice = gt_sdf_voxel[int(i + center), :, :]  # i + center
            title = title + '_' + '_slice' + str(i)

            ax.set_title(title, fontsize=8)
            ax.set_axis_off()
            plot = ax.imshow(gt_sdf_voxel_slice, origin="lower", cmap="seismic", interpolation="none")
            plot.set_clim(-plot_range, plot_range)   # TODO: this is default; -value_limit, value_limit, for the transformer, we set it to -2, 2
            fig.colorbar(plot, ax=ax)
            fig.tight_layout()
            canvas.draw()
            image = np.array(canvas.renderer.buffer_rgba())
            images.append(image)
            titles.append(title)
    return images, titles

def voxel_plot_valueset(gt_sdf_voxel: np.array, number_of_slices: int, resolution: int, title: str):
    min_sdf = np.min(gt_sdf_voxel)
    max_sdf = np.max(gt_sdf_voxel)
    # print("\n min_sdf: ", min_sdf, " , max_sdf: ", max_sdf)
    value_limit = max(np.abs(min_sdf), np.abs(max_sdf))
    # print("\n value_limit: ", -value_limit, " , value_limit: ", value_limit)

    # if (value_limit < xy_range / 10):
    #     value_limit = max(np.abs(min_sdf), np.abs(max_sdf))
    images = []
    titles = []
    for i in range(0, number_of_slices, 1):
        fig = Figure(figsize=(resolution/2, resolution/4))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        center = int(resolution/2)
        if (number_of_slices + center < resolution):
            gt_sdf_voxel_slice = gt_sdf_voxel[int(i + center), :, :]  # i + center
            title = title + '_' + '_slice' + str(i)

            ax.set_title(title, fontsize=8)
            ax.set_axis_off()
            plot = ax.imshow(gt_sdf_voxel_slice, origin="lower", cmap="seismic", interpolation="none")
            plot.set_clim(-value_limit, value_limit)   # TODO: this is default; -value_limit, value_limit, for the transformer, we set it to -2, 2
            fig.colorbar(plot, ax=ax)
            fig.tight_layout()
            canvas.draw()
            image = np.array(canvas.renderer.buffer_rgba())
            images.append(image)
            titles.append(title)
    return images, titles

def write_voxel_slice_plot(image_list: list, title_list: list,  result_dir: str):

        for j in range(len(image_list)):
            image = image_list[j]
            file_name = title_list[j]

            im = Image.fromarray(image[:, :, :3])
            # file_name = title + '_' + 'dim' + str(dim) + '_slice' + str(i) + '.png'
            # file_name_ = file_name + str(i) + '.png'
            file_name_ = file_name + '.png'
            file_name_path = os.path.join(result_dir, file_name_)
            im.save(file_name_path)
def create_scatter_plot(points: torch.Tensor, title: str = "L1 loss vs distance per sub-voxel", ylable: str = "l1 loss", xlabel: str = "distance"):

    plt.rcParams.update({'font.size': 8})
    # points = np.concatenate(x_data, y_data)
    fig = plt.figure(figsize=(128/30, 128/40))
    #canvas = FigureCanvas(fig)
    #ax = fig.add_subplot(111)
    # draw a diagonal line (representing linear relation)
    # total_min, total_max = torch.min(points), torch.max(points)
    #
    # ax.set_xlim(total_min.cpu(), total_max.cpu())
    # ax.plot([0, 10], [0, 1], color="black", linestyle="--")

    plot = plt.scatter(points[:, 1].cpu(), points[:, 0].cpu(), alpha=0.1)  # point[:, 0]==loss, points[:,1]==distances
    plt.title(title)
    plt.ylabel(ylable)
    plt.xlabel(xlabel)

    # ax.legend()
    # fig.colorbar(plot, ax=ax)
    fig.tight_layout()
    #canvas.draw()
    #image = np.array(canvas.renderer.buffer_rgba())
    #image = image[:, :, :3]
    return fig
def create_scatter_plot_for_sub_voxels(points: torch.Tensor, title: str = "L1 loss vs distance per sub-voxel", ylable: str = "l1 loss", xlabel: str = "distance"):

    plt.rcParams.update({'font.size': 8})
    fig = plt.figure(figsize=(128/10, 128/20))
    ax = fig.add_subplot(111)
    y_data = points[0, :].cpu()

    total_min_y = min(y_data).cpu()
    total_max_y = max(y_data).cpu()
    ax.set_ylim(total_min_y-0.05, total_max_y+0.05)

    x_data = points[1, :].cpu()

    total_min_x = min(x_data).cpu()
    total_max_x = max(x_data).cpu()
    ax.set_xlim(total_min_x-0.5, total_max_x+0.5)

    plot = plt.scatter(x_data, y_data, alpha=0.4)   # point[0, :]==loss, points[1, :]==sub_voxel indices
    xticks = (np.arange(total_max_x))
    plt.xticks(xticks)
    plt.title(title)
    plt.ylabel(ylable)
    plt.xlabel(xlabel)
    # ax.legend()
    # fig.colorbar(plot, ax=ax)
    #fig.tight_layout() #TODO uncomment me.

    return fig
# if __name__ == "__main__":
#
#     mesh_file_name = "/graphics/scratch/datasets/ShapeNetCorev2_remeshed/ShapeNetCore.v2/04460130/8cf718ed6a5fefd8fcaad7f5ff5ee65c/models/model_normalized.obj.obj"
#
#     scene_or_mesh = trimesh.load(mesh_file_name)
#     mesh = as_mesh(scene_or_mesh)
#
#     resolution = 128
#     # create a voxel grid
#     voxel_grid = make_voxel_from(resolution)
#     # transform/augument the voxel grid
#     voxel_grid_transformed = transform_voxel(voxel_grid, resolution)
#     # generate gt sdf for transformed voxel grid
#     voxel_grid_ = voxel_grid.reshape([-1, 3])
#     gt_sdf_voxel = generate_sdf_from_voxel(mesh, voxel_grid_, resolution)
#
#     number_of_slices = 3
#     # result_dir = "/home/zakeri/Documents/Codes/MyCodes/AutoEncoder_SDF_and_Z_Learning/results/"
#     result_dir = "/home/zakeri/Documents/Codes/MyCodes/AutoEncoder_SDF_and_Z_Learning_Rewrite/results/"
#     my_title = "gt_sdf_voxel_transformed"
#     visualize_sdf(resolution, gt_sdf_voxel, my_title, result_dir)
#     # make a  plot from slices of gt sdf for transformed voxel grid
#     image_list, title_list = voxel_slice_plot(gt_sdf_voxel, number_of_slices, resolution, my_title)
#
#     # write the plot as an image into directory
#     write_voxel_slice_plot(image_list, title_list, result_dir)

#     # mesh = trimesh.load("/graphics/scratch/datasets/ShapeNetCorev2_remeshed/ShapeNetCore.v2/02773838/4a175db3a8cad99b5dd5ceb34f6e33d9/models/model_normalized.obj.obj")
#     # mesh = trimesh.load("/graphics/scratch/datasets/ShapeNetCorev2_remeshed/ShapeNetCore.v2/04460130/8cf718ed6a5fefd8fcaad7f5ff5ee65c/models/model_normalized.obj.obj")
#     # mesh = trimesh.load("/graphics/scratch/datasets/ShapeNetCorev2_remeshed/ShapeNetCore.v2/04460130/dd36bd331fce23aa5a4e821f5ddcc98f/models/model_normalized.obj.obj")
#     # mesh = trimesh.load("/graphics/scratch/datasets/ShapeNetCorev2_remeshed/ShapeNetCore.v2/03513137/8eb4152bc0f6b91e7bc12ebbbb3689a1/models/model_normalized.obj.obj")