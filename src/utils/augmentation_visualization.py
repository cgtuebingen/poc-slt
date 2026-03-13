#!/usr/bin/env python3
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import GTSDF_python
import os
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image
import torch
from torchtyping import TensorType, patch_typeguard

patch_typeguard()


def make_voxel_from(resolution: int, value_range: int=1) -> np.array:
    # create voxel grid
    xs = np.linspace(-value_range, value_range, resolution, dtype=np.float32)
    ys = xs.copy()
    zs = xs.copy()
    xs, ys, zs = np.meshgrid(xs, ys, zs)
    voxel_grid = np.stack([xs, ys, zs], -1)
    return voxel_grid

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
                gt_sdf_voxel_slice = gt_sdf_voxel[int(i), :, :]
                title_dim = title + '_' + 'dim' + str(dim) + '_slice' + str(i)
            elif dim == 1:
                gt_sdf_voxel_slice = gt_sdf_voxel[:, int(i), :]
                title_dim = title + '_' + 'dim' + str(dim) + '_slice' + str(i)
            else:
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
            plot.set_clim(-plot_range, plot_range)   # this is default; -value_limit, value_limit, for the transformer, we set it to -2, 2
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
            plot.set_clim(-value_limit, value_limit)   # this is default; -value_limit, value_limit, for the transformer, we set it to -2, 2
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
        file_name_ = file_name + '.png'
        file_name_path = os.path.join(result_dir, file_name_)
        im.save(file_name_path)
