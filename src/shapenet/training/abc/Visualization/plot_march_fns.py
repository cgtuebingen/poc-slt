import numpy as np
from Augmentation.Augmentation_Visualization import voxel_plot_valueset
from Visualization import m_cube_fns
import torch
from typing import Tuple, Any

def generate_data_for_plotting(dict_arguments_for_vis, dict_arguments_of_variables, batch_idx, number_of_slices) -> tuple[tuple[np.ndarray, np.ndarray], tuple[Any, Any, Any]]:
    object_indicis = dict_arguments_for_vis["object_indicis"]
    sdf_predicted = dict_arguments_for_vis["sdf_predicted"]
    normalized_gt_sdf_voxels_reshaped = dict_arguments_for_vis["normalized_gt_sdf_voxels_reshaped"]

    target_resolution = dict_arguments_of_variables["target_resolution"]

    gt_sdf_voxel = normalized_gt_sdf_voxels_reshaped[batch_idx, 0, :, :, :].squeeze(0)
    gt_sdf_voxel_ = np.asarray(gt_sdf_voxel.cpu()).astype(np.float32)
    gt_sdf_voxel_image_list, gt_sdf_voxel_title_list = voxel_plot_valueset(gt_sdf_voxel_, number_of_slices, target_resolution, "gt_sdf_voxel_transformed")

    # ----
    sdf_predicted_ = sdf_predicted[batch_idx, 0, :, :, :].squeeze(0)
    sdf_predicted_ = np.asarray(sdf_predicted_.cpu()).astype(np.float32)
    sdf_predicted_image_list, sdf_predicted_title_list = voxel_plot_valueset(sdf_predicted_, number_of_slices, target_resolution, "sdf_voxel_reconstructed_from_depth")
    # ----
    diff = abs(np.subtract(sdf_predicted_, gt_sdf_voxel_))
    diff_ = np.asarray(diff).astype(float)
    diff_image_list, diff_title_list = voxel_plot_valueset(diff_, number_of_slices, target_resolution, "diff")

    return ((gt_sdf_voxel_, sdf_predicted_), (gt_sdf_voxel_image_list, sdf_predicted_image_list, diff_image_list))


def generate_plot_for_this_slice(data_to_be_plotted: dict, slice_number: int) -> torch.Tensor:
    gt_sdf_voxel_image_list = data_to_be_plotted["gt_sdf_voxel_image_list"]
    sdf_predicted_image_list = data_to_be_plotted["sdf_predicted_image_list"]
    diff_image_list = data_to_be_plotted["diff_image_list"]

    gt_sdf_voxel_slice = gt_sdf_voxel_image_list[slice_number]
    predicted_image_slice = sdf_predicted_image_list[slice_number]
    diff_image_slice = diff_image_list[slice_number]

    image_re = torch.as_tensor(predicted_image_slice[:, :, :3]).permute([2, 0, 1])
    del predicted_image_slice

    image_gt = torch.as_tensor(gt_sdf_voxel_slice[:, :, :3]).permute([2, 0, 1])
    del gt_sdf_voxel_slice

    image_diff = torch.as_tensor(diff_image_slice[:, :, :3]).permute([2, 0, 1])
    del diff_image_slice

    plot = torch.cat([image_re.cuda(), image_gt.cuda(), image_diff.cuda()], -1)

    del image_re
    del image_gt
    del image_diff

    return plot


def march_cube_give_dict(data_to_be_marched: dict, selected_index: int, batch_idx, current_epoch: int, global_step, marching_cube_result_dir: str) -> None:
    sdf_predicted_ = data_to_be_marched["sdf_predicted"]
    gt_sdf_voxel_ = data_to_be_marched["gt_sdf_voxel"]

    gt_name = "-sdf_gt-obj-ID-" + str(selected_index) + "-"
    pred_name = "-sdf_predicted-obj-ID-" + str(selected_index) + "-"
    m_cube_fns.make_mcubes_from_voxels(sdf_predicted_, current_epoch, global_step, batch_idx, pred_name, marching_cube_result_dir)
    m_cube_fns.make_mcubes_from_voxels(gt_sdf_voxel_, current_epoch, global_step, batch_idx, gt_name, marching_cube_result_dir)