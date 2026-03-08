import torch
from typing import Tuple
from src.utils.subvolume_devision import (
    subvdivide_voxel_with_batch,
    extract_empty_sub_voxel_indices_from_voxel,
extract_outside_sub_voxel_indices_from_voxel
)

def sub_divide_gt_and_normalize(gt_sdf_full_voxel: torch.Tensor, number_of_sub_voxels: int, target_resolution: int) -> torch.Tensor:
    batch_size = gt_sdf_full_voxel.shape[0]
    sub_voxels = subvdivide_voxel_with_batch(gt_sdf_full_voxel.clone(), target_resolution).to(device=gt_sdf_full_voxel.device)
    assert sub_voxels.shape == (batch_size, number_of_sub_voxels, target_resolution, target_resolution, target_resolution)
    # normalization--------------------------------------------------------------------------------------------------------------
    sub_voxels = sub_voxels * 2.0  # 2*value_range
    return sub_voxels.clone()


def extract_empty_and_non_empty_voxels(sub_voxels: torch.Tensor, number_of_sub_voxels: int, target_resolution: int) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = sub_voxels.shape[0]
    # # Empty sub-voxels -----------------------------------------------------------------------------------------------------
    assert sub_voxels.shape == (batch_size, number_of_sub_voxels, target_resolution, target_resolution, target_resolution)
    empty_sub_voxels_bool = extract_empty_sub_voxel_indices_from_voxel(sub_voxels).to(sub_voxels.device)
    non_empty_sub_voxels_bool = torch.logical_not(empty_sub_voxels_bool)
    num_empty = torch.count_nonzero(empty_sub_voxels_bool, dim=1)
    # if not torch.all(torch.any(non_empty_sub_voxels_bool, dim=1)):
    #     print("all voxels are empty")
    #     breakpoint()
    return (empty_sub_voxels_bool, non_empty_sub_voxels_bool)


def extract_outside_and_non_outside_voxels(sub_voxels: torch.Tensor, number_of_sub_voxels: int, target_resolution: int) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = sub_voxels.shape[0]
    # # Empty sub-voxels -----------------------------------------------------------------------------------------------------
    assert sub_voxels.shape == (batch_size, number_of_sub_voxels, target_resolution, target_resolution, target_resolution)
    empty_sub_voxels_bool = extract_outside_sub_voxel_indices_from_voxel(sub_voxels).to(sub_voxels.device)
    non_empty_sub_voxels_bool = torch.logical_not(empty_sub_voxels_bool)
    num_empty = torch.count_nonzero(empty_sub_voxels_bool, dim=1)
    # if not torch.all(torch.any(non_empty_sub_voxels_bool, dim=1)):
    #     print("all voxels are empty")
    #     breakpoint()
    return (empty_sub_voxels_bool, non_empty_sub_voxels_bool)