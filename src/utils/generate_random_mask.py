import torch
from typing import Any, Union

from torch import Tensor

from src.utils.masker3d import make_mask3d_from_voxel


def generate_random_mask(
    non_empty_sub_voxels_bool: torch.bool,
    masking_ratio: torch.float32,
    number_of_sub_voxels: torch.int32,
    batch_size: torch.int,
) -> Union[tuple[Any, Any]]:
    # initialize it ---------------------------------------------------------------------------------------------------------
    non_masked_bool = torch.zeros(
        [batch_size, number_of_sub_voxels], dtype=torch.bool, device="cuda"
    )
    masked_bool = torch.zeros(
        [batch_size, number_of_sub_voxels], dtype=torch.bool, device="cuda"
    )
    # # Decoder Mask First, Non-Mask Second, Masked Third-------------------------------------------------------------------------
    non_masking_ratio = 1 - masking_ratio
    for _ in range(10):
        remaining_voxels = non_empty_sub_voxels_bool.clone()
        num_remaining = torch.count_nonzero(remaining_voxels, dim=1)

        assert torch.all(
            num_remaining > 0
        )  # we already kicked out all the empty meshes so this will not happen anymore

        non_masked_bool = make_mask3d_from_voxel(
            batch_size, number_of_sub_voxels, prob=non_masking_ratio
        ).to("cuda")
        non_masked_bool = torch.logical_and(non_masked_bool, remaining_voxels)
        num_non_masked = torch.count_nonzero(non_masked_bool, dim=1)

        remaining_voxels = torch.logical_and(
            remaining_voxels, torch.logical_not(non_masked_bool)
        )
        masked_bool = remaining_voxels

        non_masking_ratio_min = non_masking_ratio - 0.05
        minimum_num_non_masked = (num_remaining * (non_masking_ratio_min)).to(
            dtype=torch.int32
        )
        if torch.any(num_non_masked < minimum_num_non_masked):
            continue
        break
    # else:
    #     # breakpoint
    #     assert False  # could not find a valid solution

    # assert torch.all(num_non_masked > 0)
    return (non_masked_bool, masked_bool)


def generate_random_mask_for_all(
    number_of_sub_voxels: int, batch_size: int, masking_ratio: float
) -> tuple[Any, Tensor]:
    mask_all_bool = make_mask3d_from_voxel(
        batch_size, number_of_sub_voxels, prob=masking_ratio
    ).to(device="cuda")
    num_mask_all = torch.count_nonzero(mask_all_bool, dim=1)

    return (mask_all_bool, num_mask_all)
