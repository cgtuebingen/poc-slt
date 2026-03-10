import torch
from typing import Any, Union


def weight_losses(
    loss_dict: dict, weight_non_empty: torch.float32 = 1.0, weight_empty: torch.float32 = 1.0, weight_masked: torch.float32 = 1.0, weight_non_masked: torch.float32 = 1.0
) -> Union[tuple[torch.float32, torch.float32, torch.float32, torch.float32, torch.float32]]:
    l1_loss_empty = loss_dict["l1_loss_empty"]
    l1_loss_masked = loss_dict["l1_loss_masked"]
    l1_loss_non_masked = loss_dict["l1_loss_non_masked"]
    l1_loss_non_empty = loss_dict["l1_loss_non_empty"]

    l1_loss_non_empty_w = l1_loss_non_empty * weight_non_empty
    l1_loss_empty_w = l1_loss_empty * weight_empty
    l1_loss_masked_w = l1_loss_masked * weight_masked
    l1_loss_non_masked_w = l1_loss_non_masked * weight_non_masked

    l1_loss_w = (l1_loss_empty_w) + (l1_loss_masked_w) + (l1_loss_non_masked_w) + (l1_loss_non_empty_w)

    return (l1_loss_w, l1_loss_non_empty_w, l1_loss_empty_w, l1_loss_masked_w, l1_loss_non_masked_w)
    # return (l1_loss_w, l1_loss_empty_w, l1_loss_masked_w, l1_loss_non_masked_w)

def weight_losses_noEmptymasking(
    loss_dict: dict, weight_masked: torch.float32 = 1.0, weight_non_masked: torch.float32 = 1.0
) -> Union[tuple[torch.float32, torch.float32, torch.float32]]:
    l1_loss_masked = loss_dict["l1_loss_masked"]
    l1_loss_non_masked = loss_dict["l1_loss_non_masked"]

    l1_loss_masked_w = l1_loss_masked * weight_masked
    l1_loss_non_masked_w = l1_loss_non_masked * weight_non_masked

    l1_loss_w = (l1_loss_masked_w) + (l1_loss_non_masked_w)

    return (l1_loss_w, l1_loss_masked_w, l1_loss_non_masked_w)
    # return (l1_loss_w, l1_loss_empty_w, l1_loss_masked_w, l1_loss_non_masked_w)

def scale_losses_noEmptymasking(loss_dict: dict, masked_bool: torch.bool, non_masked_bool: torch.bool, transformer_output_sequence_shape: list) -> tuple[Any, Any, Any]:

    assert (len(transformer_output_sequence_shape) == 2)
    l1_loss_masked = loss_dict["l1_loss_masked"]
    l1_loss_non_masked = loss_dict["l1_loss_non_masked"]

    # scale based on number of voxels (to match "all_losses")

    l1_loss_masked_scaled = l1_loss_masked * torch.count_nonzero(masked_bool.clone()) / (transformer_output_sequence_shape[0] * transformer_output_sequence_shape[1])
    l1_loss_non_masked_scaled = l1_loss_non_masked * torch.count_nonzero(non_masked_bool.clone()) / (transformer_output_sequence_shape[0] * transformer_output_sequence_shape[1])

    l1_loss_scaled = (l1_loss_masked_scaled) + (l1_loss_non_masked_scaled)

    return (l1_loss_scaled, l1_loss_masked_scaled, l1_loss_non_masked_scaled)


def scale_losses(
    loss_dict: dict, non_empty_sub_voxels_bool: torch.bool,  empty_sub_voxels_bool: torch.bool, masked_bool: torch.bool, non_masked_bool: torch.bool, transformer_output_sequence_shape: list
) -> tuple[Any, Any, Any, Any, Any]:

    assert (len(transformer_output_sequence_shape) == 2)
    l1_loss_empty = loss_dict["l1_loss_empty"]
    l1_loss_masked = loss_dict["l1_loss_masked"]
    l1_loss_non_masked = loss_dict["l1_loss_non_masked"]
    l1_loss_non_empty = loss_dict["l1_loss_non_empty"]

    # scale based on number of voxels (to match "all_losses")
    l1_loss_non_empty_scaled = l1_loss_non_empty * torch.count_nonzero(non_empty_sub_voxels_bool.clone()) / (transformer_output_sequence_shape[0] * transformer_output_sequence_shape[1])

    l1_loss_empty_scaled = l1_loss_empty * torch.count_nonzero(empty_sub_voxels_bool.clone()) / (transformer_output_sequence_shape[0] * transformer_output_sequence_shape[1])
    l1_loss_masked_scaled = l1_loss_masked * torch.count_nonzero(masked_bool.clone()) / (transformer_output_sequence_shape[0] * transformer_output_sequence_shape[1])
    l1_loss_non_masked_scaled = l1_loss_non_masked * torch.count_nonzero(non_masked_bool.clone()) / (transformer_output_sequence_shape[0] * transformer_output_sequence_shape[1])

    l1_loss_scaled = (l1_loss_masked_scaled) + (l1_loss_empty_scaled) + (l1_loss_non_masked_scaled) # + (l1_loss_non_empty_scaled)

    return (l1_loss_scaled, l1_loss_non_empty_scaled, l1_loss_empty_scaled, l1_loss_masked_scaled, l1_loss_non_masked_scaled)
    # return (l1_loss_scaled, l1_loss_empty_scaled, l1_loss_masked_scaled, l1_loss_non_masked_scaled)


def scale_chosen_loss(chosen_loss: torch.float32, corresponding_bool, transformer_output_sequence_shape: list) -> torch.float32:
    loss_scaled = chosen_loss * torch.count_nonzero(corresponding_bool.clone()) / (transformer_output_sequence_shape[0] * transformer_output_sequence_shape[1])
    return loss_scaled

def weight_chose_loss(chosen_loss: torch.float32, corresponding_weight: torch.float32) -> torch.float32:
    weighted_chosen_loss = chosen_loss * corresponding_weight
    return weighted_chosen_loss

