import torch
import numpy as np
from src.utils.plot_voxel import plot_v


def generate_plot_for_given_dict_of_items(
    dict_of_items: dict,
    resolution: torch.int32,
    number_of_slices: torch.int32,
    plot_scale_factor: torch.int32,
    plot_range: list[float],
) -> list:
    number_of_plots_to_generate = len(dict_of_items.items())
    keys = [key for key in dict_of_items.keys()]
    plots = []
    for i in range(len(keys)):
        current_key = keys[i]
        current_value = dict_of_items.get(current_key)
        # call plot for the current key:
        (
            image_list,
            title_list,
        ) = plot_v(
            current_value,
            number_of_slices,
            resolution // plot_scale_factor,
            current_key,
            plot_range,
        )
        plots.append(image_list)
    assert len(plots) == number_of_plots_to_generate
    return plots


def generate_plot_for_everything(
    number_of_slices: torch.int32,
    plot_scale_factor: torch.int32,
    resolution: torch.int32,
    collected_32cubes_array: np.ndarray,
    collected_sub_voxels_decoded_non_optimized_array: np.ndarray,
    collected_sub_voxels_decoded_optimized_array: np.ndarray,
    collected_decoded_masked_optimized_latent_codes_array: np.ndarray,
    transformer_output_sequence_up_collected_32cubes_collected_32cubes_array: np.ndarray,
    diff_transformer_output_and_optimized_latent_code_array: np.ndarray,
):
    # plot true-gt ------------------------------
    (
        gt_image_list,
        gt_title_list,
    ) = plot_v(
        collected_32cubes_array,
        number_of_slices,
        resolution // plot_scale_factor,
        "true_gt_sdf",
    )

    # plot non-optimized-latent-codes------------------------------
    (
        non_optimized_latent_codes_image_list,
        non_optimized_latent_codes_title_list,
    ) = plot_v(
        collected_sub_voxels_decoded_non_optimized_array,
        number_of_slices,
        resolution // plot_scale_factor,
        "non_optimized_latent_codes",
    )
    # plot optimized-latent-codes------------------------------
    (
        optimized_latent_codes_image_list,
        optimized_latent_codes_title_list,
    ) = plot_v(
        collected_sub_voxels_decoded_optimized_array,
        number_of_slices,
        resolution // plot_scale_factor,
        "optimized_latent_codes",
    )
    # plot masked-optimized-latent-codes------------------------------
    (
        masked_latent_codes_image_list,
        masked_latent_codes_title_list,
    ) = plot_v(
        collected_decoded_masked_optimized_latent_codes_array,
        number_of_slices,
        resolution // plot_scale_factor,
        "Masked-optimized_latent_codes",
    )
    # plot transformer-output------------------------------
    (
        transformer_output_image_list,
        transformer_output_title_list,
    ) = plot_v(
        transformer_output_sequence_up_collected_32cubes_collected_32cubes_array,
        number_of_slices,
        resolution // plot_scale_factor,
        "Transformer_output",
    )
    # plot diff------------------------------
    (
        diff_transformer_output_vs_optimized_latent_code_image_list,
        diff_transformer_output_vs_optimized_latent_code_title_list,
    ) = plot_v(
        diff_transformer_output_and_optimized_latent_code_array,
        number_of_slices,
        resolution // plot_scale_factor,
        "Diff_transformer_output_vs_optimized_latent_code",
    )
    return (
        gt_image_list,
        non_optimized_latent_codes_image_list,
        optimized_latent_codes_image_list,
        masked_latent_codes_image_list,
        transformer_output_image_list,
        diff_transformer_output_vs_optimized_latent_code_image_list,
    )


def plot_for_given_dict_of_items(slice_index: torch.int32, list_of_items: list):
    image_numpy_list = []
    for i in range(len(list_of_items)):
        item_image_list = list_of_items[i]
        item_image_list_slice = item_image_list[slice_index]
        image_numpy = torch.as_tensor(item_image_list_slice[:, :, :3]).permute(
            [2, 0, 1]
        )
        image_numpy_list.append(image_numpy)

    plot_to_be_drawn = torch.cat([x.cuda() for x in image_numpy_list], -1)
    return plot_to_be_drawn


def plot_everything(
    slice_index: torch.int32,
    gt_image_list: list,
    non_optimized_latent_codes_image_list: list,
    masked_latent_codes_image_list: list,
    transformer_output_image_list: list,
    diff_transformer_output_vs_optimized_latent_code_image_list: list,
):
    gt_image_list_slice = gt_image_list[slice_index]
    image_gt = torch.as_tensor(gt_image_list_slice[:, :, :3]).permute([2, 0, 1])
    del gt_image_list_slice

    non_optimized_latent_codes_slice = non_optimized_latent_codes_image_list[
        slice_index
    ]
    image_non_optim = torch.as_tensor(
        non_optimized_latent_codes_slice[:, :, :3]
    ).permute([2, 0, 1])
    del non_optimized_latent_codes_slice

    masked_latent_codes_slice = masked_latent_codes_image_list[slice_index]
    image_masked = torch.as_tensor(masked_latent_codes_slice[:, :, :3]).permute(
        [2, 0, 1]
    )
    del masked_latent_codes_slice

    transformer_output_image_slice = transformer_output_image_list[slice_index]
    image_transformer = torch.as_tensor(
        transformer_output_image_slice[:, :, :3]
    ).permute([2, 0, 1])
    del transformer_output_image_slice

    diff_transformer_output_vs_optimized_latent_code_image_slice = (
        diff_transformer_output_vs_optimized_latent_code_image_list[slice_index]
    )
    image_diff = torch.as_tensor(
        diff_transformer_output_vs_optimized_latent_code_image_slice[:, :, :3]
    ).permute([2, 0, 1])
    del diff_transformer_output_vs_optimized_latent_code_image_slice

    plot = torch.cat(
        [
            image_gt.cuda(),
            image_non_optim.cuda(),
            image_masked.cuda(),
            image_transformer.cuda(),
            image_diff.cuda(),
        ],
        -1,
    )
    del image_gt
    del image_non_optim
    del image_masked
    del image_transformer
    del image_diff

    return plot
