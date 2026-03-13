import torch
from typing import Any

from src.utils.subvolume_devision import (
    collect_sub_voxels_to_voxel_with_batch,
)

def generate_data_for_plotting(dict_arguments_for_vis: dict, dict_arguments_of_variables: dict, fdecoder) -> dict:

    optimized_latent_codes = dict_arguments_for_vis["optimized_latent_codes"]
    non_optimized_latent_codes = dict_arguments_for_vis["non_optimized_latent_codes"]
    masked_optimized_latent_codes = dict_arguments_for_vis["masked_optimized_latent_codes"]
    transformer_output_sequence_up_reshaped = dict_arguments_for_vis["transformer_output_sequence_up"]
    sub_voxels = dict_arguments_for_vis["sub_voxels"]

    number_of_sub_voxels = dict_arguments_of_variables["number_of_sub_voxels"]
    latent_dim = dict_arguments_of_variables["latent_dim"]
    target_resolution = dict_arguments_of_variables["target_resolution"]
    resolution = dict_arguments_of_variables["resolution"]

    batch_size = sub_voxels.shape[0]
    # decoder--------------------------
    with torch.no_grad():
        decoded_optimized_latent_codes = fdecoder(optimized_latent_codes.clone()).to(device="cuda")  # [128, 512, 2, 2, 2] -> [128, 1, 32, 32, 32]
        decoded_optimized_latent_codes_reshaped = decoded_optimized_latent_codes.reshape([batch_size, number_of_sub_voxels, target_resolution, target_resolution, target_resolution])
        collected_sub_voxels_decoded_optimized = collect_sub_voxels_to_voxel_with_batch(decoded_optimized_latent_codes_reshaped, resolution)
        assert collected_sub_voxels_decoded_optimized.shape == (batch_size, resolution, resolution, resolution)
        del decoded_optimized_latent_codes
        del decoded_optimized_latent_codes_reshaped
        # non_optimized_latent_code------------------------------------------------------------------------------------------------------------
        # decoder-------------------
        decoded_non_optimized_latent_codes = fdecoder(non_optimized_latent_codes).to(device="cuda")
        decoded_non_optimized_latent_codes_reshaped = decoded_non_optimized_latent_codes.reshape([batch_size, number_of_sub_voxels, target_resolution, target_resolution, target_resolution])
        collected_decoded_non_optimized_latent_codes_reshaped = collect_sub_voxels_to_voxel_with_batch(decoded_non_optimized_latent_codes_reshaped, resolution)
        del decoded_non_optimized_latent_codes_reshaped
        assert collected_decoded_non_optimized_latent_codes_reshaped.shape == (batch_size, resolution, resolution, resolution)
        # Masked latent code collect------------------------------------------------------------------------------------------------------------
        # decoder-------------------
        decoded_masked_optimized_latent_codes = fdecoder(masked_optimized_latent_codes).to(device="cuda")
        decoded_masked_optimized_latent_codes_reshaped = decoded_masked_optimized_latent_codes.reshape([batch_size, number_of_sub_voxels, target_resolution, target_resolution, target_resolution])
        collected_decoded_masked_optimized_latent_codes = collect_sub_voxels_to_voxel_with_batch(decoded_masked_optimized_latent_codes_reshaped, resolution)
        del decoded_masked_optimized_latent_codes_reshaped
        assert collected_decoded_masked_optimized_latent_codes.shape == (batch_size, resolution, resolution, resolution)
        # Transformer output collect------------------------------------------------------------------------------------------------------------
        # transformer_output_sequence_up_reshaped = (
        #     transformer_output_sequence_up.reshape(
        #         [
        #             batch_size,
        #             number_of_sub_voxels,
        #             latent_dim,
        #             2,
        #             2,
        #             2,
        #         ]
        #     )
        #     .to(device="cuda")
        #     .to(dtype=torch.float32)
        # )
        decoded_transformer_output_sequence = fdecoder(transformer_output_sequence_up_reshaped).to(device="cuda")
        decoded_transformer_output_sequence_reshaped = decoded_transformer_output_sequence.reshape(
            batch_size,
            number_of_sub_voxels,
            target_resolution,
            target_resolution,
            target_resolution,
        ).to(device="cuda")
        del decoded_transformer_output_sequence
        transformer_output_sequence_up_collected_32cubes = collect_sub_voxels_to_voxel_with_batch(decoded_transformer_output_sequence_reshaped, resolution)
        # Diff--------------------------------------------------------------------------------
        diff_transformer_outputs_and_optimized_latent_codes = torch.subtract(transformer_output_sequence_up_collected_32cubes, collected_sub_voxels_decoded_optimized)
        # True GT collect------------------------------------------------------------------------------------------------------------
        collected_sub_voxels = collect_sub_voxels_to_voxel_with_batch(sub_voxels, resolution)

        data_dict_for_vis = {
            "collected_sub_voxels": collected_sub_voxels,
            "collected_decoded_non_optimized_latent_codes": collected_decoded_non_optimized_latent_codes_reshaped,
            "collected_sub_voxels_decoded_optimized": collected_sub_voxels_decoded_optimized,
            "collected_decoded_masked_optimized_latent_codes": collected_decoded_masked_optimized_latent_codes,
            "transformer_output_sequence_up_collected_32cubes": transformer_output_sequence_up_collected_32cubes,
            "diff_transformer_outputs_and_optimized_latent_codes": diff_transformer_outputs_and_optimized_latent_codes,
        }
        return data_dict_for_vis

def generate_data_for_plottingv2(dict_arguments_for_vis: dict, dict_arguments_of_variables: dict, fdecoder) -> dict:
    non_optimized_latent_codes = dict_arguments_for_vis["non_optimized_latent_codes"]
    masked_non_optimized_latent_codes = dict_arguments_for_vis["masked_non_optimized_latent_codes"]
    transformer_output_sequence_up_reshaped = dict_arguments_for_vis["transformer_output_sequence_up"]
    sub_voxels = dict_arguments_for_vis["sub_voxels"]

    number_of_sub_voxels = dict_arguments_of_variables["number_of_sub_voxels"]
    latent_dim = dict_arguments_of_variables["latent_dim"]
    target_resolution = dict_arguments_of_variables["target_resolution"]
    resolution = dict_arguments_of_variables["resolution"]

    batch_size = sub_voxels.shape[0]
    # decoder--------------------------
    with torch.no_grad():
        decoded_non_optimized_latent_codes = fdecoder(non_optimized_latent_codes.clone()).to(device="cuda")  # [128, 512, 2, 2, 2] -> [128, 1, 32, 32, 32]
        decoded_non_optimized_latent_codes_reshaped = decoded_non_optimized_latent_codes.reshape([batch_size, number_of_sub_voxels, target_resolution, target_resolution, target_resolution])
        collected_sub_voxels_decoded_non_optimized = collect_sub_voxels_to_voxel_with_batch(decoded_non_optimized_latent_codes_reshaped, resolution)
        assert collected_sub_voxels_decoded_non_optimized.shape == (batch_size, resolution, resolution, resolution)
        del decoded_non_optimized_latent_codes
        del decoded_non_optimized_latent_codes_reshaped
        # Masked latent code collect------------------------------------------------------------------------------------------------------------
        # decoder-------------------
        decoded_masked_non_optimized_latent_codes = fdecoder(masked_non_optimized_latent_codes).to(device="cuda")
        decoded_masked_non_optimized_latent_codes_reshaped = decoded_masked_non_optimized_latent_codes.reshape([batch_size, number_of_sub_voxels, target_resolution, target_resolution, target_resolution])
        collected_decoded_masked_non_optimized_latent_codes = collect_sub_voxels_to_voxel_with_batch(decoded_masked_non_optimized_latent_codes_reshaped, resolution)
        del decoded_masked_non_optimized_latent_codes_reshaped
        assert collected_decoded_masked_non_optimized_latent_codes.shape == (batch_size, resolution, resolution, resolution)
        # Transformer output collect------------------------------------------------------------------------------------------------------------
        # transformer_output_sequence_up_reshaped = (
        #     transformer_output_sequence_up.reshape(
        #         [
        #             batch_size,
        #             number_of_sub_voxels,
        #             latent_dim,
        #             2,
        #             2,
        #             2,
        #         ]
        #     )
        #     .to(device="cuda")
        #     .to(dtype=torch.float32)
        # )
        decoded_transformer_output_sequence = fdecoder(transformer_output_sequence_up_reshaped).to(device="cuda")
        decoded_transformer_output_sequence_reshaped = decoded_transformer_output_sequence.reshape(
            batch_size,
            number_of_sub_voxels,
            target_resolution,
            target_resolution,
            target_resolution,
        ).to(device="cuda")
        del decoded_transformer_output_sequence
        transformer_output_sequence_up_collected_32cubes = collect_sub_voxels_to_voxel_with_batch(decoded_transformer_output_sequence_reshaped, resolution)
        # Diff--------------------------------------------------------------------------------
        diff_transformer_outputs_and_non_optimized_latent_codes = torch.subtract(transformer_output_sequence_up_collected_32cubes, collected_sub_voxels_decoded_non_optimized)
        # True GT collect------------------------------------------------------------------------------------------------------------
        collected_sub_voxels = collect_sub_voxels_to_voxel_with_batch(sub_voxels, resolution)

        data_dict_for_vis = {
            "collected_sub_voxels": collected_sub_voxels,
            "collected_sub_voxels_decoded_non_optimized": collected_sub_voxels_decoded_non_optimized,
            "collected_decoded_masked_non_optimized_latent_codes": collected_decoded_masked_non_optimized_latent_codes,
            "transformer_output_sequence_up_collected_32cubes": transformer_output_sequence_up_collected_32cubes,
            "diff_transformer_outputs_and_non_optimized_latent_codes": diff_transformer_outputs_and_non_optimized_latent_codes,
        }
        return data_dict_for_vis

def collect_generated_data_for_plottingv3(data_dict_for_vis: dict, resolution: int, batch_idx: int) -> dict[str, Any]:
    collected_sub_voxels = data_dict_for_vis["collected_sub_voxels"]
    collected_sub_voxels_decoded_non_optimized = data_dict_for_vis["collected_sub_voxels_decoded_non_optimized"]
    collected_decoded_masked_non_optimized_latent_codes = data_dict_for_vis["collected_decoded_masked_non_optimized_latent_codes"]
    transformer_output_sequence_up_collected_32cubes = data_dict_for_vis["transformer_output_sequence_up_collected_32cubes"]
    diff_transformer_outputs_and_non_optimized_latent_codes = data_dict_for_vis["diff_transformer_outputs_and_non_optimized_latent_codes"]

    # plotting starts:------------------------------------------------------------------------------
    # True GT-----------------------------------------------------------------------------------------------------------
    collected_sub_voxels_array = collected_sub_voxels[batch_idx].squeeze().cpu().detach().numpy()
    # Optimized latent code march------------------------------------------------------------------------------------------------------------
    # NON Optimized latent code march------------------------------------------------------------------------------------------------------------
    collected_sub_voxels_decoded_non_optimized_array = collected_sub_voxels_decoded_non_optimized[batch_idx].squeeze(0).cpu().detach().numpy()
    # Masked latent code  march--------------------------------------------------------------------------------------------------------------
    collected_decoded_masked_non_optimized_latent_codes_array = collected_decoded_masked_non_optimized_latent_codes[batch_idx].squeeze(0).cpu().detach().numpy()
    # Transformer output march------------------------------------------------------------------------------------------------------------
    collected_transformer_output_sequence_up_array = transformer_output_sequence_up_collected_32cubes[batch_idx].squeeze(0).cpu().detach().numpy()
    # Diff--------------------------------------------------------------------------------
    diff_transformer_output_and_non_optimized_latent_code_array = diff_transformer_outputs_and_non_optimized_latent_codes[batch_idx].squeeze(0).cpu().detach().numpy()

    # generate plots for everything--------------------------------
    collected_data_dict_for_plotting = {
        "True_gt_sdf": collected_sub_voxels_array,
        "Non_optimized_latent_codes": collected_sub_voxels_decoded_non_optimized_array,
        "Masked_non_optimized_latent_codes": collected_decoded_masked_non_optimized_latent_codes_array,
        "Transformer_output": collected_transformer_output_sequence_up_array,
        "Diff_transformer_output_and_non_optimized_latent_codes": diff_transformer_output_and_non_optimized_latent_code_array,
    }

    return collected_data_dict_for_plotting


def collect_any_generated_data_for_plotting(data_dict_for_vis: dict, batch_idx: int) -> dict[str, Any]:
    keys = [key for key in data_dict_for_vis.keys()]
    return_dict = dict()
    for i in range(len(keys)):
        current_key = keys[i]
        collected_data_current = data_dict_for_vis.get(current_key).detach()
        collected_data_current_array = collected_data_current[batch_idx].squeeze().cpu().detach().numpy()
        return_dict[current_key] = collected_data_current_array
    return return_dict

def collect_generated_data_for_plotting(data_dict_for_vis: dict, resolution: int, batch_idx: int) -> dict[str, Any]:
    collected_sub_voxels = data_dict_for_vis["collected_sub_voxels"]
    collected_sub_voxels_decoded_optimized = data_dict_for_vis["collected_sub_voxels_decoded_optimized"]
    collected_decoded_masked_optimized_latent_codes = data_dict_for_vis["collected_decoded_masked_optimized_latent_codes"]
    transformer_output_sequence_up_collected_32cubes = data_dict_for_vis["transformer_output_sequence_up_collected_32cubes"]
    diff_transformer_outputs_and_optimized_latent_codes = data_dict_for_vis["diff_transformer_outputs_and_optimized_latent_codes"]

    # plotting starts:------------------------------------------------------------------------------
    # True GT-----------------------------------------------------------------------------------------------------------
    collected_sub_voxels_array = collected_sub_voxels[batch_idx].squeeze().cpu().detach().numpy()
    # Optimized latent code march------------------------------------------------------------------------------------------------------------
    collected_sub_voxels_decoded_optimized_array = collected_sub_voxels_decoded_optimized[batch_idx].squeeze(0).cpu().detach().numpy()
    # Masked latent code  march--------------------------------------------------------------------------------------------------------------
    collected_decoded_masked_optimized_latent_codes_array = collected_decoded_masked_optimized_latent_codes[batch_idx].squeeze(0).cpu().detach().numpy()
    # Transformer output march------------------------------------------------------------------------------------------------------------
    collected_transformer_output_sequence_up_array = transformer_output_sequence_up_collected_32cubes[batch_idx].squeeze(0).cpu().detach().numpy()
    # Diff--------------------------------------------------------------------------------
    diff_transformer_output_and_optimized_latent_code_array = diff_transformer_outputs_and_optimized_latent_codes[batch_idx].squeeze(0).cpu().detach().numpy()

    # generate plots for everything--------------------------------
    collected_data_dict_for_plotting = {
        "True_gt_sdf": collected_sub_voxels_array,
        "Optimized_latent_codes": collected_sub_voxels_decoded_optimized_array,
        "Masked_optimized_latent_codes": collected_decoded_masked_optimized_latent_codes_array,
        "Transformer_output": collected_transformer_output_sequence_up_array,
        "Diff_transformer_output_&_optimized_latent_codes": diff_transformer_output_and_optimized_latent_code_array,
    }
    return collected_data_dict_for_plotting
