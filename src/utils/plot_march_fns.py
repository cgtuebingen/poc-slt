import torch
from typing import Any

from src.utils.subvolume_devision import (
    collect_sub_voxels_to_voxel_with_batch,
)
from src.utils.m_cube_fns import marche_the_cube


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
        # # TODO: Checkme, may be I am wrong!!!, no you are not!
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
        # # TODO: Checkme, may be I am wrong!!!, no you are not!
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
def generate_data_for_plotting_TransAD_vs_gt(dict_arguments_for_vis: dict, dict_arguments_of_variables: dict, fdecoder) -> dict:
    optimized_latent_codes = dict_arguments_for_vis["optimized_latent_codes"]

    gt_sdf_voxel = dict_arguments_for_vis["gt_sdf_voxel"]

    number_of_sub_voxels = dict_arguments_of_variables["number_of_sub_voxels"]
    target_resolution = dict_arguments_of_variables["target_resolution"]
    resolution = dict_arguments_of_variables["resolution"]

    batch_size = gt_sdf_voxel.shape[0]
    # decoder--------------------------
    with torch.no_grad():
        decoded_optimized_latent_codes = fdecoder(optimized_latent_codes.clone()).to(device="cuda")  # [128, 512, 2, 2, 2] -> [128, 1, 32, 32, 32]
        decoded_optimized_latent_codes_reshaped = decoded_optimized_latent_codes.reshape([batch_size, number_of_sub_voxels, target_resolution, target_resolution, target_resolution])
        collected_decoded_optimized_latent_codes = collect_sub_voxels_to_voxel_with_batch(decoded_optimized_latent_codes_reshaped, resolution)
        assert collected_decoded_optimized_latent_codes.shape == (batch_size, resolution, resolution, resolution)
        del decoded_optimized_latent_codes
        del decoded_optimized_latent_codes_reshaped

        # True GT collect------------------------------------------------------------------------------------------------------------
        # collected_sub_voxels = collect_sub_voxels_to_voxel_with_batch(sub_voxels, resolution)

        data_dict_for_vis = {
            "gt_sdf_voxel": gt_sdf_voxel,
            "collected_sub_voxels_decoded_optimized": collected_decoded_optimized_latent_codes,
        }
        return data_dict_for_vis

def generate_data_for_plotting_vae_vs_gt(dict_arguments_for_vis: dict, dict_arguments_of_variables: dict, fdecoder) -> dict:
    non_optimized_latent_codes = dict_arguments_for_vis["non_optimized_latent_codes"]

    gt_sdf_voxel = dict_arguments_for_vis["gt_sdf_voxel"]

    number_of_sub_voxels = dict_arguments_of_variables["number_of_sub_voxels"]
    target_resolution = dict_arguments_of_variables["target_resolution"]
    resolution = dict_arguments_of_variables["resolution"]

    batch_size = gt_sdf_voxel.shape[0]
    # decoder--------------------------
    with torch.no_grad():
        decoded_non_optimized_latent_codes = fdecoder(non_optimized_latent_codes.clone()).to(device="cuda")  # [128, 512, 2, 2, 2] -> [128, 1, 32, 32, 32]
        decoded_non_optimized_latent_codes_reshaped = decoded_non_optimized_latent_codes.reshape([batch_size, number_of_sub_voxels, target_resolution, target_resolution, target_resolution])
        collected_sub_voxels_decoded_non_optimized = collect_sub_voxels_to_voxel_with_batch(decoded_non_optimized_latent_codes_reshaped, resolution)
        assert collected_sub_voxels_decoded_non_optimized.shape == (batch_size, resolution, resolution, resolution)
        del decoded_non_optimized_latent_codes
        del decoded_non_optimized_latent_codes_reshaped

        # True GT collect------------------------------------------------------------------------------------------------------------
        # collected_sub_voxels = collect_sub_voxels_to_voxel_with_batch(sub_voxels, resolution)

        data_dict_for_vis = {
            "gt_sdf_voxel": gt_sdf_voxel,
            "collected_sub_voxels_decoded_non_optimized": collected_sub_voxels_decoded_non_optimized,
        }
        return data_dict_for_vis

def generate_data_for_plottingv3(dict_arguments_for_vis: dict, dict_arguments_of_variables: dict, fdecoder) -> dict:
    optimized_latent_codes = dict_arguments_for_vis["optimized_latent_codes"]
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
        # Masked latent code collect------------------------------------------------------------------------------------------------------------
        # decoder-------------------
        decoded_masked_optimized_latent_codes = fdecoder(masked_optimized_latent_codes).to(device="cuda")
        decoded_masked_optimized_latent_codes_reshaped = decoded_masked_optimized_latent_codes.reshape([batch_size, number_of_sub_voxels, target_resolution, target_resolution, target_resolution])
        collected_decoded_masked_optimized_latent_codes = collect_sub_voxels_to_voxel_with_batch(decoded_masked_optimized_latent_codes_reshaped, resolution)
        del decoded_masked_optimized_latent_codes_reshaped
        assert collected_decoded_masked_optimized_latent_codes.shape == (batch_size, resolution, resolution, resolution)
        # Transformer output collect------------------------------------------------------------------------------------------------------------
        # # TODO: Checkme, may be I am wrong!!!, no you are not!
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
            "collected_decoded_masked_optimized_latent_codes": collected_decoded_masked_optimized_latent_codes,
            "transformer_output_sequence_up_collected_32cubes": transformer_output_sequence_up_collected_32cubes,
            "diff_transformer_outputs_and_optimized_latent_codes": diff_transformer_outputs_and_optimized_latent_codes,
        }
        return data_dict_for_vis

def generate_data_for_plottingv4(dict_arguments_for_vis: dict, dict_arguments_of_variables: dict, fdecoder) -> dict:
    non_optimized_latent_codes = dict_arguments_for_vis["non_optimized_latent_codes"]
    optimized_latent_codes = dict_arguments_for_vis["optimized_latent_codes"]
    sub_voxels = dict_arguments_for_vis["sub_voxels"]

    number_of_sub_voxels = dict_arguments_of_variables["number_of_sub_voxels"]
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
        #non latent code collect------------------------------------------------------------------------------------------------------------
        # decoder-------------------
        decoded_non_optimized_latent_codes = fdecoder(non_optimized_latent_codes).to(device="cuda")
        decoded_non_optimized_latent_codes_reshaped = decoded_non_optimized_latent_codes.reshape([batch_size, number_of_sub_voxels, target_resolution, target_resolution, target_resolution])
        collected_decoded_non_optimized_latent_codes = collect_sub_voxels_to_voxel_with_batch(decoded_non_optimized_latent_codes_reshaped, resolution)
        del decoded_non_optimized_latent_codes_reshaped
        assert collected_decoded_non_optimized_latent_codes.shape == (batch_size, resolution, resolution, resolution)

        # True GT collect------------------------------------------------------------------------------------------------------------
        # True GT collect------------------------------------------------------------------------------------------------------------
        collected_sub_voxels = collect_sub_voxels_to_voxel_with_batch(sub_voxels, resolution)

        data_dict_for_vis = {
            "collected_sub_voxels": collected_sub_voxels,
            "collected_sub_voxels_decoded_optimized": collected_sub_voxels_decoded_optimized,
            "collected_sub_voxels_decoded_non_optimized": collected_decoded_non_optimized_latent_codes,
        }
        return data_dict_for_vis

def generate_data_for_plottingv5(dict_arguments_for_vis: dict, dict_arguments_of_variables: dict, fdecoder) -> dict:

    optimized_latent_codes = dict_arguments_for_vis["optimized_latent_codes"]
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
        decoded_masked_non_optimized_latent_codes = fdecoder(masked_non_optimized_latent_codes).to(device="cuda")
        decoded_masked_non_optimized_latent_codes_reshaped = decoded_masked_non_optimized_latent_codes.reshape([batch_size, number_of_sub_voxels, target_resolution, target_resolution, target_resolution])
        collected_decoded_masked_non_optimized_latent_codes = collect_sub_voxels_to_voxel_with_batch(decoded_masked_non_optimized_latent_codes_reshaped, resolution)
        del decoded_masked_non_optimized_latent_codes_reshaped
        assert collected_decoded_masked_non_optimized_latent_codes.shape == (batch_size, resolution, resolution, resolution)
        # Transformer output collect------------------------------------------------------------------------------------------------------------
        # # TODO: Checkme, may be I am wrong!!!, no you are not!
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
        # diff_transformer_outputs_and_optimized_latent_codes = torch.subtract(transformer_output_sequence_up_collected_32cubes, collected_sub_voxels_decoded_optimized)
        # True GT collect------------------------------------------------------------------------------------------------------------
        collected_sub_voxels = collect_sub_voxels_to_voxel_with_batch(sub_voxels, resolution)

        data_dict_for_vis = {
            "collected_sub_voxels": collected_sub_voxels,
            "collected_decoded_non_optimized_latent_codes": collected_decoded_non_optimized_latent_codes_reshaped,
            "collected_sub_voxels_decoded_optimized": collected_sub_voxels_decoded_optimized,
            "collected_decoded_non_masked_optimized_latent_codes": collected_decoded_masked_non_optimized_latent_codes,
            "transformer_output_sequence_up_collected_32cubes": transformer_output_sequence_up_collected_32cubes,
            # "diff_transformer_outputs_and_optimized_latent_codes": diff_transformer_outputs_and_optimized_latent_codes,
        }
        return data_dict_for_vis


def generate_data_for_plottingv7(dict_arguments_for_vis: dict, dict_arguments_of_variables: dict, fdecoder) -> dict:
    # dict_arguments_for_vis = {
    #     "non_optimized_latent_codes": non_optimized_latent_codes,
    #     "masked_non_optimized_latent_codes": masked_non_optimized_latent_codes,
    #     "transformer_output_sequence_up": transformer_output_sequence_up,
    #     "masked_transformer_output_sequence_up_ad": masked_transformer_output_sequence_up_ad,  # not masked actually except for empty ones
    #     "transformer_output_sequence_up_ad": transformer_output_sequence_up_ad,  # the refined output
    #     "sub_voxels": sub_voxels,
    #     "optimized_latent_codes": optimized_latent_codes.to(device=self.device).to(dtype=torch.float32),
    # }

    optimized_latent_codes = dict_arguments_for_vis["optimized_latent_codes"]
    non_optimized_latent_codes = dict_arguments_for_vis["non_optimized_latent_codes"]
    masked_non_optimized_latent_codes = dict_arguments_for_vis["masked_non_optimized_latent_codes"]
    transformer_output_sequence_up_reshaped = dict_arguments_for_vis["transformer_output_sequence_up"]
    sub_voxels = dict_arguments_for_vis["sub_voxels"]

    # masked_transformer_output_sequence_up_ad = dict_arguments_for_vis["masked_transformer_output_sequence_up_ad"]
    transformer_output_sequence_up_ad = dict_arguments_for_vis["transformer_output_sequence_up_ad"]


    number_of_sub_voxels = dict_arguments_of_variables["number_of_sub_voxels"]
    latent_dim = dict_arguments_of_variables["latent_dim"]
    target_resolution = dict_arguments_of_variables["target_resolution"]
    resolution = dict_arguments_of_variables["resolution"]

    batch_size = sub_voxels.shape[0]
    # decoder--------------------------
    with torch.no_grad():
        # # masked_transformer_output_sequence_up_ad------------------------------------------------------------------------------------------------------------
        # decoded_masked_transformer_output_sequence_up_ad = fdecoder(masked_transformer_output_sequence_up_ad.clone()).to(device="cuda")  # [128, 512, 2, 2, 2] -> [128, 1, 32, 32, 32]
        # decoded_masked_transformer_output_sequence_up_ad_reshaped = decoded_masked_transformer_output_sequence_up_ad.reshape([batch_size, number_of_sub_voxels, target_resolution, target_resolution, target_resolution])
        # collected_decoded_masked_transformer_output_sequence_up_ad = collect_sub_voxels_to_voxel_with_batch(decoded_masked_transformer_output_sequence_up_ad_reshaped, resolution)
        # assert collected_decoded_masked_transformer_output_sequence_up_ad.shape == (batch_size, resolution, resolution, resolution)
        # del decoded_masked_transformer_output_sequence_up_ad
        # del decoded_masked_transformer_output_sequence_up_ad_reshaped
        # transformer_output_sequence_up_ad------------------------------------------------------------------------------------------------------------
        decoded_transformer_output_sequence_up_ad = fdecoder(transformer_output_sequence_up_ad.clone()).to(device="cuda")  # [128, 512, 2, 2, 2] -> [128, 1, 32, 32, 32]
        decoded_transformer_output_sequence_up_ad_reshaped = decoded_transformer_output_sequence_up_ad.reshape([batch_size, number_of_sub_voxels, target_resolution, target_resolution, target_resolution])
        collected_decoded_transformer_output_sequence_up_ad_reshaped = collect_sub_voxels_to_voxel_with_batch(decoded_transformer_output_sequence_up_ad_reshaped, resolution)
        assert collected_decoded_transformer_output_sequence_up_ad_reshaped.shape == (batch_size, resolution, resolution, resolution)
        del decoded_transformer_output_sequence_up_ad
        del decoded_transformer_output_sequence_up_ad_reshaped
        # optimized_latent_code------------------------------------------------------------------------------------------------------------
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
        decoded_masked_non_optimized_latent_codes = fdecoder(masked_non_optimized_latent_codes).to(device="cuda")
        decoded_masked_non_optimized_latent_codes_reshaped = decoded_masked_non_optimized_latent_codes.reshape([batch_size, number_of_sub_voxels, target_resolution, target_resolution, target_resolution])
        collected_decoded_masked_non_optimized_latent_codes = collect_sub_voxels_to_voxel_with_batch(decoded_masked_non_optimized_latent_codes_reshaped, resolution)
        del decoded_masked_non_optimized_latent_codes_reshaped
        assert collected_decoded_masked_non_optimized_latent_codes.shape == (batch_size, resolution, resolution, resolution)
        # Transformer output collect------------------------------------------------------------------------------------------------------------
        # # TODO: Checkme, may be I am wrong!!!, no you are not!
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
        # diff_transformer_outputs_and_optimized_latent_codes = torch.subtract(transformer_output_sequence_up_collected_32cubes, collected_sub_voxels_decoded_optimized)
        # True GT collect------------------------------------------------------------------------------------------------------------
        collected_sub_voxels = collect_sub_voxels_to_voxel_with_batch(sub_voxels, resolution)

        data_dict_for_vis = {
            "collected_sub_voxels": collected_sub_voxels,
            "collected_decoded_non_optimized_latent_codes": collected_decoded_non_optimized_latent_codes_reshaped,
            "collected_sub_voxels_decoded_optimized": collected_sub_voxels_decoded_optimized,
            "collected_decoded_non_masked_optimized_latent_codes": collected_decoded_masked_non_optimized_latent_codes,
            "transformer_output_sequence_up_collected_32cubes": transformer_output_sequence_up_collected_32cubes,
            # "collected_decoded_masked_transformer_output_sequence_up_ad":collected_decoded_masked_transformer_output_sequence_up_ad,
            "collected_decoded_transformer_output_sequence_up_ad": collected_decoded_transformer_output_sequence_up_ad_reshaped,

            # "diff_transformer_outputs_and_optimized_latent_codes": diff_transformer_outputs_and_optimized_latent_codes,
        }
        return data_dict_for_vis
def generate_data_for_plottingv6(dict_arguments_for_vis: dict, dict_arguments_of_variables: dict, fdecoder) -> dict:

    optimized_latent_codes = dict_arguments_for_vis["optimized_latent_codes"]
    non_optimized_latent_codes = dict_arguments_for_vis["non_optimized_latent_codes"]
    masked_noisy_non_optimized_latent_codes = dict_arguments_for_vis["masked_noisy_non_optimized_latent_codes"]
    transformer_output_sequence_up_reshaped = dict_arguments_for_vis["transformer_output_sequence_up"]
    noisy_non_optimized_latent_code = dict_arguments_for_vis["noisy_non_optimized_latent_code"]
    sub_voxels = dict_arguments_for_vis["sub_voxels"]
    # noisy_gt_sdf = dict_arguments_for_vis["noisy_gt_sdf"]

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
        decoded_masked_noisy_non_optimized_latent_codes = fdecoder(masked_noisy_non_optimized_latent_codes).to(device="cuda")
        decoded_masked_noisy_non_optimized_latent_codes_reshaped = decoded_masked_noisy_non_optimized_latent_codes.reshape([batch_size, number_of_sub_voxels, target_resolution, target_resolution, target_resolution])
        collected_decoded_masked_noisy_non_optimized_latent_codes = collect_sub_voxels_to_voxel_with_batch(decoded_masked_noisy_non_optimized_latent_codes_reshaped, resolution)
        del decoded_masked_noisy_non_optimized_latent_codes_reshaped
        assert collected_decoded_masked_noisy_non_optimized_latent_codes.shape == (batch_size, resolution, resolution, resolution)

        # noisy decoder-------------------
        decoded_noisy_non_optimized_latent_code = fdecoder(noisy_non_optimized_latent_code).to(device="cuda")
        decoded_noisy_non_optimized_latent_code_reshaped = decoded_noisy_non_optimized_latent_code.reshape(
            [batch_size, number_of_sub_voxels, target_resolution, target_resolution, target_resolution])
        collected_decoded_noisy_non_optimized_latent_code = collect_sub_voxels_to_voxel_with_batch(decoded_noisy_non_optimized_latent_code_reshaped, resolution)
        del decoded_noisy_non_optimized_latent_code_reshaped
        assert collected_decoded_noisy_non_optimized_latent_code.shape == (batch_size, resolution, resolution, resolution)
        # Transformer output collect------------------------------------------------------------------------------------------------------------
        # # TODO: Checkme, may be I am wrong!!!, no you are not!
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
        # diff_transformer_outputs_and_optimized_latent_codes = torch.subtract(transformer_output_sequence_up_collected_32cubes, collected_sub_voxels_decoded_optimized)
        # True GT collect------------------------------------------------------------------------------------------------------------
        collected_sub_voxels = collect_sub_voxels_to_voxel_with_batch(sub_voxels, resolution)

        data_dict_for_vis = {
            "collected_sub_voxels": collected_sub_voxels,
            "collected_decoded_non_optimized_latent_codes": collected_decoded_non_optimized_latent_codes_reshaped,
            "collected_sub_voxels_decoded_optimized": collected_sub_voxels_decoded_optimized,
            "collected_decoded_masked_noisy_non_optimized_latent_codes": collected_decoded_masked_noisy_non_optimized_latent_codes,
            "transformer_output_sequence_up_collected_32cubes": transformer_output_sequence_up_collected_32cubes,
            # "diff_transformer_outputs_and_optimized_latent_codes": diff_transformer_outputs_and_optimized_latent_codes,
            "collected_decoded_noisy_non_optimized_latent_code": collected_decoded_noisy_non_optimized_latent_code,
            # "noisy_gt_sdf": noisy_gt_sdf,
        }
        return data_dict_for_vis


def collect_generated_data_for_plottingv2(data_dict_for_vis: dict, resolution: int, batch_idx: int) -> dict[str, Any]:
    collected_sub_voxels = data_dict_for_vis["collected_sub_voxels"]
    collected_decoded_optimized_latent_codes = data_dict_for_vis["optimized_latent_codes"]
    collected_sub_voxels_decoded_non_optimized = data_dict_for_vis["collected_sub_voxels_decoded_non_optimized"]
    collected_decoded_masked_optimized_latent_codes = data_dict_for_vis["collected_decoded_masked_optimized_latent_codes"]
    transformer_output_sequence_up_collected_32cubes = data_dict_for_vis["transformer_output_sequence_up_collected_32cubes"]
    diff_transformer_outputs_and_non_optimized_latent_codes = data_dict_for_vis["diff_transformer_outputs_and_non_optimized_latent_codes"]

    # plotting starts:------------------------------------------------------------------------------
    # True GT-----------------------------------------------------------------------------------------------------------
    collected_sub_voxels_array = collected_sub_voxels[batch_idx].squeeze().cpu().detach().numpy()
    # Optimized latent code march------------------------------------------------------------------------------------------------------------
    collected_decoded_optimized_latent_codes_array = collected_decoded_optimized_latent_codes[batch_idx].squeeze().cpu().detach().numpy()
    # NON Optimized latent code march------------------------------------------------------------------------------------------------------------
    collected_sub_voxels_decoded_non_optimized_array = collected_sub_voxels_decoded_non_optimized[batch_idx].squeeze(0).cpu().detach().numpy()
    # Masked latent code  march--------------------------------------------------------------------------------------------------------------
    collected_decoded_masked_optimized_latent_codes_array = collected_decoded_masked_optimized_latent_codes[batch_idx].squeeze(0).cpu().detach().numpy()
    # Transformer output march------------------------------------------------------------------------------------------------------------
    collected_transformer_output_sequence_up_array = transformer_output_sequence_up_collected_32cubes[batch_idx].squeeze(0).cpu().detach().numpy()
    # Diff--------------------------------------------------------------------------------
    diff_transformer_output_and_non_optimized_latent_code_array = diff_transformer_outputs_and_non_optimized_latent_codes[batch_idx].squeeze(0).cpu().detach().numpy()

    # generate plots for everything--------------------------------
    collected_data_dict_for_plotting = {
        "True_gt_sdf": collected_sub_voxels_array,
        "Optimized_latent_codes": collected_decoded_optimized_latent_codes_array,
        "Non_optimized_latent_codes": collected_sub_voxels_decoded_non_optimized_array,
        "Masked_optimized_latent_codes": collected_decoded_masked_optimized_latent_codes_array,
        "Transformer_output": collected_transformer_output_sequence_up_array,
        "Diff_transformer_output_and_non_optimized_latent_codes": diff_transformer_output_and_non_optimized_latent_code_array,
    }
    return collected_data_dict_for_plotting
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
def collect_generated_data_for_plottingv4(data_dict_for_vis: dict, resolution: int, batch_idx: int) -> dict[str, Any]:
    collected_sub_voxels = data_dict_for_vis["collected_sub_voxels"]
    collected_sub_voxels_decoded_optimized = data_dict_for_vis["collected_sub_voxels_decoded_optimized"]

    collected_decoded_masked_optimized_latent_codes = data_dict_for_vis["collected_decoded_masked_optimized_latent_codes"]
    transformer_output_sequence_up_collected_32cubes = data_dict_for_vis["transformer_output_sequence_up_collected_32cubes"]
    diff_transformer_outputs_and_optimized_latent_codes = data_dict_for_vis["diff_transformer_outputs_and_non_optimized_latent_codes"]

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
        "optimized_latent_codes": collected_sub_voxels_decoded_optimized_array,
        "Masked_optimized_latent_codes": collected_decoded_masked_optimized_latent_codes_array,
        "Transformer_output": collected_transformer_output_sequence_up_array,
        "Diff_transformer_output_and__optimized_latent_codes": diff_transformer_output_and_optimized_latent_code_array,
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
def collect_generated_data_for_plottingv2(data_dict_for_vis: dict, resolution: int, batch_idx: int) -> dict[str, Any]:
    collected_sub_voxels = data_dict_for_vis["collected_sub_voxels"]
    collected_sub_voxels_decoded_non_optimized = data_dict_for_vis["collected_decoded_non_optimized_latent_codes"]
    collected_sub_voxels_decoded_optimized = data_dict_for_vis["collected_sub_voxels_decoded_optimized"]

    collected_decoded_masked_optimized_latent_codes = data_dict_for_vis["collected_decoded_masked_optimized_latent_codes"]
    transformer_output_sequence_up_collected_32cubes = data_dict_for_vis["transformer_output_sequence_up_collected_32cubes"]
    diff_transformer_outputs_and_optimized_latent_codes = data_dict_for_vis["diff_transformer_outputs_and_optimized_latent_codes"]

    # plotting starts:------------------------------------------------------------------------------
    # True GT-----------------------------------------------------------------------------------------------------------
    collected_sub_voxels_array = collected_sub_voxels[batch_idx].squeeze().cpu().detach().numpy()
    # NON Optimized latent code march------------------------------------------------------------------------------------------------------------
    collected_sub_voxels_decoded_non_optimized_array = collected_sub_voxels_decoded_non_optimized[batch_idx].squeeze().cpu().detach().numpy()
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
        "Non_Optimized_latent_codes": collected_sub_voxels_decoded_non_optimized_array,
        "Optimized_latent_codes": collected_sub_voxels_decoded_optimized_array,
        "Masked_optimized_latent_codes": collected_decoded_masked_optimized_latent_codes_array,
        "Transformer_output": collected_transformer_output_sequence_up_array,
        "Diff_transformer_output_&_optimized_latent_codes": diff_transformer_output_and_optimized_latent_code_array,
    }
    return collected_data_dict_for_plotting

def march_results_every_n_epoch(dict_of_items: dict, selected_index: int, current_epoch: int, global_step: int, marching_cube_result_dir: str) -> None:
    if (global_step // 5000) == 0:
        collected_32cubes_array = dict_of_items["True_gt_sdf"]
        collected_sub_voxels_decoded_non_optimized_array = dict_of_items["Non_Optimized_latent_codes"]
        collected_sub_voxels_decoded_optimized_array = dict_of_items["Optimized_latent_codes"]
        collected_transformer_output_sequence_up_array = dict_of_items["Transformer_output"]

        current_step = global_step
        marche_the_cube(collected_32cubes_array, current_epoch, current_step, marching_cube_result_dir, "True_Gt_collected_sub_voxels-Obj-ID-", selected_index)
        marche_the_cube(collected_sub_voxels_decoded_non_optimized_array, current_epoch, current_step, marching_cube_result_dir, "Non-Optimized-collected_decoded_sub_voxels-Obj-ID-", selected_index)
        marche_the_cube(collected_sub_voxels_decoded_optimized_array, current_epoch, current_step, marching_cube_result_dir, "Optimized-collected_decoded_sub_voxels-Obj-ID-", selected_index)
        marche_the_cube(collected_transformer_output_sequence_up_array, current_epoch, current_step, marching_cube_result_dir, "Transformer_output-Obj-ID-", selected_index)


# for repairing latent_codes_vs_DeepSDF-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def generate_data_for_plotting_repairLTvsDeepSDF(dict_arguments_for_vis: dict, dict_arguments_of_variables: dict, fdecoder) -> dict:
    non_optimized_latent_codes = dict_arguments_for_vis["non_optimized_latent_codes"]
    optimized_latent_codes = dict_arguments_for_vis["optimized_latent_codes"]
    masked_non_optimized_latent_codes = dict_arguments_for_vis["masked_non_optimized_latent_codes"]
    transformer_output_sequence_up = dict_arguments_for_vis["transformer_output_sequence_up"]
    sub_voxels = dict_arguments_for_vis["sub_voxels"]

    number_of_sub_voxels = dict_arguments_of_variables["number_of_sub_voxels"]
    latent_dim = dict_arguments_of_variables["latent_dim"]
    target_resolution = dict_arguments_of_variables["target_resolution"]
    resolution = dict_arguments_of_variables["resolution"]

    batch_size = sub_voxels.shape[0]
    # decoder--------------------------
    with torch.no_grad():
        # non-optimized latent-codes--------------------------------------------------------------------------------------------------------------------
        non_optimized_latent_codes_reshaped = non_optimized_latent_codes.reshape(batch_size, number_of_sub_voxels, latent_dim, 2, 2, 2)
        assert non_optimized_latent_codes_reshaped.shape == (batch_size, number_of_sub_voxels, latent_dim, 2, 2, 2)
        # Non-Optimized latent code collect------------------------------------------------------------------------------------------------------------
        # decoder--------------------------
        decoded_non_optimized_latent_codes = fdecoder(non_optimized_latent_codes_reshaped.clone()).to(device=fdecoder.device)  # [128, 512, 2, 2, 2] -> [128, 1, 32, 32, 32]
        decoded_non_optimized_latent_codes_reshaped = decoded_non_optimized_latent_codes.reshape([batch_size, number_of_sub_voxels, target_resolution, target_resolution, target_resolution])
        collected_decoded_non_optimized_latent_codes = collect_sub_voxels_to_voxel_with_batch(decoded_non_optimized_latent_codes_reshaped, resolution)
        assert collected_decoded_non_optimized_latent_codes.shape == (batch_size, resolution, resolution, resolution)
        del decoded_non_optimized_latent_codes
        del decoded_non_optimized_latent_codes_reshaped
        # optimized latent-codes--------------------------------------------------------------------------------------------------------------------
        decoded_optimized_latent_codes = fdecoder(optimized_latent_codes.clone()).to(device=fdecoder.device)  # [128, 512, 2, 2, 2] -> [128, 1, 32, 32, 32]
        decoded_optimized_latent_codes_reshaped = decoded_optimized_latent_codes.reshape([batch_size, number_of_sub_voxels, target_resolution, target_resolution, target_resolution])
        collected_decoded_optimized_latent_codes = collect_sub_voxels_to_voxel_with_batch(decoded_optimized_latent_codes_reshaped, resolution)
        assert collected_decoded_optimized_latent_codes.shape == (batch_size, resolution, resolution, resolution)
        del decoded_optimized_latent_codes
        del decoded_optimized_latent_codes_reshaped
        # Masked-non-latent-codes------------------------------------------------------------------------------------------------------------
        decoded_masked_non_optimized_latent_codes = fdecoder(masked_non_optimized_latent_codes.clone()).to(device=fdecoder.device)
        decoded_masked_non_optimized_latent_codes_reshaped = decoded_masked_non_optimized_latent_codes.reshape([batch_size, number_of_sub_voxels, target_resolution, target_resolution, target_resolution])
        collected_decoded_masked_non_optimized_latent_codes_reshaped = collect_sub_voxels_to_voxel_with_batch(decoded_masked_non_optimized_latent_codes_reshaped, resolution)
        assert collected_decoded_masked_non_optimized_latent_codes_reshaped.shape == (batch_size, resolution, resolution, resolution)
        del decoded_masked_non_optimized_latent_codes
        del decoded_masked_non_optimized_latent_codes_reshaped
        # Diff of non-optimized-latent_codes and optimized latent codes from DeepSDF-----------------------------------------------------------
        diff_non_optimized_latent_codes_vs_optimized_latent_codes_fromDeepSDF = torch.subtract(collected_decoded_optimized_latent_codes, collected_decoded_non_optimized_latent_codes)
        # Transformer output collect------------------------------------------------------------------------------------------------------------
        # TODO: Checkme, may be I am wrong!!!, no you are not!
        if target_resolution == 32:
            transformer_output_sequence_up_reshaped = (
                transformer_output_sequence_up.reshape(
                    [
                        batch_size,
                        number_of_sub_voxels,
                        latent_dim,
                        2,
                        2,
                        2,
                    ]
                )
                .to(device=fdecoder.device)
                .to(dtype=torch.float32)
            )
        elif ((target_resolution == 16) or (target_resolution == 8)):
            transformer_output_sequence_up_reshaped = (
                transformer_output_sequence_up.reshape(
                    [
                        batch_size,
                        number_of_sub_voxels,
                        latent_dim,
                        1,
                        1,
                        1,
                    ]
                )
                .to(device=fdecoder.device)
                .to(dtype=torch.float32)
            )

        decoded_transformer_output_sequence_up_reshaped = fdecoder(transformer_output_sequence_up_reshaped.clone()).to(device=fdecoder.device)
        decoded_transformer_output_sequence_up_reshaped_reshaped = decoded_transformer_output_sequence_up_reshaped.reshape(
            batch_size,
            number_of_sub_voxels,
            target_resolution,
            target_resolution,
            target_resolution,
        ).to(device=fdecoder.device)
        collected_transformer_output_sequence_up = collect_sub_voxels_to_voxel_with_batch(decoded_transformer_output_sequence_up_reshaped_reshaped, resolution)
        del decoded_transformer_output_sequence_up_reshaped
        del decoded_transformer_output_sequence_up_reshaped_reshaped
        # Diff--------------------------------------------------------------------------------
        diff_transformer_outputs_and_optimized_latent_codes = torch.subtract(collected_transformer_output_sequence_up, collected_decoded_optimized_latent_codes)
        # collect sub-voxels------------------------------------------------------------------------------------------------------------------
        collected_sub_voxels = collect_sub_voxels_to_voxel_with_batch(sub_voxels, resolution)
        data_dict_for_vis = {
            "collected_sub_voxels": collected_sub_voxels,
            "collected_decoded_non_optimized_latent_codes": collected_decoded_non_optimized_latent_codes,
            "collected_decoded_optimized_latent_codes": collected_decoded_optimized_latent_codes,
            "collected_decoded_masked_non_optimized_latent_codes": collected_decoded_masked_non_optimized_latent_codes_reshaped,
            "collected_transformer_output_sequence": collected_transformer_output_sequence_up,
            "diff_non_optimized_latent_codes_vs_optimized_latent_codes_fromDeepSDF": diff_non_optimized_latent_codes_vs_optimized_latent_codes_fromDeepSDF,
            "diff_transformer_outputs_and_optimized_latent_codes": diff_transformer_outputs_and_optimized_latent_codes,
        }
        return data_dict_for_vis
def generate_data_for_plotting_repairLTvsDeepSDF16cube(dict_arguments_for_vis: dict, dict_arguments_of_variables: dict, fdecoder) -> dict:
    non_optimized_latent_codes = dict_arguments_for_vis["non_optimized_latent_codes"]
    optimized_latent_codes = dict_arguments_for_vis["optimized_latent_codes"]
    masked_non_optimized_latent_codes = dict_arguments_for_vis["masked_non_optimized_latent_codes"]
    transformer_output_sequence_up = dict_arguments_for_vis["transformer_output_sequence_up"]
    sub_voxels = dict_arguments_for_vis["sub_voxels"]

    number_of_sub_voxels = dict_arguments_of_variables["number_of_sub_voxels"]
    latent_dim = dict_arguments_of_variables["latent_dim"]
    target_resolution = dict_arguments_of_variables["target_resolution"]
    resolution = dict_arguments_of_variables["resolution"]

    batch_size = sub_voxels.shape[0]
    # decoder--------------------------
    with torch.no_grad():
        # non-optimized latent-codes--------------------------------------------------------------------------------------------------------------------
        non_optimized_latent_codes_reshaped = non_optimized_latent_codes.reshape(batch_size, number_of_sub_voxels, latent_dim, 1, 1, 1)
        assert non_optimized_latent_codes_reshaped.shape == (batch_size, number_of_sub_voxels, latent_dim, 1, 1, 1)
        # Non-Optimized latent code collect------------------------------------------------------------------------------------------------------------
        # decoder--------------------------
        decoded_non_optimized_latent_codes = fdecoder(non_optimized_latent_codes_reshaped.clone()).to(device=fdecoder.device)  # [128, 512, 1, 1, 1] -> [128, 1, 16, 16, 16]
        decoded_non_optimized_latent_codes_reshaped = decoded_non_optimized_latent_codes.reshape([batch_size, number_of_sub_voxels, target_resolution, target_resolution, target_resolution])
        collected_decoded_non_optimized_latent_codes = collect_sub_voxels_to_voxel_with_batch(decoded_non_optimized_latent_codes_reshaped, resolution)
        assert collected_decoded_non_optimized_latent_codes.shape == (batch_size, resolution, resolution, resolution)
        del decoded_non_optimized_latent_codes
        del decoded_non_optimized_latent_codes_reshaped
        # optimized latent-codes--------------------------------------------------------------------------------------------------------------------
        decoded_optimized_latent_codes = fdecoder(optimized_latent_codes.clone()).to(device=fdecoder.device)  # [128, 512, 1, 1, 1] -> [128, 1, 16, 16, 16]
        decoded_optimized_latent_codes_reshaped = decoded_optimized_latent_codes.reshape([batch_size, number_of_sub_voxels, target_resolution, target_resolution, target_resolution])
        collected_decoded_optimized_latent_codes = collect_sub_voxels_to_voxel_with_batch(decoded_optimized_latent_codes_reshaped, resolution)
        assert collected_decoded_optimized_latent_codes.shape == (batch_size, resolution, resolution, resolution)
        del decoded_optimized_latent_codes
        del decoded_optimized_latent_codes_reshaped
        # Masked-non-latent-codes------------------------------------------------------------------------------------------------------------
        decoded_masked_non_optimized_latent_codes = fdecoder(masked_non_optimized_latent_codes.clone()).to(device=fdecoder.device)
        decoded_masked_non_optimized_latent_codes_reshaped = decoded_masked_non_optimized_latent_codes.reshape([batch_size, number_of_sub_voxels, target_resolution, target_resolution, target_resolution])
        collected_decoded_masked_non_optimized_latent_codes_reshaped = collect_sub_voxels_to_voxel_with_batch(decoded_masked_non_optimized_latent_codes_reshaped, resolution)
        assert collected_decoded_masked_non_optimized_latent_codes_reshaped.shape == (batch_size, resolution, resolution, resolution)
        del decoded_masked_non_optimized_latent_codes
        del decoded_masked_non_optimized_latent_codes_reshaped
        # Diff of non-optimized-latent_codes and optimized latent codes from DeepSDF-----------------------------------------------------------
        diff_non_optimized_latent_codes_vs_optimized_latent_codes_fromDeepSDF = torch.subtract(collected_decoded_optimized_latent_codes, collected_decoded_non_optimized_latent_codes)
        # Transformer output collect------------------------------------------------------------------------------------------------------------
        # TODO: Checkme, may be I am wrong!!!, no you are not!

        if True and ((target_resolution == 16) or (target_resolution == 8)):
            transformer_output_sequence_up_reshaped = (
                transformer_output_sequence_up.reshape(
                    [
                        batch_size,
                        number_of_sub_voxels,
                        latent_dim,
                        1,
                        1,
                        1,
                    ]
                )
                .to(device=fdecoder.device)
                .to(dtype=torch.float32)
            )

        decoded_transformer_output_sequence_up_reshaped = fdecoder(transformer_output_sequence_up_reshaped.clone()).to(device=fdecoder.device)
        decoded_transformer_output_sequence_up_reshaped_reshaped = decoded_transformer_output_sequence_up_reshaped.reshape(
            batch_size,
            number_of_sub_voxels,
            target_resolution,
            target_resolution,
            target_resolution,
        ).to(device=fdecoder.device)
        collected_transformer_output_sequence_up = collect_sub_voxels_to_voxel_with_batch(decoded_transformer_output_sequence_up_reshaped_reshaped, resolution)
        del decoded_transformer_output_sequence_up_reshaped
        del decoded_transformer_output_sequence_up_reshaped_reshaped
        # Diff--------------------------------------------------------------------------------
        diff_transformer_outputs_and_optimized_latent_codes = torch.subtract(collected_transformer_output_sequence_up, collected_decoded_optimized_latent_codes)
        # collect sub-voxels------------------------------------------------------------------------------------------------------------------
        collected_sub_voxels = collect_sub_voxels_to_voxel_with_batch(sub_voxels, resolution)
        data_dict_for_vis = {
            "collected_sub_voxels": collected_sub_voxels,
            "collected_decoded_non_optimized_latent_codes": collected_decoded_non_optimized_latent_codes,
            "collected_decoded_optimized_latent_codes": collected_decoded_optimized_latent_codes,
            "collected_decoded_masked_non_optimized_latent_codes": collected_decoded_masked_non_optimized_latent_codes_reshaped,
            "collected_transformer_output_sequence": collected_transformer_output_sequence_up,
            "diff_non_optimized_latent_codes_vs_optimized_latent_codes_fromDeepSDF": diff_non_optimized_latent_codes_vs_optimized_latent_codes_fromDeepSDF,
            "diff_transformer_outputs_and_optimized_latent_codes": diff_transformer_outputs_and_optimized_latent_codes,
        }
        return data_dict_for_vis

def collect_generated_data_for_plotting_repairLTvsDeepSDF(data_dict_for_vis: dict, resolution: int, batch_idx: int) -> dict[str, Any]:
    collected_sub_voxels = data_dict_for_vis["collected_sub_voxels"]
    collected_decoded_non_optimized_latent_codes = data_dict_for_vis["collected_decoded_non_optimized_latent_codes"]
    collected_decoded_optimized_latent_codes = data_dict_for_vis["collected_decoded_optimized_latent_codes"]
    collected_decoded_masked_non_optimized_latent_codes = data_dict_for_vis["collected_decoded_masked_non_optimized_latent_codes"]
    collected_transformer_output_sequence = data_dict_for_vis["collected_transformer_output_sequence"]
    diff_non_optimized_latent_codes_vs_optimized_latent_codes_fromDeepSDF = data_dict_for_vis["diff_non_optimized_latent_codes_vs_optimized_latent_codes_fromDeepSDF"]
    diff_transformer_outputs_and_optimized_latent_codes = data_dict_for_vis["diff_transformer_outputs_and_optimized_latent_codes"]

    # plotting starts:------------------------------------------------------------------------------
    # True-GT march------------------------------------------------------------------------------------------------------------
    collected_sub_voxels_array = collected_sub_voxels[batch_idx].squeeze().cpu().detach().numpy()
    # # Non-Optimized-latent-codes march------------------------------------------------------------------------------------------------------------
    collected_decoded_non_optimized_latent_codes_array = collected_decoded_non_optimized_latent_codes[batch_idx].squeeze(0).cpu().detach().numpy()
    # Optimized-latent-codes march------------------------------------------------------------------------------------------------------------
    collected_decoded_optimized_latent_codes_array = collected_decoded_optimized_latent_codes[batch_idx].squeeze(0).cpu().detach().numpy()
    # Masked-non-optimized-latent-codes  march--------------------------------------------------------------------------------------------------------------
    collected_decoded_masked_non_optimized_latent_codes_array = collected_decoded_masked_non_optimized_latent_codes[batch_idx].squeeze(0).cpu().detach().numpy()
    # Transformer-output march------------------------------------------------------------------------------------------------------------
    collected_transformer_output_sequence_array = collected_transformer_output_sequence[batch_idx].squeeze(0).cpu().detach().numpy()
    # Diffs--------------------------------------------------------------------------------
    diff_transformer_output_and_optimized_latent_code_array = diff_transformer_outputs_and_optimized_latent_codes[batch_idx].squeeze(0).cpu().detach().numpy()
    diff_non_optimized_latent_codes_vs_optimized_latent_codes_fromDeepSDF_array = diff_non_optimized_latent_codes_vs_optimized_latent_codes_fromDeepSDF[batch_idx].squeeze(0).cpu().detach().numpy()
    # generate plots for everything--------------------------------
    collected_data_dict_for_plotting = {
        "True_gt_sdf": collected_sub_voxels_array,
        "diff_non_optimized_latent_codes_vs_optimized_latent_codes_fromDeepSDF": diff_non_optimized_latent_codes_vs_optimized_latent_codes_fromDeepSDF_array,
        "Optimized_latent_codes": collected_decoded_optimized_latent_codes_array,
        "Non_Optimized_latent_codes": collected_decoded_non_optimized_latent_codes_array,
        "Masked_non_optimized_latent_codes": collected_decoded_masked_non_optimized_latent_codes_array,
        "Transformer_output": collected_transformer_output_sequence_array,
        "Diff_transformer_output_&_optimized_latent_codes": diff_transformer_output_and_optimized_latent_code_array,
    }
    return collected_data_dict_for_plotting
# Eval--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def generate_data_for_eval(dict_arguments_for_vis: dict, dict_arguments_of_variables: dict, fdecoder) -> dict:
    non_optimized_latent_codes = dict_arguments_for_vis["non_optimized_latent_codes"]
    optimized_latent_codes = dict_arguments_for_vis["optimized_latent_codes"]
    masked_optimized_latent_codes = dict_arguments_for_vis["masked_optimized_latent_codes"]
    transformer_output_sequence_up = dict_arguments_for_vis["transformer_output_sequence_up"]
    sub_voxels = dict_arguments_for_vis["sub_voxels"]

    number_of_sub_voxels = dict_arguments_of_variables["number_of_sub_voxels"]
    latent_dim = dict_arguments_of_variables["latent_dim"]
    target_resolution = dict_arguments_of_variables["target_resolution"]
    resolution = dict_arguments_of_variables["resolution"]

    batch_size = sub_voxels.shape[0]
    # decoder--------------------------
    with torch.no_grad():
        # non-optimized latent-codes--------------------------------------------------------------------------------------------------------------------
        non_optimized_latent_codes_reshaped = non_optimized_latent_codes.reshape(batch_size, number_of_sub_voxels, latent_dim, 2, 2, 2)
        assert non_optimized_latent_codes_reshaped.shape == (batch_size, number_of_sub_voxels, latent_dim, 2, 2, 2)
        # Non-Optimized latent code collect------------------------------------------------------------------------------------------------------------
        # decoder--------------------------
        decoded_non_optimized_latent_codes = fdecoder(non_optimized_latent_codes_reshaped.clone()).to(device=fdecoder.device)  # [128, 512, 2, 2, 2] -> [128, 1, 32, 32, 32]
        decoded_non_optimized_latent_codes_reshaped = decoded_non_optimized_latent_codes.reshape([batch_size, number_of_sub_voxels, target_resolution, target_resolution, target_resolution])
        collected_decoded_non_optimized_latent_codes = collect_sub_voxels_to_voxel_with_batch(decoded_non_optimized_latent_codes_reshaped, resolution)
        assert collected_decoded_non_optimized_latent_codes.shape == (batch_size, resolution, resolution, resolution)
        del decoded_non_optimized_latent_codes
        del decoded_non_optimized_latent_codes_reshaped
        # optimized latent-codes--------------------------------------------------------------------------------------------------------------------
        decoded_optimized_latent_codes = fdecoder(optimized_latent_codes.clone()).to(device=fdecoder.device)  # [128, 512, 2, 2, 2] -> [128, 1, 32, 32, 32]
        decoded_optimized_latent_codes_reshaped = decoded_optimized_latent_codes.reshape([batch_size, number_of_sub_voxels, target_resolution, target_resolution, target_resolution])
        collected_decoded_optimized_latent_codes = collect_sub_voxels_to_voxel_with_batch(decoded_optimized_latent_codes_reshaped, resolution)
        assert collected_decoded_optimized_latent_codes.shape == (batch_size, resolution, resolution, resolution)
        del decoded_optimized_latent_codes
        del decoded_optimized_latent_codes_reshaped
        # Masked-non-latent-codes------------------------------------------------------------------------------------------------------------
        decoded_masked_optimized_latent_codes = fdecoder(masked_optimized_latent_codes.clone()).to(device=fdecoder.device)
        decoded_masked_optimized_latent_codes_reshaped = decoded_masked_optimized_latent_codes.reshape([batch_size, number_of_sub_voxels, target_resolution, target_resolution, target_resolution])
        collected_decoded_masked_optimized_latent_codes_reshaped = collect_sub_voxels_to_voxel_with_batch(decoded_masked_optimized_latent_codes_reshaped, resolution)
        assert collected_decoded_masked_optimized_latent_codes_reshaped.shape == (batch_size, resolution, resolution, resolution)
        del decoded_masked_optimized_latent_codes
        del decoded_masked_optimized_latent_codes_reshaped
        # Diff of non-optimized-latent_codes and optimized latent codes from DeepSDF-----------------------------------------------------------
        diff_non_optimized_latent_codes_vs_optimized_latent_codes_fromDeepSDF = torch.subtract(collected_decoded_optimized_latent_codes, collected_decoded_non_optimized_latent_codes)
        # Transformer output collect------------------------------------------------------------------------------------------------------------
        # TODO: Checkme, may be I am wrong!!!, no you are not!
        transformer_output_sequence_up_reshaped = (
            transformer_output_sequence_up.reshape(
                [
                    batch_size,
                    number_of_sub_voxels,
                    latent_dim,
                    2,
                    2,
                    2,
                ]
            )
            .to(device=fdecoder.device)
            .to(dtype=torch.float32)
        )
        decoded_transformer_output_sequence_up_reshaped = fdecoder(transformer_output_sequence_up_reshaped.clone()).to(device=fdecoder.device)
        decoded_transformer_output_sequence_up_reshaped_reshaped = decoded_transformer_output_sequence_up_reshaped.reshape(
            batch_size,
            number_of_sub_voxels,
            target_resolution,
            target_resolution,
            target_resolution,
        ).to(device=fdecoder.device)
        collected_transformer_output_sequence_up = collect_sub_voxels_to_voxel_with_batch(decoded_transformer_output_sequence_up_reshaped_reshaped, resolution)
        del decoded_transformer_output_sequence_up_reshaped
        del decoded_transformer_output_sequence_up_reshaped_reshaped
        # Diff--------------------------------------------------------------------------------
        diff_transformer_outputs_and_optimized_latent_codes = torch.subtract(collected_transformer_output_sequence_up, collected_decoded_optimized_latent_codes)
        # collect sub-voxels------------------------------------------------------------------------------------------------------------------
        collected_sub_voxels = collect_sub_voxels_to_voxel_with_batch(sub_voxels, resolution)
        data_dict_for_vis = {
            "collected_sub_voxels": collected_sub_voxels,
            "collected_decoded_non_optimized_latent_codes": collected_decoded_non_optimized_latent_codes,
            "collected_decoded_optimized_latent_codes": collected_decoded_optimized_latent_codes,
            "collected_decoded_masked_optimized_latent_codes": collected_decoded_masked_optimized_latent_codes_reshaped,
            "collected_transformer_output_sequence": collected_transformer_output_sequence_up,
            "diff_non_optimized_latent_codes_vs_optimized_latent_codes_fromDeepSDF": diff_non_optimized_latent_codes_vs_optimized_latent_codes_fromDeepSDF,
            "diff_transformer_outputs_and_optimized_latent_codes": diff_transformer_outputs_and_optimized_latent_codes,
        }
        return data_dict_for_vis

def generate_data_for_eval_orig_meshes(dict_arguments_for_vis: dict, dict_arguments_of_variables: dict, fdecoder) -> dict:
    non_optimized_latent_codes = dict_arguments_for_vis["non_optimized_latent_codes"]
    masked_non_optimized_latent_codes = dict_arguments_for_vis["masked_non_optimized_latent_codes"]
    transformer_output_sequence_up = dict_arguments_for_vis["transformer_output_sequence_up"]
    sub_voxels = dict_arguments_for_vis["sub_voxels"]

    number_of_sub_voxels = dict_arguments_of_variables["number_of_sub_voxels"]
    latent_dim = dict_arguments_of_variables["latent_dim"]
    target_resolution = dict_arguments_of_variables["target_resolution"]
    resolution = dict_arguments_of_variables["resolution"]

    batch_size = sub_voxels.shape[0]
    # decoder--------------------------
    with torch.no_grad():
        # non-optimized latent-codes--------------------------------------------------------------------------------------------------------------------
        non_optimized_latent_codes_reshaped = non_optimized_latent_codes.reshape(batch_size, number_of_sub_voxels, latent_dim, 2, 2, 2)
        assert non_optimized_latent_codes_reshaped.shape == (batch_size, number_of_sub_voxels, latent_dim, 2, 2, 2)
        # Non-Optimized latent code collect------------------------------------------------------------------------------------------------------------
        # decoder--------------------------
        decoded_non_optimized_latent_codes = fdecoder(non_optimized_latent_codes_reshaped.clone()).to(device=fdecoder.device)  # [128, 512, 2, 2, 2] -> [128, 1, 32, 32, 32]
        decoded_non_optimized_latent_codes_reshaped = decoded_non_optimized_latent_codes.reshape([batch_size, number_of_sub_voxels, target_resolution, target_resolution, target_resolution])
        collected_decoded_non_optimized_latent_codes = collect_sub_voxels_to_voxel_with_batch(decoded_non_optimized_latent_codes_reshaped, resolution)
        assert collected_decoded_non_optimized_latent_codes.shape == (batch_size, resolution, resolution, resolution)
        del decoded_non_optimized_latent_codes
        del decoded_non_optimized_latent_codes_reshaped

        # Masked-non-latent-codes------------------------------------------------------------------------------------------------------------
        decoded_masked_non_optimized_latent_codes = fdecoder(masked_non_optimized_latent_codes.clone()).to(device=fdecoder.device)
        decoded_masked_non_optimized_latent_codes_reshaped = decoded_masked_non_optimized_latent_codes.reshape([batch_size, number_of_sub_voxels, target_resolution, target_resolution, target_resolution])
        collected_decoded_masked_non_optimized_latent_codes_reshaped = collect_sub_voxels_to_voxel_with_batch(decoded_masked_non_optimized_latent_codes_reshaped, resolution)
        assert collected_decoded_masked_non_optimized_latent_codes_reshaped.shape == (batch_size, resolution, resolution, resolution)
        del decoded_masked_non_optimized_latent_codes
        del decoded_masked_non_optimized_latent_codes_reshaped
        # Transformer output collect------------------------------------------------------------------------------------------------------------
        decoded_transformer_output_sequence_up = fdecoder(transformer_output_sequence_up.clone()).to(device=fdecoder.device)
        decoded_transformer_output_sequence_up_reshaped = decoded_transformer_output_sequence_up.reshape(
            batch_size,
            number_of_sub_voxels,
            target_resolution,
            target_resolution,
            target_resolution,
        ).to(device=fdecoder.device)
        collected_transformer_output_sequence_up = collect_sub_voxels_to_voxel_with_batch(decoded_transformer_output_sequence_up_reshaped, resolution)
        del decoded_transformer_output_sequence_up_reshaped
        # Diff--------------------------------------------------------------------------------
        diff_transformer_outputs_and_non_optimized_latent_codes = torch.subtract(collected_transformer_output_sequence_up, collected_decoded_non_optimized_latent_codes)
        # collect sub-voxels------------------------------------------------------------------------------------------------------------------
        collected_sub_voxels = collect_sub_voxels_to_voxel_with_batch(sub_voxels, resolution)
        data_dict_for_vis = {
            "collected_sub_voxels": collected_sub_voxels,
            "collected_decoded_non_optimized_latent_codes": collected_decoded_non_optimized_latent_codes,
            "collected_decoded_masked_non_optimized_latent_codes": collected_decoded_masked_non_optimized_latent_codes_reshaped,
            "collected_transformer_output_sequence": collected_transformer_output_sequence_up,
            "diff_transformer_outputs_and_non_optimized_latent_codes": diff_transformer_outputs_and_non_optimized_latent_codes,
        }
        return data_dict_for_vis

def collect_generated_data_for_eval(data_dict_for_vis: dict, resolution: int, batch_idx: int) -> dict[str, Any]:
    collected_sub_voxels = data_dict_for_vis["collected_sub_voxels"]
    collected_decoded_non_optimized_latent_codes = data_dict_for_vis["collected_decoded_non_optimized_latent_codes"]
    collected_decoded_optimized_latent_codes = data_dict_for_vis["collected_decoded_optimized_latent_codes"]
    collected_decoded_masked_optimized_latent_codes = data_dict_for_vis["collected_decoded_masked_optimized_latent_codes"]
    collected_transformer_output_sequence = data_dict_for_vis["collected_transformer_output_sequence"]
    diff_non_optimized_latent_codes_vs_optimized_latent_codes_fromDeepSDF = data_dict_for_vis["diff_non_optimized_latent_codes_vs_optimized_latent_codes_fromDeepSDF"]
    diff_transformer_outputs_and_optimized_latent_codes = data_dict_for_vis["diff_transformer_outputs_and_optimized_latent_codes"]

    # plotting starts:------------------------------------------------------------------------------
    # True-GT march------------------------------------------------------------------------------------------------------------
    collected_sub_voxels_array = collected_sub_voxels[batch_idx].squeeze().cpu().detach().numpy()
    # # Non-Optimized-latent-codes march------------------------------------------------------------------------------------------------------------
    collected_decoded_non_optimized_latent_codes_array = collected_decoded_non_optimized_latent_codes[batch_idx].squeeze(0).cpu().detach().numpy()
    # Optimized-latent-codes march------------------------------------------------------------------------------------------------------------
    collected_decoded_optimized_latent_codes_array = collected_decoded_optimized_latent_codes[batch_idx].squeeze(0).cpu().detach().numpy()
    # Masked-non-optimized-latent-codes  march--------------------------------------------------------------------------------------------------------------
    collected_collected_decoded_masked_optimized_latent_codes_array = collected_decoded_masked_optimized_latent_codes[batch_idx].squeeze(0).cpu().detach().numpy()
    # Transformer-output march------------------------------------------------------------------------------------------------------------
    collected_transformer_output_sequence_array = collected_transformer_output_sequence[batch_idx].squeeze(0).cpu().detach().numpy()
    # Diffs--------------------------------------------------------------------------------
    diff_transformer_output_and_optimized_latent_code_array = diff_transformer_outputs_and_optimized_latent_codes[batch_idx].squeeze(0).cpu().detach().numpy()
    diff_non_optimized_latent_codes_vs_optimized_latent_codes_fromDeepSDF_array = diff_non_optimized_latent_codes_vs_optimized_latent_codes_fromDeepSDF[batch_idx].squeeze(0).cpu().detach().numpy()
    # generate plots for everything--------------------------------
    collected_data_dict_for_plotting = {
        "True_gt_sdf": collected_sub_voxels_array,
        "diff_non_optimized_latent_codes_vs_optimized_latent_codes_fromDeepSDF": diff_non_optimized_latent_codes_vs_optimized_latent_codes_fromDeepSDF_array,
        "Optimized_latent_codes": collected_decoded_optimized_latent_codes_array,
        "Non_Optimized_latent_codes": collected_decoded_non_optimized_latent_codes_array,
        "Masked_optimized_latent_codes": collected_collected_decoded_masked_optimized_latent_codes_array,
        "Transformer_output": collected_transformer_output_sequence_array,
        "Diff_transformer_output_&_optimized_latent_codes": diff_transformer_output_and_optimized_latent_code_array,
    }
    return collected_data_dict_for_plotting
def collect_generated_data_for_eval_orig_meshes(data_dict_for_vis: dict, resolution: int, batch_idx: int) -> dict[str, Any]:
    collected_sub_voxels = data_dict_for_vis["collected_sub_voxels"]
    collected_decoded_non_optimized_latent_codes = data_dict_for_vis["collected_decoded_non_optimized_latent_codes"]
    collected_decoded_masked_non_optimized_latent_codes = data_dict_for_vis["collected_decoded_masked_non_optimized_latent_codes"]
    collected_transformer_output_sequence = data_dict_for_vis["collected_transformer_output_sequence"]
    diff_transformer_outputs_and_non_optimized_latent_codes = data_dict_for_vis["diff_transformer_outputs_and_non_optimized_latent_codes"]

    # plotting starts:------------------------------------------------------------------------------
    # True-GT march------------------------------------------------------------------------------------------------------------
    collected_sub_voxels_array = collected_sub_voxels[batch_idx].squeeze().cpu().detach().numpy()
    # # Non-Optimized-latent-codes march------------------------------------------------------------------------------------------------------------
    collected_decoded_non_optimized_latent_codes_array = collected_decoded_non_optimized_latent_codes[batch_idx].squeeze(0).cpu().detach().numpy()
    # Masked-non-optimized-latent-codes  march--------------------------------------------------------------------------------------------------------------
    collected_collected_decoded_masked_non_optimized_latent_codes_array = collected_decoded_masked_non_optimized_latent_codes[batch_idx].squeeze(0).cpu().detach().numpy()
    # Transformer-output march------------------------------------------------------------------------------------------------------------
    collected_transformer_output_sequence_array = collected_transformer_output_sequence[batch_idx].squeeze(0).cpu().detach().numpy()
    # Diffs--------------------------------------------------------------------------------
    diff_transformer_output_and_non_optimized_latent_code_array = diff_transformer_outputs_and_non_optimized_latent_codes[batch_idx].squeeze(0).cpu().detach().numpy()
    # generate plots for everything--------------------------------
    collected_data_dict_for_plotting = {
        "True_gt_sdf": collected_sub_voxels_array,
        "Non_Optimized_latent_codes": collected_decoded_non_optimized_latent_codes_array,
        "Masked_non_optimized_latent_codes": collected_collected_decoded_masked_non_optimized_latent_codes_array,
        "Transformer_output": collected_transformer_output_sequence_array,
        "Diff_transformer_output_and_non_optimized_latent_codes": diff_transformer_output_and_non_optimized_latent_code_array,
    }
    return collected_data_dict_for_plotting


def collect_generated_data_for_eval(data_dict_for_vis: dict, resolution: int, batch_idx: int) -> dict[str, Any]:
    collected_sub_voxels = data_dict_for_vis["collected_sub_voxels"]
    collected_decoded_non_optimized_latent_codes = data_dict_for_vis["collected_decoded_non_optimized_latent_codes"]
    collected_decoded_optimized_latent_codes = data_dict_for_vis["collected_decoded_optimized_latent_codes"]
    collected_decoded_masked_optimized_latent_codes = data_dict_for_vis["collected_decoded_masked_optimized_latent_codes"]
    collected_transformer_output_sequence = data_dict_for_vis["collected_transformer_output_sequence"]
    diff_non_optimized_latent_codes_vs_optimized_latent_codes_fromDeepSDF = data_dict_for_vis["diff_non_optimized_latent_codes_vs_optimized_latent_codes_fromDeepSDF"]
    diff_transformer_outputs_and_optimized_latent_codes = data_dict_for_vis["diff_transformer_outputs_and_optimized_latent_codes"]

    # plotting starts:------------------------------------------------------------------------------
    # True-GT march------------------------------------------------------------------------------------------------------------
    collected_sub_voxels_array = collected_sub_voxels[batch_idx].squeeze().cpu().detach().numpy()
    # # Non-Optimized-latent-codes march------------------------------------------------------------------------------------------------------------
    collected_decoded_non_optimized_latent_codes_array = collected_decoded_non_optimized_latent_codes[batch_idx].squeeze(0).cpu().detach().numpy()
    # Optimized-latent-codes march------------------------------------------------------------------------------------------------------------
    collected_decoded_optimized_latent_codes_array = collected_decoded_optimized_latent_codes[batch_idx].squeeze(0).cpu().detach().numpy()
    # Masked-non-optimized-latent-codes  march--------------------------------------------------------------------------------------------------------------
    collected_collected_decoded_masked_optimized_latent_codes_array = collected_decoded_masked_optimized_latent_codes[batch_idx].squeeze(0).cpu().detach().numpy()
    # Transformer-output march------------------------------------------------------------------------------------------------------------
    collected_transformer_output_sequence_array = collected_transformer_output_sequence[batch_idx].squeeze(0).cpu().detach().numpy()
    # Diffs--------------------------------------------------------------------------------
    diff_transformer_output_and_optimized_latent_code_array = diff_transformer_outputs_and_optimized_latent_codes[batch_idx].squeeze(0).cpu().detach().numpy()
    diff_non_optimized_latent_codes_vs_optimized_latent_codes_fromDeepSDF_array = diff_non_optimized_latent_codes_vs_optimized_latent_codes_fromDeepSDF[batch_idx].squeeze(0).cpu().detach().numpy()
    # generate plots for everything--------------------------------
    collected_data_dict_for_plotting = {
        "True_gt_sdf": collected_sub_voxels_array,
        "diff_non_optimized_latent_codes_vs_optimized_latent_codes_fromDeepSDF": diff_non_optimized_latent_codes_vs_optimized_latent_codes_fromDeepSDF_array,
        "Optimized_latent_codes": collected_decoded_optimized_latent_codes_array,
        "Non_Optimized_latent_codes": collected_decoded_non_optimized_latent_codes_array,
        "Masked_optimized_latent_codes": collected_collected_decoded_masked_optimized_latent_codes_array,
        "Transformer_output": collected_transformer_output_sequence_array,
        "Diff_transformer_output_&_optimized_latent_codes": diff_transformer_output_and_optimized_latent_code_array,
    }
    return collected_data_dict_for_plotting