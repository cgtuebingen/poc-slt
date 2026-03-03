from einops import rearrange
import torch
from Helpers.vae_specific_fns import sample_from_distribution, prepare_encoded_voxel_for_sampling
from typing import Tuple, Any
def prep_32cube_sub_voxels_and_encode(fencoder, sub_voxels: torch.Tensor, number_of_sub_voxels: int, latent_dim: int, target_resolution: int, train: bool) -> Tuple[torch.Tensor, Any, Any]:
    #  Prep for Encoding--------------------------------------------------------------------------------------------------------------------
    # TODO : I was the reason for fucking you. dimension broadcasting, nasty python
    batch_size = sub_voxels.shape[0]
    sub_voxels_reshaped = rearrange(sub_voxels.clone(), "A B C D E -> (A B) 1 C D E")
    # TODO : I was the reason for fucking you. dimension broadcasting, nasty python, checked
    assert sub_voxels_reshaped.shape == (batch_size * number_of_sub_voxels, 1, target_resolution, target_resolution, target_resolution)
    # encoder------------------------------------------------------------------------------------------------------------------------------
    with torch.no_grad():
        sub_voxels_encoded = fencoder(sub_voxels_reshaped)
    # del sub_voxels_reshaped
    sub_voxels_encoded_rearranged = prepare_encoded_voxel_for_sampling(number_of_sub_voxels, sub_voxels_encoded, latent_dim, batch_size=batch_size)
    # del sub_voxels_encoded
    # sampling------------------------------------------------------------------------------------------------------------------------------
    latent_code_sampled, std, var = sample_from_distribution(number_of_sub_voxels, sub_voxels_encoded_rearranged, batch_size, train)
    # del sub_voxels_encoded_rearranged
    latent_codes = latent_code_sampled.reshape([batch_size * number_of_sub_voxels, latent_dim, 2, 2, 2])  # [1024, 2, 2, 2] -> [512, 2, 2, 2]
    # del latent_code_sampled

    # Prepare for forward----------------------------------------------------------------------------------------------------------------
    latent_codes_reshaped = latent_codes.reshape(batch_size, number_of_sub_voxels, latent_dim, 2, 2, 2)
    assert latent_codes_reshaped.shape == (batch_size, number_of_sub_voxels, latent_dim, 2, 2, 2)  # input

    return (latent_codes_reshaped, std, var)
    # return latent_codes_reshaped

def prep_16cube_sub_voxels_and_encode(fencoder, sub_voxels: torch.Tensor, number_of_sub_voxels: int, latent_dim: int, target_resolution: int) -> Tuple[torch.Tensor, Any, Any]:
    #  Prep for Encoding--------------------------------------------------------------------------------------------------------------------
    # TODO : I was the reason for fucking you. dimension broadcasting, nasty python
    batch_size = sub_voxels.shape[0]
    sub_voxels_reshaped = rearrange(sub_voxels.clone(), "A B C D E -> (A B) 1 C D E")
    # TODO : I was the reason for fucking you. dimension broadcasting, nasty python, checked
    assert sub_voxels_reshaped.shape == (batch_size * number_of_sub_voxels, 1, target_resolution, target_resolution, target_resolution)
    # encoder------------------------------------------------------------------------------------------------------------------------------
    with torch.no_grad():
        sub_voxels_encoded = fencoder(sub_voxels_reshaped)
    # del sub_voxels_reshaped
    sub_voxels_encoded_rearranged = prepare_encoded_voxel_for_sampling(number_of_sub_voxels, sub_voxels_encoded, latent_dim, batch_size=batch_size)
    # del sub_voxels_encoded
    # sampling------------------------------------------------------------------------------------------------------------------------------
    mean_latent_code, std, var = sample_from_distribution(number_of_sub_voxels, sub_voxels_encoded_rearranged, batch_size=batch_size)
    # del sub_voxels_encoded_rearranged
    latent_codes = mean_latent_code.reshape([batch_size * number_of_sub_voxels, latent_dim, 1, 1, 1])  # [1024, 2, 2, 2] -> [512, 2, 2, 2]
    # del latent_code_sampled

    # Prepare for forward----------------------------------------------------------------------------------------------------------------
    latent_codes_reshaped = latent_codes.reshape(batch_size, number_of_sub_voxels, latent_dim, 1, 1, 1)
    assert latent_codes_reshaped.shape == (batch_size, number_of_sub_voxels, latent_dim, 1, 1, 1)  # input

    # return (latent_codes_reshaped)
    return (latent_codes_reshaped, std, var)

def prep_8cube_sub_voxels_and_encode(fencoder, sub_voxels: torch.Tensor, number_of_sub_voxels: int, latent_dim: int, target_resolution: int) -> Tuple[torch.Tensor, Any, Any]:
    #  Prep for Encoding--------------------------------------------------------------------------------------------------------------------
    # TODO : I was the reason for fucking you. dimension broadcasting, nasty python
    batch_size = sub_voxels.shape[0]
    sub_voxels_reshaped = rearrange(sub_voxels.clone(), "A B C D E -> (A B) 1 C D E")
    # TODO : I was the reason for fucking you. dimension broadcasting, nasty python, checked
    assert sub_voxels_reshaped.shape == (batch_size * number_of_sub_voxels, 1, target_resolution, target_resolution, target_resolution)
    # encoder------------------------------------------------------------------------------------------------------------------------------
    with torch.no_grad():
        sub_voxels_encoded = fencoder(sub_voxels_reshaped)
    # del sub_voxels_reshaped
    sub_voxels_encoded_rearranged = prepare_encoded_voxel_for_sampling(number_of_sub_voxels, sub_voxels_encoded, latent_dim, batch_size=batch_size)
    # del sub_voxels_encoded
    # sampling------------------------------------------------------------------------------------------------------------------------------
    latent_code_sampled , std, var = sample_from_distribution(number_of_sub_voxels, sub_voxels_encoded_rearranged, batch_size=batch_size)
    # del sub_voxels_encoded_rearranged
    latent_codes = latent_code_sampled.reshape([batch_size * number_of_sub_voxels, latent_dim, 1, 1, 1])  # [1024, 2, 2, 2] -> [512, 2, 2, 2]
    # del latent_code_sampled

    # Prepare for forward----------------------------------------------------------------------------------------------------------------
    latent_codes_reshaped = latent_codes.reshape(batch_size, number_of_sub_voxels, latent_dim, 1, 1, 1)
    assert latent_codes_reshaped.shape == (batch_size, number_of_sub_voxels, latent_dim, 1, 1, 1)  # input

    # return (latent_codes_reshaped, std, var)
    return (latent_codes_reshaped, std, var)
