from einops import rearrange
import torch
from typing import Tuple, Any
from src.p_vae import distribution


def prep_32cube_sub_voxels_and_encode(
    fencoder,
    sub_voxels: torch.Tensor,
    number_of_sub_voxels: int,
    latent_dim: int,
    target_resolution: int,
    train: bool,
) -> Tuple[torch.Tensor, Any, Any]:
    #  Prep for Encoding--------------------------------------------------------------------------------------------------------------------
    batch_size = sub_voxels.shape[0]
    sub_voxels_reshaped = rearrange(sub_voxels.clone(), "A B C D E -> (A B) 1 C D E")
    assert sub_voxels_reshaped.shape == (
        batch_size * number_of_sub_voxels,
        1,
        target_resolution,
        target_resolution,
        target_resolution,
    )
    # encoder------------------------------------------------------------------------------------------------------------------------------
    with torch.no_grad():
        sub_voxels_encoded = fencoder(sub_voxels_reshaped)
    # del sub_voxels_reshaped
    sub_voxels_encoded_rearranged = prepare_encoded_voxel_for_sampling(
        number_of_sub_voxels, sub_voxels_encoded, latent_dim, batch_size=batch_size
    )
    # sampling------------------------------------------------------------------------------------------------------------------------------
    latent_code_sampled, std, var = sample_from_distribution(
        number_of_sub_voxels, sub_voxels_encoded_rearranged, batch_size, train
    )
    # del sub_voxels_encoded_rearranged
    latent_codes = latent_code_sampled.reshape(
        [batch_size * number_of_sub_voxels, latent_dim, 2, 2, 2]
    )  # [1024, 2, 2, 2] -> [512, 2, 2, 2]

    # Prepare for forward----------------------------------------------------------------------------------------------------------------
    latent_codes_reshaped = latent_codes.reshape(
        batch_size, number_of_sub_voxels, latent_dim, 2, 2, 2
    )
    assert latent_codes_reshaped.shape == (
        batch_size,
        number_of_sub_voxels,
        latent_dim,
        2,
        2,
        2,
    )  # input

    return (latent_codes_reshaped, std, var)


def prepare_encoded_voxel_for_sampling(
    number_of_sub_voxels,
    sub_voxels_encoded: torch.Tensor,
    latent_dim: int,
    batch_size: int = 1,
) -> torch.Tensor:
    # Prep for Sampling-------------------------------------------------------------------------------------------------
    B_r, CH_r, D_r, H_r, W_r = sub_voxels_encoded.shape
    sub_voxels_encoded_reshaped = sub_voxels_encoded.reshape(
        [batch_size, number_of_sub_voxels, 2 * latent_dim, D_r, H_r, W_r]
    )
    B, SqL, Ch, D, H, W = sub_voxels_encoded_reshaped.shape
    sub_voxels_encoded_rearranged: torch.Tensor = rearrange(
        sub_voxels_encoded_reshaped, "B SqL Ch D H W -> (B SqL D H W) Ch"
    )
    return sub_voxels_encoded_rearranged


def sample_from_distribution(
    number_of_sub_voxels: int,
    sub_voxels_encoded_rearranged: torch.Tensor,
    batch_size: int,
    train: bool,
) -> Tuple[torch.Tensor, Any, Any]:
    # Sampling from Distribution----------------------------------------------------------------------------------------
    posterior = distribution.Distribution(sub_voxels_encoded_rearranged.clone())
    # for latent code we need the actual z which is mode
    if train:
        noisy_latent_code = posterior.sample()  # noisy
        std = posterior.std
        var = posterior.var
        # Reshaping latent code--------------------------------------------------------------------------------------------
        latent_code_reshaped = noisy_latent_code.reshape(
            [batch_size, number_of_sub_voxels, -1]
        )
    else:

        mean_latent_code = posterior.mode()  # mean
        std = posterior.std
        var = posterior.var
        # Reshaping latent code--------------------------------------------------------------------------------------------
        latent_code_reshaped = mean_latent_code.reshape(
            [batch_size, number_of_sub_voxels, -1]
        )

    return (latent_code_reshaped, std, var)
