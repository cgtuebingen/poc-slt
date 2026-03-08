import numpy
import numpy as np
from typing import Optional

import torch

def subvdivide_voxel(original_voxel: np.array, target_resolution: int) -> np.array:
    """
    :param original_voxel:
        ground truth sdf voxel of size [original_resolustion, original_resolustion, original_resolustion] :e.x[128, 128, 128]
    :param target_resolution:
        the size of sub-volume. the voxel size we want to divide original_voxel into.
    :return:
        sub_voxels :an array contains sub-volumes of original_voxel sith the size of [N, target_resolution, target_resolution, target_resolution]
        N is the number of sub-volumes with target_resolution exist in original_voxel
    """
    original_resolution, original_resolution, original_resolution = original_voxel.shape
    #
    # orig_tmp = (original_resolution ** 3)
    # target_tmp = (target_resolution ** 3)
    N = int(original_resolution ** 3 / target_resolution ** 3)
    # print("\n number of sub-voxels: ", N)
    sub_voxels = np.empty([N, target_resolution, target_resolution, target_resolution], device=torch.device('cuda'), dtype=torch.float32)

    t = 0
    for x in range(0, original_resolution, target_resolution):
        for y in range(0, original_resolution, target_resolution):
            for z in range(0, original_resolution, target_resolution):
                sub_voxels[t, :, :, :] = original_voxel[x: x + target_resolution, y: y + target_resolution, z: z + target_resolution]
                t += 1

    return sub_voxels

def subvdivide_voxel_with_batch(original_voxel: torch.Tensor, target_resolution: int) -> torch.Tensor:
    """
    :param original_voxel:
        ground truth sdf voxel of size [B, original_resolustion, original_resolustion, original_resolustion] :e.x[128, 128, 128]
    :param target_resolution:
        the size of sub-volume. the voxel size we want to divide original_voxel into.
    :return:
        sub_voxels :an array contains sub-volumes of original_voxel sith the size of [B, N, target_resolution, target_resolution, target_resolution]
        N is the number of sub-volumes with target_resolution exist in original_voxel
    """
    # FIXME, is it possible to implement this with reshape and not messh it up
    B, original_resolution, original_resolution, original_resolution = original_voxel.shape

    N = original_resolution // target_resolution
    N = N * N * N
    # print("\n number of sub-voxels: ", N)
    sub_voxels = torch.empty((B, N, target_resolution, target_resolution, target_resolution), device=original_voxel.device, dtype=original_voxel.dtype)

    for b in range(B):
        t = 0
        for x in range(0, original_resolution, target_resolution):
            for y in range(0, original_resolution, target_resolution):
                for z in range(0, original_resolution, target_resolution):
                    #assert(t == z//target_resolution + y//target_resolution * original_resolution//target_resolution + x//target_resolution * original_resolution//target_resolution * original_resolution//target_resolution)
                    sub_voxels[b, t, :, :, :] = original_voxel[b, x: x + target_resolution, y: y + target_resolution, z: z + target_resolution]
                    t += 1
    return sub_voxels

def collect_sub_voxels_to_voxel_with_batch(sub_voxels: torch.Tensor, original_resolution: int) -> torch.Tensor:
    """
    :param original_voxel:
    :param target_resolution:
    :return:
    """
    # FIXME, is it possible to implement this with reshape and not messh it up
    B, N, target_resolution, target_resolution, target_resolution = sub_voxels.shape

    original_voxel = torch.empty((B, original_resolution, original_resolution, original_resolution), device=sub_voxels.device, dtype=sub_voxels.dtype)

    for b in range(B):
        for t in range(0, N):
            # t = z + y * target_resolution + x * target_resolution * target_resolution
            z = target_resolution * (t % (original_resolution//target_resolution))
            y = target_resolution * ((t // (original_resolution//target_resolution)) % (original_resolution//target_resolution))
            x = target_resolution * (t // ((original_resolution//target_resolution) * (original_resolution//target_resolution)))
            #assert (t == z // target_resolution + y // target_resolution * original_resolution // target_resolution + x // target_resolution * original_resolution // target_resolution * original_resolution // target_resolution)
            original_voxel[b, x: x + target_resolution, y: y + target_resolution, z: z + target_resolution] = sub_voxels[b, t, :, :, :]
    return original_voxel

def extract_surface_contained_voxels(sub_voxels: np.array) -> np.array:
    """
    :param sub_voxels:
        sub_voxels :an array contains sub-volumes of original_voxel sith the size of [ N, X, X, X]
        N is the number of sub-volumes with target_resolution exist in original_voxel
    :return:
        surface_contained_sub_voxels: as the name says. The size is [B_prime, X, X, X]
        B_prime is a number less than or equal with B*N. Usually less because there are sub_voxels that do not contain any surface and they get excluded.

    """

    signs = np.signbit(sub_voxels)
    neg = np.any(signs, axis=(1, 2, 3))
    pos = np.logical_not(np.all(signs, axis=(1, 2, 3)))

    return sub_voxels[np.logical_and(pos, neg)]

def extract_empty_sub_voxel_indices_from_voxel(sub_voxels: torch.bool) -> torch.bool:
    """
    :param sub_voxels:
        sub_voxels :an array contains sub-volumes of original_voxel sith the size of [B, N, target_resolution, target_resolution, target_resolution]
        N is the number of sub-volumes with target_resolution exist in original_voxel
    :return:
        The indices belong to empty sub-voxel. The size is [N_prime,2]
        N_prime is a number of empty voxels and 2 belong to the Batch index and N index. basically this boolean array indicates the B index and the N index where the sub-voxel is empty.

    """
    # B, N, target_resolution, target_resolution, target_resolution = sub_voxels.shape
    # print("\n sub_voxels shape: ", sub_voxels.shape)
    signs = torch.signbit(sub_voxels)
    # print("\n signs shape: ", signs.shape)

    neg = torch.any(torch.flatten(signs, 2), dim=2)

    pos = torch.logical_not(torch.all(torch.flatten(signs, 2), dim=2))

    empty_indices = (torch.logical_not(torch.logical_and(pos, neg)))
    # print("\n empty_indices shape: ", empty_indices.shape)
    return empty_indices

def extract_outside_sub_voxel_indices_from_voxel(sub_voxels: torch.bool) -> torch.bool:
    """
    :param sub_voxels:
        sub_voxels :an array contains sub-volumes of original_voxel sith the size of [B, N, target_resolution, target_resolution, target_resolution]
        N is the number of sub-volumes with target_resolution exist in original_voxel
    :return:
        The indices belong to empty sub-voxel. The size is [N_prime,2]
        N_prime is a number of empty voxels and 2 belong to the Batch index and N index. basically this boolean array indicates the B index and the N index where the sub-voxel is empty.

    """
    # B, N, target_resolution, target_resolution, target_resolution = sub_voxels.shape
    # print("\n sub_voxels shape: ", sub_voxels.shape)
    signs = torch.signbit(sub_voxels)
    # print("\n signs shape: ", signs.shape)

    # neg = torch.any(torch.flatten(signs, 2), dim=2)

    pos = torch.logical_not(torch.all(torch.flatten(signs, 2), dim=2))

    empty_indices = (torch.logical_not(pos))
    # print("\n empty_indices shape: ", empty_indices.shape)
    return empty_indices

#-----------------------------------------------------------------------------
    # N, target_resolution, target_resolution, target_resolution = sub_voxels.shape
    # sub_voxels_reshaped = np.reshape(sub_voxels, [N, target_resolution, target_resolution, target_resolution])
    # print("\n sub_voxels_reshaped:", sub_voxels_reshaped.shape)
    # surface_contained_sub_voxels = []
    # for v in range(sub_voxels_reshaped.shape[0]):
    #     signs = np.signbit(sub_voxels_reshaped[v, :, :, :])
    #
    #     pos = np.any(signs)  # any bit set ==> at least one negative number
    #     neg = np.array(not(np.all(signs)))  # not all bits set ==> not all numbers are negative
    #     if ((pos) and (neg)):
    #         surface_contained_sub_voxels.append(sub_voxels_reshaped[v, :, :, :])
    #
    # if (not(len(surface_contained_sub_voxels))):
    #     print("\n the list is empty so the gt_sdf was either totally inside or totally outside!")
    #     return None
    #
    # surface_contained_sub_voxels_tensor = np.stack(surface_contained_sub_voxels)
    # # print("\n surface_contained_sub_voxels_tensor:", surface_contained_sub_voxels_tensor.shape)
    # return surface_contained_sub_voxels_tensor

# surface_contained_sub_voxels_tensor = extract_surface_contained_voxels(sub_voxels)
# print("\n sub_voxels_reshaped:", len(surface_contained_sub_voxels_tensor))
# print("\n sub_voxels_reshaped:", surface_contained_sub_voxels_tensor.shape)
#
# print("\n ")

