import torch


def make_mask3d_from_latent_code(batch_size: int, sql: int, prob: float) -> torch.bool:
    src_mask = torch.rand([batch_size, sql])
    src_mask_bool = src_mask < prob
    return src_mask_bool


def make_mask3d_from_voxel(
    batch_size: int, number_of_sub_voxels: int, prob: float
) -> torch.bool:
    src_mask = torch.rand([batch_size, number_of_sub_voxels])
    src_mask_bool = src_mask < prob
    return src_mask_bool
