import torch


def normalize_voxels_to_their_distance_with_batch(
    voxels: torch.Tensor,
    distances_per_voxel: torch.Tensor,
    batch_size: int,
    target_resolution: int,
) -> torch.Tensor:
    assert (voxels.shape) == (
        batch_size,
        target_resolution,
        target_resolution,
        target_resolution,
    )
    assert (distances_per_voxel.shape) == (batch_size, 1)
    normalized_voxels = torch.zeros(
        voxels.shape, dtype=torch.float32, device=torch.device("cuda")
    )

    for b in range(batch_size):
        voxel = voxels[b, :, :, :].squeeze()
        d = distances_per_voxel[b].squeeze()
        d = d.to(torch.float32)
        normalized_voxels[b, :, :, :] = voxel / d

    return normalized_voxels


def normalize_voxels_to_their_distance(
    voxel: torch.Tensor, distances_per_voxel: torch.Tensor, target_resolution: int
) -> torch.Tensor:
    assert (voxel.shape) == (target_resolution, target_resolution, target_resolution)
    assert (distances_per_voxel.shape) == (1)

    normalized_voxel = voxel / distances_per_voxel
    if torch.min(normalized_voxel) < -torch.sqrt(torch.as_tensor(3)) or torch.max(
        normalized_voxel
    ) > torch.sqrt(torch.as_tensor(3)):
        raise "normalization is wrong"
    return normalized_voxel
