import torch
# validation data loader helper
def fetch_latent_codes_from_32cubelmdb(dict_for_latent_codes: dict, lmdb_optimized_latent_codes_obj, number_of_sub_voxels: int, latent_dim: int, val_batch_size: int) -> torch.Tensor:
    optimized_latent_codes = torch.zeros([val_batch_size, number_of_sub_voxels, latent_dim, 2, 2, 2], dtype=torch.float32, device='cuda')

    object_indices = dict_for_latent_codes["object_indices"]
    mesh_file_names = dict_for_latent_codes["mesh_file_names"]
    for b in range(val_batch_size):
        object_index = object_indices[b].item()

        key, object_index_lmdb, mesh_file_name_lmdb, optimized_latent_code = lmdb_optimized_latent_codes_obj[object_index]
        optimized_latent_code = optimized_latent_code.to(device='cuda')
        assert object_index == object_index_lmdb  # since we chose a few selected meshes then they are not going to be the same
        assert mesh_file_names[b] == mesh_file_name_lmdb
        optimized_latent_codes[b] = optimized_latent_code

    del optimized_latent_code
    return optimized_latent_codes


def fetch_latent_codes_from_16cubelmdb(dict_for_latent_codes: dict, lmdb_optimized_latent_codes_obj, number_of_sub_voxels: int, latent_dim: int, val_batch_size: int) -> torch.Tensor:
    optimized_latent_codes = torch.zeros([val_batch_size, number_of_sub_voxels, latent_dim, 1, 1, 1], dtype=torch.float32, device='cuda')

    object_indices = dict_for_latent_codes["object_indices"]
    mesh_file_names = dict_for_latent_codes["mesh_file_names"]
    for b in range(val_batch_size):
        object_index = object_indices[b].item()

        key, object_index_lmdb, mesh_file_name_lmdb, optimized_latent_code = lmdb_optimized_latent_codes_obj[object_index]
        optimized_latent_code = optimized_latent_code.to(device='cuda')
        assert object_index == object_index_lmdb  # since we chose a few selected meshes then they are not going to be the same
        assert mesh_file_names[b] == mesh_file_name_lmdb
        optimized_latent_codes[b] = optimized_latent_code

    del optimized_latent_code
    return optimized_latent_codes
