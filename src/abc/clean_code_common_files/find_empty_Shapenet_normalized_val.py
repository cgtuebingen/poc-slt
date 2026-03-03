import os
import sys
from typing import Tuple, Any, Union

sys.path.append("//")
import torch
from tqdm import tqdm

# if __name__ == "__main__":
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from Transformer.Attention.TransformerExperiments.clean_code import sub_voxel_related_fns as pp_fns
from Dataset.Dataset_Class_128CubeFullmesh_ShapenetCorev1_ExcludingvalSplit_normalized_val_with_Non_andOptimized__LatentCodes import ShapeNetcorev1NormalizedValWith_Non_and_Optimized_LatentCodes  # val

def main():
    mesh_path = "/graphics/scratch2/staff/zakeri/LMDBs/ShapeNetCorev2_remeshed_0.008/ShapeNetCore.v2/"
    lmdb_path = "/graphics/scratch2/staff/zakeri/LMDBs/shapenetcorev2_SDF_SpanningMultiResVoxel32_128fullmesh_normalized_val/encoded_combined/_with_NonOptimizedLatentCodes_new"

    optimized_latent_code_lmdb_path = "/graphics/scratch2/staff/zakeri/LMDBs/shapenetcorev2_SDF_SpanningMultiResVoxel32_128fullmesh_normalized_val/optimized_latent_code/combined"

    value_range = 1
    resolution = 128
    # target_resolution = 32

    points_to_sample = 1024
    query_number = 1000
    examples_per_epoch: int = 1000

    val_dataset = ShapeNetcorev1NormalizedValWith_Non_and_Optimized_LatentCodes(
        mesh_path,
        points_to_sample,
        query_number,
        lmdb_path,
        optimized_latent_code_lmdb_path,
        value_range,
        resolution,
        examples_per_epoch,
    )

    print(len(val_dataset))

    empty_indices = []
    for batch_idx, batch in tqdm(enumerate(val_dataset)):
        (
            object_indices,
            mesh_file_names,
            non_optimized_latent_code,
            gt_sdf_voxel,
            optimized_latent_code_copy,
            folder_name,
            sub_folder_name,
        ) = batch

        object_indices = torch.as_tensor(object_indices).unsqueeze(0)
        gt_sdf_voxel = gt_sdf_voxel.unsqueeze(0)

        number_of_sub_voxels = 64
        target_resolution = 32

        batch_size = object_indices.shape[0]
        # -----------
        sub_voxels = pp_fns.sub_divide_gt_and_normalize(gt_sdf_voxel.clone(), number_of_sub_voxels, target_resolution)
        # generate empty and non-empty bool-----------------------------------------------------------------------------------------------
        empty_sub_voxels_bool, non_empty_sub_voxels_bool = pp_fns.extract_empty_and_non_empty_voxels(sub_voxels.clone(), number_of_sub_voxels, target_resolution)
        # empty_indices = []
        if not torch.all(torch.any(non_empty_sub_voxels_bool, dim=1)):
            pair = {"object_index": object_indices.item(), "mesh_file_names": mesh_file_names}
            # empty_indices.append(pair)
            print("\nall voxels are empty", pair)
            empty_indices.append(batch_idx)

    with open(lmdb_path + "empty_indices", 'w') as file:
        for i in empty_indices:
            file.write(str(i)+'\n')

#
# if __name__ == "__main__":
#     main()
