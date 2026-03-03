import os
import sys
sys.path.append("/home/zakeri/Documents/Codes/MyCodes/Proposal2/SDF_VAE/")
import torch
from tqdm import tqdm

# if __name__ == "__main__":
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from Transformer.Attention.TransformerExperiments.clean_code.clean_code_common_files import sub_voxel_related_fns as pp_fns
from Dataset.Dataset_Class_128CubeFullmesh_ShapenetCorev1_ExcludingvalSplit_normalized_train_with_NonOptimizedLatentCodes import ShapeNetcorev1NormalizedTrainWithNonOptimizedLatentCodes  # training

def main():
    mesh_path = "/graphics/scratch2/staff/ruppert/scart/ShapeNetCorev2_remeshed_0.008/ShapeNetCore.v2/"
    # lmdb_path = "/graphics/scratch2/staff/zakeri/LMDBs/shapenetcorev2_SDF_SpanningMultiResVoxel32_128fullmesh_normalized_train/encoded_combined/"
    lmdb_path = "/graphics/scratch2/staff/zakeri/LMDBs/for_comparison_128shapenet_normalized/train/"
    points_to_sample = 1024
    query_number = 1000
    value_range = 1
    resolution = 128
    examples_per_epoch = 1000

    train_dataset = ShapeNetcorev1NormalizedTrainWithNonOptimizedLatentCodes(
        mesh_path,
        points_to_sample,
        query_number,
        lmdb_path,
        value_range,
        resolution,
        examples_per_epoch,
    )

    print(len(train_dataset))

    empty_indices = []
    for batch_idx, batch in tqdm(enumerate(train_dataset)):
        (
            object_indices,
            mesh_file_names,
            gt_sdf_voxel,
            std,
            var,
            non_optimized_latent_code,
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
            print("\nall voxels are empty", pair)
            empty_indices.append(batch_idx)

    with open(lmdb_path + "empty_indices", 'w') as file:
        for i in empty_indices:
            file.write(str(i)+'\n')


if __name__ == "__main__":
    main()
