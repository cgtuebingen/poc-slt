import os
import sys
sys.path.append("/home/zakeri/Documents/Codes/MyCodes/Proposal2/SDF_VAE/")
import torch
from tqdm import tqdm

# if __name__ == "__main__":
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# from Dataset.Dataset_Class_128fullmesh_ABC_with_NonOptimizedLatentCodes import ABCWITHNONOPTIMIZEDLATENTCODES training
from Transformer.Attention.TransformerExperiments.clean_code.clean_code_common_files import sub_voxel_related_fns as pp_fns
from Dataset.Dataset_Class_128fullmesh_ABC_with_NonOptimizedLatentCodes_Val import ABCWITHNONOPTIMIZEDLATENTCODESVAL # val

def main():
    obj_dir = "/graphics/scratch/datasets/ABC/obj/"
    lmdb_path = "/graphics/scratch2/staff/zakeri/LMDBs/ABC_128cube_5KLMDB_Test_cuda/_nonOptimizedLatentCodes/"
    dataset = ABCWITHNONOPTIMIZEDLATENTCODESVAL(obj_dir, lmdb_path, "",1, 128)

    empty_indices = []
    for batch_idx, batch in tqdm(enumerate(dataset)):
        (
            keys,
            object_indices,
            obj_file_names,
            gt_sdf_full_voxel,
            # std_copy,
            # var_copy,
            non_optimized_latent_codes,
        ) = batch

        object_indices = torch.as_tensor(object_indices).unsqueeze(0)
        gt_sdf_full_voxel = gt_sdf_full_voxel.unsqueeze(0)

        number_of_sub_voxels = 64
        target_resolution = 32

        batch_size = object_indices.shape[0]
        # -----------
        sub_voxels = pp_fns.sub_divide_gt_and_normalize(gt_sdf_full_voxel.clone(), number_of_sub_voxels, target_resolution)
        # generate empty and non-empty bool-----------------------------------------------------------------------------------------------
        empty_sub_voxels_bool, non_empty_sub_voxels_bool = pp_fns.extract_empty_and_non_empty_voxels(sub_voxels.clone(), number_of_sub_voxels, target_resolution)
        # empty_indices = []
        if not torch.all(torch.any(non_empty_sub_voxels_bool, dim=1)):
            pair = {"key": keys, "object_index": object_indices.item(), "obj_file_name": obj_file_names}
            # empty_indices.append(pair)
            print("\nall voxels are empty", pair)
            empty_indices.append(batch_idx)

    with open(lmdb_path + "empty_indices", 'w') as file:
        for i in empty_indices:
            file.write(str(i)+'\n')
    print("\n the empty indices are written!")
#
# if __name__ == "__main__":
#     main()
