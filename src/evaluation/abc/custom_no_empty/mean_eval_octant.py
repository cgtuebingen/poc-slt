import os
import torch
import sys

sys.path.append("/home/zakeri/Documents/Codes/MyCodes/Proposal2/SDF_VAE/")
# from Transformer.Attention.TransformerEval.evaluation_utils.mesh_excluding_fns import extract_original_meshes_file_name_from_mesh_file_name, extract_remeshed_meshes_file_name_from_mesh_file_name

from Transformer.Attention.TransformerEval.evaluation_analysis.common_file_ABC import (
    extract_files_with_given_extension_general,
    extract_broken_object_indices,
    extract_metric_results,
    pickle_all_dict_files,
    extract_non_broken_object_indices
)


if __name__ == "__main__":
    eval_dir = "/graphics/scratch2/staff/zakeri/train_logs/Transformer/flash_attention/with_optimized_latent_codes/full_dataset/overfitting/clean_code/regular_cat_fulldataset_alternative_test3_ABC_custom_noEmpty/lightning_logs/eval/ev0/front_bottom_right_octant/eval_dir/"

    eval_file_list = extract_files_with_given_extension_general(eval_dir, '.pkl')
    print("\n eval_file_list len: ", len(eval_file_list))

    broken_object_index_all = extract_broken_object_indices(eval_file_list, eval_dir)
    # read the data:
    broken_object_index_all_read = []
    broken_out_name = os.path.join(eval_dir, "broken_object_index_all.txt")
    with open(broken_out_name, "r") as file:
        for line in file:
            broken_object_index_all_read.append(int(line.rstrip("\n")))
    print()
    assert (broken_object_index_all == broken_object_index_all_read)
    extract_non_broken_object_indices(eval_file_list, eval_dir, broken_object_index_all)
    pickle_all_dict_files(eval_file_list, eval_dir)
    outname = os.path.join(eval_dir, "combined_pickles")
    combined_pickles = torch.load(outname)
    print()
    extract_metric_results(eval_dir, "front_bottom_right_eval_results", broken_object_index_all_read, combined_pickles)
    print()