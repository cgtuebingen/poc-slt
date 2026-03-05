import os
import torch
import sys

sys.path.append("/home/zakeri/Documents/Codes/MyCodes/Proposal2/SDF_VAE/")
# from Transformer.Attention.TransformerEval.evaluation_utils.mesh_excluding_fns import extract_original_meshes_file_name_from_mesh_file_name, extract_remeshed_meshes_file_name_from_mesh_file_name

from Transformer.Attention.TransformerEval.evaluation_analysis.common_files import (
    extract_files_with_given_extension_general,
    extract_broken_object_indices,
    extract_metric_results,
    extract_per_class_eval_results,
    pickle_all_dict_files,
    extract_non_broken_object_indices
)

if __name__ == "__main__":

    eval_dir = "/graphics/scratch2/staff/zakeri/train_logs/Transformer/flash_attention/with_optimized_latent_codes/full_dataset/overfitting/clean_code/regular_cat_fulldataset_alternative_test3_normalized_shapenet_noEmptymasking_custom/eval/ev0/MR0.25/eval_dir/"

    eval_file_list = extract_files_with_given_extension_general(eval_dir, '.pkl')

    # load chamfer remeshed_meshes vs original_meshes
    broken_meshes_dir = "/graphics/scratch2/staff/zakeri/shapenetcorev1_broken_meshes/shapenetcorev1_55_broken_meshes/remeshed_vs_orig_chamfer_all.pkl"
    remeshed_vs_orig_chamfer_all = torch.load(broken_meshes_dir)
    sortedobjID_remeshed_vs_orig_chamfer_all = sorted(remeshed_vs_orig_chamfer_all, key=lambda d: d["object_index"])

    broken_object_index_all = extract_broken_object_indices(eval_file_list, eval_dir, sortedobjID_remeshed_vs_orig_chamfer_all)
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
    extract_metric_results(eval_dir, "MR0.25_eval_results", broken_object_index_all_read, combined_pickles)
    print()
    extract_per_class_eval_results(eval_dir, "MR0.25_eval_results", broken_object_index_all_read, combined_pickles)
    print()
