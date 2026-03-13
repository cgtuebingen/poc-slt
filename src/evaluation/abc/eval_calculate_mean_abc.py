import os
import torch

from src.evaluation.abc.common_files_abc import (
    extract_files_with_given_extension_general,
    extract_broken_object_indices,
    extract_metric_results,
    pickle_all_dict_files,
    extract_non_broken_object_indices
)


if __name__ == "__main__":
    eval_dir = "/graphics/scratch2/staff/zakeri/tmp/pocslt_test/eval/ev0/bottom_half/eval_dir/"
    eval_file_list = extract_files_with_given_extension_general(eval_dir, '.pkl')
    print("\n eval_file_list len: ", len(eval_file_list))

    broken_object_index_all = extract_broken_object_indices(eval_file_list, eval_dir)
    # read the data:
    broken_object_index_all_read = []
    broken_out_name = os.path.join(eval_dir, "broken_object_index_all.txt")
    with open(broken_out_name, "r") as file:
        for line in file:
            broken_object_index_all_read.append(int(line.rstrip("\n")))
    assert (broken_object_index_all == broken_object_index_all_read)
    extract_non_broken_object_indices(eval_file_list, eval_dir, broken_object_index_all)
    pickle_all_dict_files(eval_file_list, eval_dir)
    outname = os.path.join(eval_dir, "combined_pickles")
    combined_pickles = torch.load(outname)
    extract_metric_results(eval_dir, "bottom_half_eval_results", broken_object_index_all_read, combined_pickles)
