import os
import torch
import sys
sys.path.append("....")
from src.evaluation.metrics_eval_helpers import extract_files_with_given_extension_general, write_dict_eval_into_text
from tqdm import tqdm
from src.evaluation.shapenet.shapenetcorev1_and_shapenetcorev2_info import synsteIds_categories_shapentcorev2


def extract_broken_object_indices(eval_file_list: list, eval_dir: str, sortedobjID_remeshed_vs_orig_chamfer_all: list):

    broken_object_index_all = []
    for i in tqdm(range(len(eval_file_list)), desc="Object Indices"):
        eval_file_current = eval_file_list[i]
        eval_dict_current = torch.load(eval_file_current)
        object_index_current = eval_dict_current["object_index"]# .detach() #.item()
        # first check if the remeshed_mesh_belong to this objID is broken so we exclude them
        chamfer_dist_remeshed_vs_orig = sortedobjID_remeshed_vs_orig_chamfer_all[object_index_current]["chamfer"]
        objcet_index_remeshed_vs_orig = sortedobjID_remeshed_vs_orig_chamfer_all[object_index_current]["object_index"]
        assert (object_index_current == objcet_index_remeshed_vs_orig)
        # 1- 0.0015359   # 2- 0.0014000 #3- 0.0013000 #4- 0.0012000 # 5-0.0011000 # 6- 0.001000  # 7-0.0009500  **# 8-0.000900**  # 9- 0.000850  # 10- 0.000800 # 11- 0.000750 # 12-0.000700 # 13- 0.000650 # 14- 0.000600 # 15- 0.000550 *# 16- 0.000500*
        # 17 - 0.000450  # 18- 0.000400  # 19- 0.000350
        if (chamfer_dist_remeshed_vs_orig > 0.000500):
            print("remeshed_mesh for this object id:" + str(object_index_current), " is broken!")
            broken_object_index_all.append(object_index_current)
            continue
        # now check for metrics
        iou_current = eval_dict_current['iou']
        if ((iou_current == -1) or (torch.isnan(torch.tensor(iou_current)))):
            print("iou nan")
            broken_object_index_all.append(object_index_current)

        fscore_current = eval_dict_current['fscore']
        if ((fscore_current == -1) or (torch.isnan(torch.tensor(fscore_current)))):
            print("fscore nan ")
            broken_object_index_all.append(object_index_current)

        uhd_current = eval_dict_current['uhd']
        if (uhd_current == -1 or (torch.isnan(torch.tensor(uhd_current)))):
            print("uhd -1")
            broken_object_index_all.append(object_index_current)

        fscore_one_percent = eval_dict_current['fscore1%']
        if ((fscore_one_percent == -1) or (torch.isnan(torch.tensor(fscore_one_percent)))):
            print("fscore_one_percent nan")
            broken_object_index_all.append(object_index_current)

        hausdorff_current = eval_dict_current['hausdorff']
        if (hausdorff_current == -1 or (torch.isnan(torch.tensor(hausdorff_current)))):
            print("hausdorff_current -1")
            broken_object_index_all.append(object_index_current)

        normal_consistency_current = eval_dict_current['normal_consistency']
        if (normal_consistency_current == -1 or (torch.isnan(torch.tensor(normal_consistency_current)))):
            print("normal_consistency_current -1")
            broken_object_index_all.append(object_index_current)

        inaccurate_normals_current = eval_dict_current['inaccurate_normals']
        if (inaccurate_normals_current == -1 or (torch.isnan(torch.tensor(inaccurate_normals_current)))):
            print("inaccurate_normals_current -1")
            broken_object_index_all.append(object_index_current)

        completeness_current = eval_dict_current['completeness']
        if (completeness_current == -1 or (torch.isnan(torch.tensor(completeness_current)))):
            print("completeness_current -1")
            broken_object_index_all.append(object_index_current)

        chamfer_current = eval_dict_current['chamfer']
        if (chamfer_current == -1 or (torch.isnan(torch.tensor(chamfer_current)))):
            print("chamfer_current -1")
            broken_object_index_all.append(object_index_current)

    print("\n num broken objects: ", len(broken_object_index_all))
    out_name = os.path.join(eval_dir, "broken_object_index_all.txt")
    with open(out_name, "w") as out:
        # Iterating over each element of the list
        for line in broken_object_index_all:
            out.write(str(line))  # Adding the line to the text.txt
            out.write('\n')  # Adding a new line character
    return broken_object_index_all

def extract_non_broken_object_indices(eval_file_list: list, eval_dir: str, broken_object_index_all: list):
    non_broken_object_index_all = []
    for i in tqdm(range(len(eval_file_list)), desc="Object Indices"):
        eval_file_current = eval_file_list[i]
        eval_dict_current = torch.load(eval_file_current)
        object_index_current = eval_dict_current["object_index"]# .detach().item()
        if (object_index_current in broken_object_index_all):
            continue
        else:
            non_broken_object_index_all.append(object_index_current)
            print("\n num non-broken objects: ", len(non_broken_object_index_all))
            out_name = os.path.join(eval_dir, "non_broken_object_index_all.txt")
            with open(out_name, "w") as out:
                # Iterating over each element of the list
                for line in non_broken_object_index_all:
                    out.write(str(line))  # Adding the line to the text.txt
                    out.write('\n')  # Adding a new line character

def pickle_all_dict_files(eval_file_list: list, eval_dir: str) -> dict:
    eval_file_list_dict = [torch.load(i) for i in eval_file_list]
    out_name = os.path.join(eval_dir, "combined_pickles")
    torch.save(eval_file_list_dict, out_name)

def extract_eval_results_for_given_class(eval_dir: str, output_name: str, class_label: str, broken_object_index_all: list, combined_pickles: dict):
    # given_class_label_all = []

    iou_all = []
    fscore_all = []
    uhd_all = []
    hausdorff_all = []
    normal_consistency_all = []
    inaccurate_normals_all = []
    completeness_all = []
    fscore_one_percent_all = []
    chamfer_all = []
    class_object_count = 0
    for i in tqdm(range(len(combined_pickles)), desc="EVAL RESULTS per " + class_label):
        eval_dict_current = combined_pickles[i]
        object_index_current = eval_dict_current["object_index"]# .detach().item()
        label_current = eval_dict_current['label']
        if object_index_current in broken_object_index_all:
            continue
        elif (label_current != class_label):
            continue
        else:
            iou_current = eval_dict_current['iou']
            if ((iou_current == -1) or (torch.isnan(torch.tensor(iou_current)))):
                raise ("iou nan")
            else:
                iou_all.append(iou_current)

            fscore_current = eval_dict_current['fscore']
            if ((fscore_current == -1) or (torch.isnan(torch.tensor(fscore_current)))):
                raise ("fscore nan ")
            else:
                fscore_all.append(fscore_current)

            uhd_current = eval_dict_current['uhd']
            if (uhd_current == -1):
                raise ("uhd -1")
            else:
                uhd_all.append(uhd_current)

            fscore_one_percent = eval_dict_current['fscore1%']
            if ((fscore_one_percent == -1) or (torch.isnan(torch.tensor(fscore_one_percent)))):
                raise ("fscore_one_percent nan")
            else:
                fscore_one_percent_all.append(fscore_one_percent)

            hausdorff_current = eval_dict_current['hausdorff']
            if (hausdorff_current == -1 or (torch.isnan(torch.tensor(hausdorff_current)))):
                raise ("hausdorff_current -1")
            else:
                hausdorff_all.append(hausdorff_current)

            normal_consistency_current = eval_dict_current['normal_consistency']
            if (normal_consistency_current == -1 or (torch.isnan(torch.tensor(normal_consistency_current)))):
                raise ("normal_consistency_current -1")
            else:
                normal_consistency_all.append(normal_consistency_current)

            inaccurate_normals_current = eval_dict_current['inaccurate_normals']
            if (inaccurate_normals_current == -1 or (torch.isnan(torch.tensor(inaccurate_normals_current)))):
                raise ("inaccurate_normals_current -1")
            else:
                inaccurate_normals_all.append(inaccurate_normals_current)

            completeness_current = eval_dict_current['completeness']
            if (completeness_current == -1 or (torch.isnan(torch.tensor(completeness_current)))):
                raise ("completeness_current -1")
            else:
                completeness_all.append(completeness_current)

            chamfer_current = eval_dict_current['chamfer']
            if (chamfer_current == -1 or (torch.isnan(torch.tensor(chamfer_current)))):
                raise ("chamfer_current -1")
            else:
                chamfer_all.append(chamfer_current)
        class_object_count += 1
    print("\n" + str(class_object_count) + "objects per " + class_label)
        # convert to tensor
    iou_t = torch.tensor(iou_all)
    fscore_t = torch.tensor(fscore_all)
    uhd_t = torch.tensor(uhd_all)
    hausdorff_t = torch.tensor(hausdorff_all)
    normal_consistency_t = torch.tensor(normal_consistency_all)
    inaccurate_normals_t = torch.tensor(inaccurate_normals_all)
    completeness_t = torch.tensor(completeness_all)
    fscore_one_percent_t = torch.tensor(fscore_one_percent_all)
    chamfer_t = torch.tensor(chamfer_all)
    # mean
    iou_mean = torch.mean(iou_t)
    fscore_mean = torch.mean(fscore_t)
    uhd_mean = torch.mean(uhd_t)
    hausdorff_mean = torch.mean(hausdorff_t)
    normal_consistency_mean = torch.mean(normal_consistency_t)
    inaccurate_normals_mean = torch.mean(inaccurate_normals_t)
    completeness_mean = torch.mean(completeness_t)
    fscore_one_percent_mean = torch.mean(fscore_one_percent_t)
    chamfer_mean = torch.mean(chamfer_t)
    # write them into one file
    result_dict = {"iou_mean": iou_mean,
                   "fscore_mean": fscore_mean,
                   "uhd_mean": uhd_mean,
                   "hausdorff_mean": hausdorff_mean,
                   "normal_consistency_mean": normal_consistency_mean,
                   "inaccurate_normals_mean": inaccurate_normals_mean,
                   "completeness_mean": completeness_mean,
                   "fscore_one_percent_mean": fscore_one_percent_mean,
                   "chamfer_mean": chamfer_mean}
    write_dict_eval_into_text(result_dict, eval_dir, output_name + "_for_" + str(class_object_count) + "-objects_" + class_label)
    print("\n Done writing eval results for " + class_label)


def extract_metric_results(eval_dir, output_name: str, broken_object_index_all: list, combined_pickles: dict):
    iou_all = []
    fscore_all = []
    uhd_all = []
    hausdorff_all = []
    normal_consistency_all = []
    inaccurate_normals_all = []
    completeness_all = []
    fscore_one_percent_all = []
    chamfer_all = []

    for i in tqdm(range(len(combined_pickles)), desc="EVAL RESULTS"):
        eval_dict_current = combined_pickles[i]
        object_index_current = eval_dict_current["object_index"]# .detach().item()
        if object_index_current in broken_object_index_all:
            continue
        else:
            iou_current = eval_dict_current['iou']
            if ((iou_current == -1) or (torch.isnan(torch.tensor(iou_current)))):
                raise("iou nan")
            else:
                iou_all.append(iou_current)

            fscore_current = eval_dict_current['fscore']
            if ((fscore_current == -1) or (torch.isnan(torch.tensor(fscore_current)))):
                raise("fscore nan ")
            else:
                fscore_all.append(fscore_current)

            uhd_current = eval_dict_current['uhd']
            if (uhd_current == -1  or (torch.isnan(torch.tensor(uhd_current)))):
                raise("uhd -1")
            else:
                uhd_all.append(uhd_current)

            fscore_one_percent = eval_dict_current['fscore1%']
            if ((fscore_one_percent == -1) or (torch.isnan(torch.tensor(fscore_one_percent)))):
                raise("fscore_one_percent nan")
            else:
                fscore_one_percent_all.append(fscore_one_percent)

            hausdorff_current = eval_dict_current['hausdorff']
            if (hausdorff_current == -1  or (torch.isnan(torch.tensor(hausdorff_current)))):
                raise("hausdorff_current -1")
            else:
                hausdorff_all.append(hausdorff_current)

            normal_consistency_current = eval_dict_current['normal_consistency']
            if (normal_consistency_current == -1 or (torch.isnan(torch.tensor(normal_consistency_current)))):
                raise("normal_consistency_current -1")
            else:
                normal_consistency_all.append(normal_consistency_current)

            inaccurate_normals_current = eval_dict_current['inaccurate_normals']
            if (inaccurate_normals_current == -1 or (torch.isnan(torch.tensor(inaccurate_normals_current)))):
                raise("inaccurate_normals_current -1")
            else:
                inaccurate_normals_all.append(inaccurate_normals_current)

            completeness_current = eval_dict_current['completeness']
            if (completeness_current == -1 or (torch.isnan(torch.tensor(completeness_current)))):
                raise("completeness_current -1")
            else:
                completeness_all.append(completeness_current)

            chamfer_current = eval_dict_current['chamfer']
            if (chamfer_current == -1 or (torch.isnan(torch.tensor(chamfer_current)))):
                raise("chamfer_current -1")
            else:
                chamfer_all.append(chamfer_current)

    # convert to tensor
    iou_t = torch.tensor(iou_all)
    fscore_t = torch.tensor(fscore_all)
    uhd_t = torch.tensor(uhd_all)
    hausdorff_t = torch.tensor(hausdorff_all)
    normal_consistency_t = torch.tensor(normal_consistency_all)
    inaccurate_normals_t = torch.tensor(inaccurate_normals_all)
    completeness_t = torch.tensor(completeness_all)
    fscore_one_percent_t = torch.tensor(fscore_one_percent_all)
    chamfer_t = torch.tensor(chamfer_all)
    # mean
    iou_mean = torch.mean(iou_t)
    fscore_mean = torch.mean(fscore_t)
    uhd_mean = torch.mean(uhd_t)
    hausdorff_mean = torch.mean(hausdorff_t)
    normal_consistency_mean = torch.mean(normal_consistency_t)
    inaccurate_normals_mean = torch.mean(inaccurate_normals_t)
    completeness_mean = torch.mean(completeness_t)
    fscore_one_percent_mean = torch.mean(fscore_one_percent_t)
    chamfer_mean = torch.mean(chamfer_t)
    # write them into one file
    result_dict = {"iou_mean": iou_mean,
                   "fscore_mean": fscore_mean,
                   "uhd_mean": uhd_mean,
                   "hausdorff_mean": hausdorff_mean,
                   "normal_consistency_mean": normal_consistency_mean,
                   "inaccurate_normals_mean": inaccurate_normals_mean,
                   "completeness_mean": completeness_mean,
                   "fscore_one_percent_mean": fscore_one_percent_mean,
                   "chamfer_mean": chamfer_mean}
    write_dict_eval_into_text(result_dict, eval_dir, output_name)
    print("\n Done writing eval results")

def extract_per_class_eval_results(eval_dir: str, output_name: str, broken_object_index_all, combined_pickles: dict):
    shapenetcorev2_classes = [key for key in synsteIds_categories_shapentcorev2.keys()]

    assert (len(shapenetcorev2_classes) == 55)

    for k in tqdm(range(len(shapenetcorev2_classes)), desc="Classes"):
        class_current = shapenetcorev2_classes[k]
        extract_eval_results_for_given_class(eval_dir, output_name, class_current, broken_object_index_all, combined_pickles)
        print()

def extract_files_with_given_extension_general(eval_dir: str, ext: str) -> list:
    all_files = os.listdir(eval_dir)
    print(all_files)
    file_path_list = []
    for i in range(len(all_files)):
        file = all_files[i]
        if file.endswith(ext):
            file_path_list.append(os.path.join(eval_dir, file))
    return file_path_list