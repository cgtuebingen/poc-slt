from src.utils.metrics_eval_helpers import write_dict_eval_into_text
from tqdm import tqdm


def extract_files_with_given_extension_general(eval_dir: str, ext: str) -> list:
    all_files = os.listdir(eval_dir)
    print(all_files)
    file_path_list = []
    for i in range(len(all_files)):
        file = all_files[i]
        if file.endswith(ext):
            file_path_list.append(os.path.join(eval_dir, file))
    return file_path_list


def extract_broken_object_indices(eval_file_list: list, eval_dir: str):

    broken_object_index_all = []
    for i in tqdm(range(len(eval_file_list)), desc="Object Indices"):
        eval_file_current = eval_file_list[i]
        eval_dict_current = torch.load(eval_file_current)
        object_index_current = eval_dict_current["object_index"].detach().item()
        # now check for metrics
        iou_current = eval_dict_current["iou"]
        if (iou_current == -1) or (torch.isnan(torch.tensor(iou_current))):
            print("iou nan")
            broken_object_index_all.append(object_index_current)

        fscore_current = eval_dict_current["fscore"]
        if (fscore_current == -1) or (torch.isnan(torch.tensor(fscore_current))):
            print("fscore nan ")
            broken_object_index_all.append(object_index_current)

        uhd_current = eval_dict_current["uhd"]
        if uhd_current == -1 or (torch.isnan(torch.tensor(uhd_current))):
            print("uhd -1")
            broken_object_index_all.append(object_index_current)

        fscore_one_percent = eval_dict_current["fscore1%"]
        if (fscore_one_percent == -1) or (
            torch.isnan(torch.tensor(fscore_one_percent))
        ):
            print("fscore_one_percent nan")
            broken_object_index_all.append(object_index_current)

        hausdorff_current = eval_dict_current["hausdorff"]
        if hausdorff_current == -1 or (torch.isnan(torch.tensor(hausdorff_current))):
            print("hausdorff_current -1")
            broken_object_index_all.append(object_index_current)

        normal_consistency_current = eval_dict_current["normal_consistency"]
        if normal_consistency_current == -1 or (
            torch.isnan(torch.tensor(normal_consistency_current))
        ):
            print("normal_consistency_current -1")
            broken_object_index_all.append(object_index_current)

        inaccurate_normals_current = eval_dict_current["inaccurate_normals"]
        if inaccurate_normals_current == -1 or (
            torch.isnan(torch.tensor(inaccurate_normals_current))
        ):
            print("inaccurate_normals_current -1")
            broken_object_index_all.append(object_index_current)

        completeness_current = eval_dict_current["completeness"]
        if completeness_current == -1 or (
            torch.isnan(torch.tensor(completeness_current))
        ):
            print("completeness_current -1")
            broken_object_index_all.append(object_index_current)

        chamfer_current = eval_dict_current["chamfer"]
        if chamfer_current == -1 or (torch.isnan(torch.tensor(chamfer_current))):
            print("chamfer_current -1")
            broken_object_index_all.append(object_index_current)

    print("\n num broken objects: ", len(broken_object_index_all))
    out_name = os.path.join(eval_dir, "broken_object_index_all.txt")
    with open(out_name, "w") as out:
        for line in broken_object_index_all:
            out.write(str(line))
            out.write("\n")
    return broken_object_index_all


def extract_non_broken_object_indices(
    eval_file_list: list, eval_dir: str, broken_object_index_all: list
):
    non_broken_object_index_all = []
    for i in tqdm(range(len(eval_file_list)), desc="Object Indices"):
        eval_file_current = eval_file_list[i]
        eval_dict_current = torch.load(eval_file_current)
        object_index_current = eval_dict_current["object_index"].detach().item()
        if object_index_current in broken_object_index_all:
            continue
        else:
            non_broken_object_index_all.append(object_index_current)
            print("\n num non-broken objects: ", len(non_broken_object_index_all))
            out_name = os.path.join(eval_dir, "non_broken_object_index_all.txt")
            with open(out_name, "w") as out:
                for line in non_broken_object_index_all:
                    out.write(str(line))
                    out.write("\n")


def pickle_all_dict_files(eval_file_list: list, eval_dir: str) -> dict:
    eval_file_list_dict = [torch.load(i) for i in eval_file_list]
    out_name = os.path.join(eval_dir, "combined_pickles")
    torch.save(eval_file_list_dict, out_name)


def extract_metric_results(
    eval_dir, output_name: str, broken_object_index_all: list, combined_pickles: dict
):
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
        object_index_current = eval_dict_current["object_index"].detach().item()
        if object_index_current in broken_object_index_all:
            continue
        else:
            iou_current = eval_dict_current["iou"]
            if (iou_current == -1) or (torch.isnan(torch.tensor(iou_current))):
                print("iou nan")
            else:
                iou_all.append(iou_current)

            fscore_current = eval_dict_current["fscore"]
            if (fscore_current == -1) or (torch.isnan(torch.tensor(fscore_current))):
                print("fscore nan ")
            else:
                fscore_all.append(fscore_current)

            uhd_current = eval_dict_current["uhd"]
            if uhd_current == -1 or (torch.isnan(torch.tensor(uhd_current))):
                print("uhd -1")
            else:
                uhd_all.append(uhd_current)

            fscore_one_percent = eval_dict_current["fscore1%"]
            if (fscore_one_percent == -1) or (
                torch.isnan(torch.tensor(fscore_one_percent))
            ):
                print("fscore_one_percent nan")
            else:
                fscore_one_percent_all.append(fscore_one_percent)

            hausdorff_current = eval_dict_current["hausdorff"]
            if hausdorff_current == -1 or (
                torch.isnan(torch.tensor(hausdorff_current))
            ):
                print("hausdorff_current -1")
            else:
                hausdorff_all.append(hausdorff_current)

            normal_consistency_current = eval_dict_current["normal_consistency"]
            if normal_consistency_current == -1 or (
                torch.isnan(torch.tensor(normal_consistency_current))
            ):
                print("normal_consistency_current -1")
            else:
                normal_consistency_all.append(normal_consistency_current)

            inaccurate_normals_current = eval_dict_current["inaccurate_normals"]
            if inaccurate_normals_current == -1 or (
                torch.isnan(torch.tensor(inaccurate_normals_current))
            ):
                print("inaccurate_normals_current -1")
            else:
                inaccurate_normals_all.append(inaccurate_normals_current)

            completeness_current = eval_dict_current["completeness"]
            if completeness_current == -1 or (
                torch.isnan(torch.tensor(completeness_current))
            ):
                print("completeness_current -1")
            else:
                completeness_all.append(completeness_current)

            chamfer_current = eval_dict_current["chamfer"]
            if chamfer_current == -1 or (torch.isnan(torch.tensor(chamfer_current))):
                print("chamfer_current -1")
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
    result_dict = {
        "iou_mean": iou_mean,
        "fscore_mean": fscore_mean,
        "uhd_mean": uhd_mean,
        "hausdorff_mean": hausdorff_mean,
        "normal_consistency_mean": normal_consistency_mean,
        "inaccurate_normals_mean": inaccurate_normals_mean,
        "completeness_mean": completeness_mean,
        "fscore_one_percent_mean": fscore_one_percent_mean,
        "chamfer_mean": chamfer_mean,
    }
    write_dict_eval_into_text(result_dict, eval_dir, output_name)
    print("\n Done writing eval results")


import os
import torch
from typing import Tuple
from src.pycu3d.test.evaluate_all import EVALALLMETRICS
from src.utils import plot_march_fns as pmt_fns
from src.utils.m_cube_fns import make_mcubes_from_voxels_obj_for_pad


def evaluate(
    dict_arguments_for_eval: dict,
    dict_arguments_of_variables: dict,
    collected_data_dict_for_plotting: dict,
) -> dict:
    object_index = dict_arguments_of_variables["object_index"]
    obj_file_name = dict_arguments_of_variables["obj_file_name"]
    num_samples = dict_arguments_of_variables["num_samples"]

    hausdorff_scale = dict_arguments_of_variables["hausdorff_scale"]
    chamfer_scale = dict_arguments_of_variables["chamfer_scale"]

    completed_voxel = collected_data_dict_for_plotting[
        "transformer_output_sequence_up_collected_32cubes"
    ]
    gt_voxel = collected_data_dict_for_plotting["collected_sub_voxels"]

    partial_obj = dict_arguments_for_eval[
        "Masked_non_optimized_latent_codes_file_name_obj"
    ]
    completed_obj = dict_arguments_for_eval["Transformer_output_file_name_obj"]
    gt_obj = dict_arguments_for_eval["True_gt_sdf_file_name_obj"]

    # on voxels
    eval_obj = EVALALLMETRICS()
    iou, fscore = eval_obj.eval_iou_and_fscore_voxels_cud3d(completed_voxel, gt_voxel)
    if torch.isnan(torch.tensor(iou)) or torch.isnan(torch.tensor(fscore)):
        iou = -1
        fscore = -1

    completed_pc = eval_obj.get_obj_return_pc(completed_obj, num_samples)
    gt_pc = eval_obj.get_obj_return_pc(gt_obj, num_samples)
    partial_pc = eval_obj.get_obj_return_pc(partial_obj, num_samples)
    if (completed_pc == None) or (gt_pc == None) or (partial_pc == None):
        chamfer = -1
        fscore_one_percent = -1
        uhd = -1
        hausdorff = -1
        normal_consistency = -1
        inaccurate_normals = -1
        completeness = -1
    else:
        chamfer = eval_obj.eval_chamfer(completed_pc, gt_pc)
        chamfer = chamfer * chamfer_scale

        fscore_one_percent = eval_obj.eval_fscore_pc_cud3d(
            completed_pc, gt_pc, thres=0.02
        )

        uhd = eval_obj.eval_uhd(partial_pc, completed_pc)
        uhd = uhd * hausdorff_scale

        hausdorff = eval_obj.eval_hausdorff(completed_pc, gt_pc)
        hausdorff = hausdorff * hausdorff_scale

        normal_consistency, inaccurate_normals = eval_obj.eval_nc_and_in(
            completed_pc, gt_pc
        )

        completeness = eval_obj.eval_completeness(completed_pc, gt_pc, thres=0.03)

    # pack all the eval results together
    eval_results = {
        "iou": iou,
        "fscore": fscore,
        "chamfer": chamfer,
        "fscore1%": fscore_one_percent,
        "uhd": uhd,
        "hausdorff": hausdorff,
        "normal_consistency": normal_consistency,
        "inaccurate_normals": inaccurate_normals,
        "completeness": completeness,
        "object_index": object_index,
        "obj_file_name": obj_file_name,
    }
    return eval_results


def write_evaluation_result(eval_results: dict, eval_dir: str):
    object_index = eval_results["object_index"]

    pickle_name = os.path.join(
        eval_dir, "eval_results_for_" + "objID=" + str(object_index.item()) + ".pkl"
    )
    if os.path.isdir(eval_dir):
        torch.save(
            eval_results,
            pickle_name,
        )
    else:
        raise print(eval_dir + "does not exist!")

    # # test
    eval_results_read = torch.load(pickle_name)
    assert eval_results_read["object_index"] == eval_results["object_index"]


def march_gt_and_mask_only_and_write_objs(
    dict_arguments_for_vis: dict,
    dict_arguments_of_variables: dict,
    object_indices: torch.Tensor,
    fdecoder,
) -> None:
    common_obj_dir = dict_arguments_of_variables["common_obj_dir"]
    data_dict_for_vis = pmt_fns.generate_data_for_plottingv2(
        dict_arguments_for_vis, dict_arguments_of_variables, fdecoder
    )

    collected_data_dict_for_plotting = pmt_fns.collect_any_generated_data_for_plotting(
        data_dict_for_vis, batch_idx=0
    )
    # march------------------------------------------------------------------------------------------------------------------------------------------
    collected_decoded_masked_non_optimized_latent_codes_array = (
        collected_data_dict_for_plotting[
            "collected_decoded_masked_non_optimized_latent_codes"
        ]
    )
    collected_sub_voxels_array = collected_data_dict_for_plotting[
        "collected_sub_voxels"
    ]
    # write objs------------------------------------------------------------------------------------------------------------------------------------------
    Masked_non_optimized_latent_codes_file_name_obj = (
        make_mcubes_from_voxels_obj_for_pad(
            collected_decoded_masked_non_optimized_latent_codes_array,
            object_indices[0].item(),
            "Masked_non_optimized_latent_codes",
            common_obj_dir,
        )
    )
    True_gt_sdf_file_name_obj = make_mcubes_from_voxels_obj_for_pad(
        collected_sub_voxels_array,
        object_indices[0].item(),
        "True_gt_sdf",
        common_obj_dir,
    )


def march_voxels_and_write_objs(
    dict_arguments_for_vis: dict,
    dict_arguments_of_variables: dict,
    object_indices: torch.Tensor,
    fdecoder,
) -> Tuple[dict, dict]:
    marching_cube_result_dir = dict_arguments_of_variables["marching_cube_result_dir"]
    common_obj_dir = dict_arguments_of_variables["common_obj_dir"]

    data_dict_for_vis = pmt_fns.generate_data_for_plottingv2(
        dict_arguments_for_vis, dict_arguments_of_variables, fdecoder
    )

    collected_data_dict_for_plotting = pmt_fns.collect_any_generated_data_for_plotting(
        data_dict_for_vis, batch_idx=0
    )
    # march------------------------------------------------------------------------------------------------------------------------------------------
    collected_decoded_non_optimized_latent_codes_array = (
        collected_data_dict_for_plotting["collected_sub_voxels_decoded_non_optimized"]
    )
    collected_decoded_masked_non_optimized_latent_codes_array = (
        collected_data_dict_for_plotting[
            "collected_decoded_masked_non_optimized_latent_codes"
        ]
    )
    collected_transformer_output_sequence_array = collected_data_dict_for_plotting[
        "transformer_output_sequence_up_collected_32cubes"
    ]
    collected_sub_voxels_array = collected_data_dict_for_plotting[
        "collected_sub_voxels"
    ]
    # write objs------------------------------------------------------------------------------------------------------------------------------------------
    Non_Optimized_latent_codes_file_name_obj = make_mcubes_from_voxels_obj_for_pad(
        collected_decoded_non_optimized_latent_codes_array,
        object_indices[0].item(),
        "Non_Optimized_latent_codes",
        marching_cube_result_dir,
    )

    # Non_Optimized_latent_codes_file_name_obj = marching_cube_result_dir + "Non_Optimized_latent_codes" + '-' + 'ObjID=' + str(object_indices[0].item()) + '_' + '.obj'

    Transformer_output_file_name_obj = make_mcubes_from_voxels_obj_for_pad(
        collected_transformer_output_sequence_array,
        object_indices[0].item(),
        "Transformer_output",
        marching_cube_result_dir,
    )
    # Transformer_output_file_name_obj = marching_cube_result_dir + "Transformer_output" + '-' + 'ObjID=' + str(object_indices[0].item()) + '_' + '.obj'

    # We do not march them again

    # Masked_non_optimized_latent_codes_file_name_obj = make_mcubes_from_voxels_obj_for_pad(
    #     collected_decoded_masked_non_optimized_latent_codes_array, object_indices[0].item(), "Masked_non_optimized_latent_codes", marching_cube_result_dir
    # )
    Masked_non_optimized_latent_codes_file_name_obj = (
        common_obj_dir
        + "Masked_non_optimized_latent_codes"
        + "-"
        + "ObjID="
        + str(object_indices[0].item())
        + "_"
        + ".obj"
    )

    # True_gt_sdf_file_name_obj = make_mcubes_from_voxels_obj_for_pad(collected_sub_voxels_array, object_indices[0].item(), "True_gt_sdf", marching_cube_result_dir)
    True_gt_sdf_file_name_obj = (
        common_obj_dir
        + "True_gt_sdf"
        + "-"
        + "ObjID="
        + str(object_indices[0].item())
        + "_"
        + ".obj"
    )

    dict_arguments_for_eval = {
        "Non_Optimized_latent_codes_file_name_obj": Non_Optimized_latent_codes_file_name_obj,
        "Masked_non_optimized_latent_codes_file_name_obj": Masked_non_optimized_latent_codes_file_name_obj,
        "Transformer_output_file_name_obj": Transformer_output_file_name_obj,
        "True_gt_sdf_file_name_obj": True_gt_sdf_file_name_obj,
    }
    return (dict_arguments_for_eval, collected_data_dict_for_plotting)
