import os
import torch
from typing import Tuple
from pycu3d.test.evaluate_all import EVALALLMETRICS
from Transformer.Attention.TransformerExperiments.clean_code.clean_code_common_files import plot_march_fns as pmt_fns
from Visualization.m_cube_fns import make_mcubes_from_voxels_obj_for_pad


def evaluate(dict_arguments_for_eval: dict, dict_arguments_of_variables: dict, collected_data_dict_for_plotting: dict) -> dict:
    object_index = dict_arguments_of_variables["object_index"]
    obj_file_name = dict_arguments_of_variables["obj_file_name"]
    num_samples = dict_arguments_of_variables["num_samples"]

    hausdorff_scale = dict_arguments_of_variables["hausdorff_scale"]
    chamfer_scale = dict_arguments_of_variables["chamfer_scale"]

    completed_voxel = collected_data_dict_for_plotting["transformer_output_sequence_up_collected_32cubes"]
    gt_voxel = collected_data_dict_for_plotting["collected_sub_voxels"]

    partial_obj = dict_arguments_for_eval["Masked_non_optimized_latent_codes_file_name_obj"]
    completed_obj = dict_arguments_for_eval["Transformer_output_file_name_obj"]
    gt_obj = dict_arguments_for_eval["True_gt_sdf_file_name_obj"]

    # on voxels
    eval_obj = EVALALLMETRICS()
    iou, fscore = eval_obj.eval_iou_and_fscore_voxels_cud3d(completed_voxel, gt_voxel)
    if torch.isnan(torch.tensor(iou)) or torch.isnan(torch.tensor(fscore)):
        iou = -1
        fscore = -1
    # print("\n iou:", iou, ', fscore: ', fscore)

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
        # print("\n chamfer:", chamfer)

        fscore_one_percent = eval_obj.eval_fscore_pc_cud3d(completed_pc, gt_pc, thres=0.02)

        # print("\nfscore1%:", fscore_one_percent)
        uhd = eval_obj.eval_uhd(partial_pc, completed_pc)
        uhd = uhd * hausdorff_scale
        # print("\n uhd:", uhd)

        hausdorff = eval_obj.eval_hausdorff(completed_pc, gt_pc)
        hausdorff = hausdorff * hausdorff_scale
        # print("\n hausdorff:", hausdorff)

        normal_consistency, inaccurate_normals = eval_obj.eval_nc_and_in(completed_pc, gt_pc)
        # print("\n normal_consistency:", normal_consistency, ", inaccurate_normals: ", inaccurate_normals)

        completeness = eval_obj.eval_completeness(completed_pc, gt_pc, thres=0.03)
        # print("\n completeness:", completeness)
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

    pickle_name = os.path.join(eval_dir, "eval_results_for_" + "objID=" + str(object_index.item())  + ".pkl")
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


def march_gt_and_mask_only_and_write_objs(dict_arguments_for_vis: dict, dict_arguments_of_variables: dict, object_indices: torch.Tensor, fdecoder) -> None:
    common_obj_dir = dict_arguments_of_variables['common_obj_dir']
    data_dict_for_vis = pmt_fns.generate_data_for_plottingv2(dict_arguments_for_vis, dict_arguments_of_variables, fdecoder)

    collected_data_dict_for_plotting = pmt_fns.collect_any_generated_data_for_plotting(data_dict_for_vis, batch_idx=0)
    # march------------------------------------------------------------------------------------------------------------------------------------------
    collected_decoded_masked_non_optimized_latent_codes_array = collected_data_dict_for_plotting["collected_decoded_masked_non_optimized_latent_codes"]
    collected_sub_voxels_array = collected_data_dict_for_plotting["collected_sub_voxels"]
    # write objs------------------------------------------------------------------------------------------------------------------------------------------
    Masked_non_optimized_latent_codes_file_name_obj = make_mcubes_from_voxels_obj_for_pad(
        collected_decoded_masked_non_optimized_latent_codes_array, object_indices[0].item(), "Masked_non_optimized_latent_codes", common_obj_dir
    )
    True_gt_sdf_file_name_obj = make_mcubes_from_voxels_obj_for_pad(collected_sub_voxels_array, object_indices[0].item(), "True_gt_sdf", common_obj_dir)


def march_voxels_and_write_objs(dict_arguments_for_vis: dict, dict_arguments_of_variables: dict, object_indices: torch.Tensor, fdecoder) -> Tuple[dict, dict]:
    marching_cube_result_dir = dict_arguments_of_variables['marching_cube_result_dir']
    common_obj_dir = dict_arguments_of_variables['common_obj_dir']

    data_dict_for_vis = pmt_fns.generate_data_for_plottingv2(dict_arguments_for_vis, dict_arguments_of_variables, fdecoder)

    collected_data_dict_for_plotting = pmt_fns.collect_any_generated_data_for_plotting(data_dict_for_vis, batch_idx=0)
    # march------------------------------------------------------------------------------------------------------------------------------------------
    collected_decoded_non_optimized_latent_codes_array = collected_data_dict_for_plotting["collected_sub_voxels_decoded_non_optimized"]
    collected_decoded_masked_non_optimized_latent_codes_array = collected_data_dict_for_plotting["collected_decoded_masked_non_optimized_latent_codes"]
    collected_transformer_output_sequence_array = collected_data_dict_for_plotting["transformer_output_sequence_up_collected_32cubes"]
    collected_sub_voxels_array = collected_data_dict_for_plotting["collected_sub_voxels"]
    # write objs------------------------------------------------------------------------------------------------------------------------------------------
    Non_Optimized_latent_codes_file_name_obj = make_mcubes_from_voxels_obj_for_pad(
        collected_decoded_non_optimized_latent_codes_array, object_indices[0].item(), "Non_Optimized_latent_codes", marching_cube_result_dir
    )

    # Non_Optimized_latent_codes_file_name_obj = marching_cube_result_dir + "Non_Optimized_latent_codes" + '-' + 'ObjID=' + str(object_indices[0].item()) + '_' + '.obj'


    Transformer_output_file_name_obj = make_mcubes_from_voxels_obj_for_pad(collected_transformer_output_sequence_array, object_indices[0].item(), "Transformer_output", marching_cube_result_dir)
    # Transformer_output_file_name_obj = marching_cube_result_dir + "Transformer_output" + '-' + 'ObjID=' + str(object_indices[0].item()) + '_' + '.obj'

    # We do not march them again

    # Masked_non_optimized_latent_codes_file_name_obj = make_mcubes_from_voxels_obj_for_pad(
    #     collected_decoded_masked_non_optimized_latent_codes_array, object_indices[0].item(), "Masked_non_optimized_latent_codes", marching_cube_result_dir
    # )
    Masked_non_optimized_latent_codes_file_name_obj = common_obj_dir + "Masked_non_optimized_latent_codes" + '-' + 'ObjID=' + str(object_indices[0].item()) + '_' + '.obj'

    # True_gt_sdf_file_name_obj = make_mcubes_from_voxels_obj_for_pad(collected_sub_voxels_array, object_indices[0].item(), "True_gt_sdf", marching_cube_result_dir)
    True_gt_sdf_file_name_obj = common_obj_dir + "True_gt_sdf" + '-' + 'ObjID=' + str(object_indices[0].item()) + '_' + '.obj'

    dict_arguments_for_eval = {
        "Non_Optimized_latent_codes_file_name_obj": Non_Optimized_latent_codes_file_name_obj,
        "Masked_non_optimized_latent_codes_file_name_obj": Masked_non_optimized_latent_codes_file_name_obj,
        "Transformer_output_file_name_obj": Transformer_output_file_name_obj,
        "True_gt_sdf_file_name_obj": True_gt_sdf_file_name_obj,
    }
    return (dict_arguments_for_eval, collected_data_dict_for_plotting)




