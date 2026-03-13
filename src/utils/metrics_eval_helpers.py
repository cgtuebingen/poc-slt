import torch
import os
import re
from typing import Any

def extract_files_with_given_extension_general(eval_dir: str, ext: str) -> list:
    all_files = os.listdir(eval_dir)
    print(all_files)
    file_path_list = []
    for i in range(len(all_files)):
        file = all_files[i]
        if file.endswith(ext):
            file_path_list.append(os.path.join(eval_dir, file))
    return file_path_list

def extract_files_with_given_extension(metric_dir: str, ext: str) -> list:
    all_files = os.listdir(metric_dir)
    print(all_files)
    file_path_list = []
    for i in range(len(all_files)):
        file = all_files[i]
        if file.endswith(ext):
            file_path_list.append(os.path.join(metric_dir, file))
    return file_path_list

def read_dict_file(file_path: str) -> dict:
    dict_data = torch.load(file_path, map_location='cuda:0')
    return dict_data

def write_list_to_text(given_list: list, metric_dir: str, name: str, num_points: int, resolution: int = 128):
    # open file in write mode for broken files
    if name == "Iou-broken":
        file_name = name + "-Voxel" + "_" + str(len(given_list)) + "-broken_files" + "_numPoints-" + str(resolution) + "-cube" + ".txt"
    elif name == "Iou-non-broken":
        file_name = name + "-Voxel" + "_" + str(len(given_list)) + "-non-broken_files" + "_numPoints-" + str(resolution) + "-cube" + ".txt"
    elif name == "TMD-Chamfer-broken":
        file_name = name + "-PointCloud" + "_" + str(len(given_list)) + "-broken_files" + "_numPoints-" + str(resolution) + "-cube" + ".txt"
    elif name == "TMD-Chamfer-non-broken":
        file_name = name + "-PointCloud" + "_" + str(len(given_list)) + "-non-broken_files" + "_numPoints-" + str(resolution) + "-cube" + ".txt"
    elif name == "Intel-Fscore1%-broken":
        file_name = name + "-PointCloud" + "_" + str(len(given_list)) + "-broken_files" + "_numPoints-" + str(num_points) + "-cube" + ".txt"
    elif name == "Intel-Fscore1%-non-broken":
        file_name = name + "-PointCloud" + "_" + str(len(given_list)) + "-non-broken_files" + "_numPoints-" + str(num_points) + "-cube" + ".txt"
    elif name == "UHD-Hausdorff-broken":
        file_name = name + "-PointCloud" + "_" + str(len(given_list)) + "-broken_files" + "_numPoints-" + str(num_points) + "-cube" + ".txt"
    elif name == "UHD-Hausdorff-non-broken":
        file_name = name + "-PointCloud" + "_" + str(len(given_list)) + "-non-broken_files" + "_numPoints-" + str(num_points) + "-cube" + ".txt"
    else:
        raise "name is not valid"

    broken_file_path = os.path.join(metric_dir, file_name)
    with open(broken_file_path, 'w') as fp:
        for item in given_list:
            # write each item on a new line
            fp.write("%s\n" % item)
        print("\n" + name + 'files writing Done')

def extract_object_id_from_file_name(list_of_file_names, metric_name: str) -> list[Any]:
    object_indices_list = []
    for i in range(len(list_of_file_names)):
        file_name_path = list_of_file_names[i]
        file_name_str = os.path.basename(os.path.normpath(file_name_path))  # path =
        num = re.findall(r'\d+', file_name_str)
        if metric_name == "Fscore1%":
            object_indices_list.append(num[1])
        else:
            object_indices_list.append(num[0])

    return object_indices_list

def read_text_file(file_path):
    with open(file_path, 'r') as f:
        string = f.read()
    return string

def extract_from_dict(dict_data):
    keys = [key for key in dict_data.keys()]
    key_value_tensor = torch.zeros(len(keys))
    for i in range(len(keys)):
        current_key = keys[i]
        current_value = dict_data.get(current_key)
        string_tensor = torch.as_tensor(dict_data, dtype=torch.float32)
    print("\n")

def write_dict_eval_into_text(dict_data: dict, out_path: str, name: str) -> None:
    eval_file_name = name + ".txt"

    keys = [key for key in dict_data.keys()]

    if os.path.isdir(out_path):
        file_obj = open(os.path.join(out_path, eval_file_name), "w")
        for i in range(len(keys)):
            current_key = keys[i]
            current_value = dict_data.get(current_key)

            tmp = current_key + " = "
            tmp_v = "{:.7f}".format(current_value)
            file_obj.write(tmp + str(tmp_v))
            file_obj.write("\n--------------------------------------------")
            file_obj.write("\n")

    else:
        raise ("\n dir for writing eval data does not exist!")