import os
def extract_remeshed_meshes_file_name_from_mesh_file_name(mesh_file_name: str, remeshed_mesh_path:str) -> str:
    common_pattern1 = "/graphics/scratch2/staff/zakeri/LMDBs/ShapeNetCorev2_remeshed_0.008/ShapeNetCore.v2/"
    # remeshed_meshes_dataset_path = self.mesh_path
    sub_name = mesh_file_name.replace(common_pattern1, "")
    remeshed_mesh_file_name = os.path.join(remeshed_mesh_path, sub_name)
    return remeshed_mesh_file_name


def extract_original_meshes_file_name_from_mesh_file_name(mesh_file_name: str, orig_mesh_path: str) -> str:
    common_pattern1 = "/graphics/scratch2/staff/zakeri/LMDBs/ShapeNetCorev2_remeshed_0.008/ShapeNetCore.v2/"
    # original_meshes_dataset_path = self.orig_mesh_path
    sub_name = mesh_file_name.replace(common_pattern1, "")
    sub_sub_name = sub_name.replace(".obj", "")
    original_mesh_file_name = os.path.join(orig_mesh_path, sub_sub_name + '.obj')
    return original_mesh_file_name
