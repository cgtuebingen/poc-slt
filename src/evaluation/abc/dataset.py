from Dataset.Dataset_Class_128fullmesh_ABC_with_NonOptimizedLatentCodes_Val import ABCWITHNONOPTIMIZEDLATENTCODESVAL
def setup_dataset(obj_dir: str, val_lmdb_path: str, value_range: int, resolution: int):

    val_empty_list_file = "/graphics/scratch2/staff/zakeri/LMDBs/ABC_128cube_5KLMDB_Test_cuda/_WithnonOptimizedLatentCodes/empty_indices"
    val_dataset = ABCWITHNONOPTIMIZEDLATENTCODESVAL(obj_dir, val_lmdb_path, val_empty_list_file, value_range, resolution)

    print("\n setup: val_dataset len: ", len(val_dataset))
    return val_dataset