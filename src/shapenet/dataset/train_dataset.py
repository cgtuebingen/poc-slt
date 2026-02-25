import sys
sys.path.append("/home/zakeri/Documents/Codes/MyCodes/Proposal2/SDF_VAE/")
import os
import pytorch_lightning as pl
import lmdb
import msgpack
import msgpack_numpy as m
m.patch()
from tqdm.auto import tqdm
import torch
import argparse
import numpy as np
class ShapeNetcorev1NormalizedTrainWithNonOptimizedLatentCodes(pl.LightningDataModule):
    def __init__(self, mesh_path: str, points_to_sample: int, query_number: int, lmdb_path: str, value_range: int = 1, resolution: int = 128,
                 examples_per_epoch: int = 1000, exclude_empty: bool = True):
        super().__init__()

        self.mesh_path = mesh_path
        self.points_to_sample = points_to_sample
        self.query_number = query_number
        self.lmdb_path = lmdb_path
        self.examples_per_epoch = examples_per_epoch

        self.value_range = value_range
        self.resolution = resolution

        self.len: int

        self.exclude_empty: bool = exclude_empty

        # TODO, add the list of empty ones for training only here
        self.empty_list = []
        if self.exclude_empty:
            # empty_list_file = "/scratch/zakeri/shapenetcorev2_SDF_SpanningMultiResVoxel32_128fullmesh_normalized_train/encoded_combined/empty_indices"
            empty_list_file = self.lmdb_path+"/empty_indices"
            with open(empty_list_file, 'r') as file:
                for line in file:
                    self.empty_list.append(int(line.rstrip('\n')))
            print("\n len self.empty_list:", len(self.empty_list))

        self.my_lmdb = None
        if (os.path.isdir(lmdb_path)):  # if the database exists already:
            print("\n LMDB exits")
            with lmdb.open(
                    lmdb_path,
                    # max_dbs=2,
                    readonly=True,
                    lock=False,
                    readahead=True,
                    map_size=32 * 1024 * 1024 * 1024 * 1024,
                    # max_readers=50,
            ) as my_lmdb:
                with my_lmdb.begin(write=False) as lmdb_txn:  # read it
                    self.mesh_path = msgpack.unpackb(lmdb_txn.get(b"__mesh_path__"))
                    self.points_to_sample = msgpack.unpackb(lmdb_txn.get(b"__points_to_sample__"))
                    self.examples_per_epoch = msgpack.unpackb(lmdb_txn.get(b"__examples_per_epoch__"))
                    self.query_number = msgpack.unpackb(lmdb_txn.get(b"__query_number__"))
                    self.value_range = msgpack.unpackb(lmdb_txn.get(b"__value_range__"))
                    self.resolution = msgpack.unpackb(lmdb_txn.get(b"__resolution__"))
                    self.keys = msgpack.unpackb(lmdb_txn.get(b"__keys__"))  # list of keys

                    print("\n orig len keys:", len(self.keys))
                    if self.exclude_empty:
                        self.keys = [x for x in self.keys if x not in self.empty_list]
                        print("\n non empty keys len: ", len(self.keys))
                    self.len = len(self.keys)

                    if ((self.points_to_sample != points_to_sample) or (self.examples_per_epoch != examples_per_epoch) or (self.resolution != resolution) or (self.query_number != query_number) or (
                            self.value_range != value_range)):
                        print("\n warning: LMDB has different points_to_sample:", self.points_to_sample)
                        print("\n warning: LMDB has different examples_per_epoch:", self.examples_per_epoch)
                        print("\n warning: LMDB has different resolution:", self.resolution)
                        print("\n warning: LMDB has different query_number:", self.query_number)
                        print("\n warning: LMDB has different value_range:", self.value_range)
        else:  # if it does not exist
            raise ("\n LMDB does not exits")

    def __len__(self):
        print("\n 128 fullmesh dataset class :len fn: ", len(self.keys))
        # return len(self.keys)
        return self.len

    def openLMDB(self, path: str):
        my_lmdb = lmdb.open(
            path,
            # max_dbs=2,
            readonly=True,  # we just want to read it
            lock=False,  # reading!!
            readahead=False,
            map_size=32 * 1024 * 1024 * 1024,
            # max_readers=10000,
        )
        my_lmdb.open_db()
        return my_lmdb

    def __getitem__(self, idx: int):
        if self.my_lmdb is None:  # if database object is none
            self.my_lmdb = self.openLMDB(self.lmdb_path)  # create an object and open the database

        if idx < 0 or idx is None:
            raise "invalid item index"

        if idx > len(self.keys):
            idx = idx % len(self.keys)  # reduce the idx to the len(keys)
        key = self.keys[idx]

        with self.my_lmdb.begin(write=False) as lmdb_txn:  # reading what is written before using the object
            raw_example = msgpack.unpackb(lmdb_txn.get(msgpack.packb(key)))
            # object_index = raw_example["object_index"]
            mesh_file_name = raw_example["mesh_file_name"]
            gt_sdf_voxel = np.array(raw_example["gt_sdf_voxel"], copy=True)
            non_optimized_latent_code = np.array(raw_example["non_optimized_latent_code"], copy=True)
            std = np.array(raw_example["std"], copy=True)
            var = np.array(raw_example["var"], copy=True)
            folder_name = raw_example["folder_name"]
            sub_folder_name = str(raw_example["sub_folder_name"])

        gt_sdf_voxel = torch.from_numpy(gt_sdf_voxel).to(dtype=torch.float32)
        non_optimized_latent_code = torch.from_numpy(non_optimized_latent_code).to(dtype=torch.float32)
        std = torch.from_numpy(std).to(dtype=torch.float32)
        var = torch.from_numpy(var).to(dtype=torch.float32)
        # example = {
        #     "object_index": object_index,
        #     "mesh_file_name": mesh_file_name,
        #     "gt_sdf_voxel": gt_sdf_voxel_array,
        #     "non_optimized_latent_code": non_optimized_latent_code_array,
        #     "std": std_array,
        #     "var": var_array,
        #     "folder_name": folder_name,
        #     "sub_folder_name": sub_folder_name,
        # }
        # TESTS:
        assert gt_sdf_voxel.shape == (128, 128, 128)
        assert non_optimized_latent_code.shape == (64, 512, 2, 2, 2)

        # gt_sdf_voxel_copy: torch.Tensor = torch.from_numpy(gt_sdf_voxel).to(dtype=torch.float32)
        # if gt_sdf_voxel_copy.is_shared():
        #     gt_sdf_voxel_copy = gt_sdf_voxel_copy.clone()
        #
        # non_optimized_latent_code_copy: torch.Tensor = torch.from_numpy(non_optimized_latent_code).to(dtype=torch.float32)
        # if non_optimized_latent_code_copy.is_shared():
        #     non_optimized_latent_code_copy = non_optimized_latent_code_copy.clone()
        #
        # var_copy: torch.Tensor = torch.from_numpy(var).to(dtype=torch.float32)
        # if var_copy.is_shared():
        #     var_copy = var_copy.clone()
        #
        # std_copy: torch.Tensor = torch.from_numpy(std).to(dtype=torch.float32)
        # if std_copy.is_shared():
        #     std_copy = std_copy.clone()

        return [
            key,
            mesh_file_name,
            gt_sdf_voxel,
            std,
            var,
            non_optimized_latent_code,
            folder_name,
            sub_folder_name,
        ]

    def extract_new_mesh_file_names(self, mesh_file_name: str):
        common_pattern1 = "/graphics/scratch2/staff/zakeri/LMDBs/ShapeNetCorev2_remeshed_0.008/ShapeNetCore.v2/"
        sub_name = mesh_file_name.replace(common_pattern1, "")  # "04090263/e32501e54d05abf4f1b2b421ee7abb94/models/model_normalized.obj.obj"

        common_pattern2 = "model_normalized.obj.obj"
        sub_sub_name = sub_name.replace(common_pattern2, "")  # "04090263/e32501e54d05abf4f1b2b421ee7abb94/models"

        meta_data = sub_sub_name.replace("/models", "")
        folder_name, sub_folder_name, _ = meta_data.split("/")

        new_pattern = "/ceph/ruppert/ShapeNetCorev2_remeshed_0.008/ShapeNetCore.v2/"
        new_mesh_file_name = os.path.join(new_pattern, sub_sub_name + "model_normalized.obj")
        return (new_mesh_file_name, folder_name, sub_folder_name)



