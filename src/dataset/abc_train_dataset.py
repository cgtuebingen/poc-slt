# we read from the lmdb that is a combination of optimized_latent_codes and manually optimized latent codes call it old_lmdb
# for every batch that we parse, we get original gt_sdf_voxel and subdivide it to sub-voxels
# we normalize sub-voxels by multiply them with 2 and reshape them correctly for encoding
# we have uploaded the encoder frozen, and we encode sub_voxels to get the non-optimized_latent code
# with the previous batch, and adding the non-optimized_latent_code, we create a new example
# we open a new lmdb and write the new example and write it in a loop until the old_lmdb is completely read
# we test the new one by once going through all of its example and do assert and so on
# #
import os
import torch
import lmdb
import msgpack
import msgpack_numpy as m
m.patch()
# from LMDB_Distributed_Generator.Cpu_.helper_functions import *
import pytorch_lightning as pl
import numpy as np

# ------------------------------------------------------------------------------------------------------------------
# we merge non-optimized-latent-codes with their corresponding lmdb sample from (manually optimized latent codes and 128 full mesh) lmdb
class ABCWITHNONOPTIMIZEDLATENTCODES(pl.LightningDataModule):
    def __init__(self, obj_dir, lmdb_path, value_range, resolution):
        super(ABCWITHNONOPTIMIZEDLATENTCODES).__init__()

        self.obj_dir = obj_dir
        self.lmdb_path = lmdb_path
        self.value_range = value_range
        self.resolution = resolution

        self.my_lmdb = None

        self.empty_list = []
        # old ABC
        # empty_list_file = "/graphics/scratch2/staff/zakeri/LMDBs/ABC_128cube_100KLMDB_combined/ABC_128cube_100KLMDB_combined_with_NonOptimizedLatentCodes/empty_indices"
        empty_list_file = self.lmdb_path + "/empty_indices"
        with open(empty_list_file, 'r') as file:
            for line in file:
                self.empty_list.append(int(line.rstrip('\n')))
        print("\n len self.empty_list:", len(self.empty_list))

        if (os.path.isdir(lmdb_path)):  # if the database exists already:
            print("\n LMDB exits")
            with lmdb.open(
                lmdb_path,
                max_dbs=2,
                readonly=True,
                lock=False,
                readahead=True,
                map_size=32 * 1024 * 1024 * 1024 * 1024,
                max_readers=50,
            ) as my_lmdb:
                with my_lmdb.begin(write=False) as lmdb_txn:  # read it
                    self.mesh_path = msgpack.unpackb(lmdb_txn.get(b"__obj_dir__"))
                    self.resolution = msgpack.unpackb(lmdb_txn.get(b"__resolution__"))
                    self.keys = msgpack.unpackb(lmdb_txn.get(b"__keys__"))  # list of keys
                    print("\n orig len keys:", len(self.keys))
                    self.keys = [x for x in self.keys if x not in self.empty_list]
                    print("\n non empty keys len: ", len(self.keys))
                    self.len = len(self.keys)
                    if ( (self.resolution != resolution) or (self.obj_dir != obj_dir)):
                        print("\n warning: LMDB has different obj_dir:", self.obj_dir)
                        print("\n warning: LMDB has different resolution:", self.resolution)
        else:  # if it does not exist
            raise ("\n LMDB does not exits")

    def __len__(self):
        print(" non-optimized merged lmdb len: ", len(self.keys))
        return self.len

    def openLMDB(self, path: str):
        my_lmdb = lmdb.open(
            path,
            max_dbs=2,
            readonly=True,  # we just want to read it
            lock=False,  # reading!!
            readahead=False,
            map_size=32 * 1024 * 1024 * 1024,
            max_readers=10000,
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
            object_index = raw_example["object_index"]
            obj_file_name = raw_example["obj_file_name"]
            gt_sdf_voxel = np.array(raw_example["gt_sdf_voxel"], copy=True)
            non_optimized_latent_code = np.array(raw_example["non_optimized_latent_code"], copy=True)
            # std = np.array(raw_example["std"], copy=True)
            # var = np.array(raw_example["var"], copy=True)
            # scale = raw_example["scale"]
            # center = raw_example["center"]

        # example = {
        #     "object_index": object_index,
        #     "obj_file_name": obj_file_name,
        #     "gt_sdf_voxel": gt_sdf_voxel_array,
        #     "non_optimized_latent_code": non_optimized_latent_code_array,
        #     "std": std_array,
        #     "var": var_array,
        #     "scale": scale,
        #     "center": center,
        # }
        # TESTS:
        # assert gt_sdf_voxel.shape == (128, 128, 128)
        # assert non_optimized_latent_code.shape == (64, 512, 2, 2, 2)

        gt_sdf_voxel_copy: torch.Tensor = torch.from_numpy(gt_sdf_voxel).to(dtype=torch.float32)
        non_optimized_latent_code_copy: torch.Tensor = torch.from_numpy(non_optimized_latent_code).to(dtype=torch.float32)
        # var_copy: torch.Tensor = torch.from_numpy(var).to(dtype=torch.float32)
        # std_copy: torch.Tensor = torch.from_numpy(std).to(dtype=torch.float32)

        return [
            key,
            object_index,
            obj_file_name,
            gt_sdf_voxel_copy,
            # std_copy,
            # var_copy,
            non_optimized_latent_code_copy,
        ]
