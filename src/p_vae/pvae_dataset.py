import sys

sys.path.append("..")

import os
import pytorch_lightning as pl

import numpy as np

import lmdb
import msgpack
import msgpack_numpy as m

m.patch()


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class ReadLMDBDataset(pl.LightningDataModule):
    def __init__(
        self,
        data: dict,
        mesh_path: str,
        target_resolution: int,
        points_to_sample: int,
        query_number: int,
        lmdb_path: str,
        value_range: int = 1,
        resolution: int = 128,
        examples_per_epoch: int = 1000,
    ):
        super().__init__()
        self.data = data
        self.mesh_path = mesh_path
        self.points_to_sample = points_to_sample
        self.query_number = query_number
        self.lmdb_path = lmdb_path
        self.examples_per_epoch = examples_per_epoch

        self.value_range = value_range
        self.resolution = resolution
        self.target_resolution = target_resolution

        self.len: int
        self.my_lmdb = None

        if os.path.isdir(lmdb_path):  # if the database exists already:
            with lmdb.open(
                lmdb_path,
                max_dbs=2,
                readonly=True,
                lock=True,
                readahead=True,
                map_size=32 * 1024 * 1024 * 1024 * 1024,
                max_readers=50,
            ) as my_lmdb:
                with my_lmdb.begin(write=False) as lmdb_txn:  # read it
                    self.mesh_path = msgpack.unpackb(lmdb_txn.get(b"__mesh_path__"))
                    self.points_to_sample = msgpack.unpackb(
                        lmdb_txn.get(b"__points_to_sample__")
                    )
                    self.examples_per_epoch = msgpack.unpackb(
                        lmdb_txn.get(b"__examples_per_epoch__")
                    )
                    self.query_number = msgpack.unpackb(
                        lmdb_txn.get(b"__query_number__")
                    )
                    self.value_range = msgpack.unpackb(lmdb_txn.get(b"__value_range__"))
                    self.resolution = msgpack.unpackb(lmdb_txn.get(b"__resolution__"))
                    self.keys = msgpack.unpackb(
                        lmdb_txn.get(b"__keys__")
                    )  # list of keys
                    self.len = len(self.keys)
                    if (
                        (self.points_to_sample != points_to_sample)
                        or (self.examples_per_epoch != examples_per_epoch)
                        or (self.resolution != resolution)
                        or (self.query_number != query_number)
                        or (self.value_range != value_range)
                    ):
                        print(
                            "\n warning: LMDB has different points_to_sample:",
                            self.points_to_sample,
                        )
                        print(
                            "\n warning: LMDB has different examples_per_epoch:",
                            self.examples_per_epoch,
                        )
                        print(
                            "\n warning: LMDB has different resolution:",
                            self.resolution,
                        )
                        print(
                            "\n warning: LMDB has different query_number:",
                            self.query_number,
                        )
                        print(
                            "\n warning: LMDB has different value_range:",
                            self.value_range,
                        )
        else:  # if it does not exist
            raise ("\n LMDB does not exits")

    def __len__(self):
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
            self.my_lmdb = self.openLMDB(
                self.lmdb_path
            )  # create an object and open the database

        if idx < 0 or idx is None:
            raise "invalid item index"

        if idx > len(self.keys):
            idx = idx % len(self.keys)  # reduce the idx to the len(keys)
        key = self.keys[idx]

        with self.my_lmdb.begin(
            write=False
        ) as lmdb_txn:  # reading what is written before using the object
            raw_example = msgpack.unpackb(lmdb_txn.get(msgpack.packb(key)))
            gt_sdf_voxel = np.array(
                raw_example["gt_sdf_voxel"], copy=True
            )  # FIXME change the type as well
            d = np.array(raw_example["d"], copy=True)
            left = np.array(raw_example["left"], copy=True)
            x_offset = np.array(raw_example["x_offset"], copy=True)
            top = np.array(raw_example["top"], copy=True)
            y_offset = np.array(raw_example["y_offset"], copy=True)
            front = np.array(raw_example["front"], copy=True)
            z_offset = np.array(raw_example["z_offset"], copy=True)
            mesh_file_name = raw_example["mesh_file_name"]
            folder = raw_example["folder"]
            label = raw_example["label"]
            dataset_index = raw_example["dataset_index"]
            sub_folder_index = raw_example["sub_folder_index"]

        # example = {'mesh_file_name': file_name, 'gt_sdf_voxel': gt_sdf_voxel,
        #            'd': d, 'left': left, 'x_offset': x_offset, 'top': top, 'y_offset': y_offset, 'front': front, 'z_offset': z_offset,
        #            'folder': folder, 'label': label, 'dataset_index': dataset_index, 'sub_folder_index': sub_folder_index}

        d_copy = d.copy()
        gt_sdf_voxel_copy = gt_sdf_voxel.copy()
        left_copy = left.copy()
        x_offset_copy = x_offset.copy()
        top_copy = top.copy()
        y_offset_copy = y_offset.copy()
        front_copy = front.copy()
        z_offset_copy = z_offset.copy()

        return [
            key,
            mesh_file_name,
            gt_sdf_voxel_copy,
            d_copy,
            left_copy,
            x_offset_copy,
            top_copy,
            y_offset_copy,
            front_copy,
            z_offset_copy,
            folder,
            label,
            dataset_index,
            sub_folder_index,
        ]  # for vae training
