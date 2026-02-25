import os
import sys

sys.path.append("..")

import torch
import pytorch_lightning as pl

import argparse
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor
from pvae import SDFtoSDF


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dim", default=512, type=int)
    parser.add_argument("--resolution", default=128, type=int)
    parser.add_argument("--target_resolution", default=32, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)

    # By mistake the test dictionary is called validation throughout the p_vae pipeline
    parser.add_argument(
        "--train_val_dict_path",
        default="/path_to_poc-slt_data/",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--lmdb_path", default="/path_to_train_lmdb/", type=str, required=True
    )

    # TODO: if you are using our pre-maed lmdb you can set the mesh_path hard-coded as "/graphics/scratch2/staff/zakeri/LMDBs/ShapeNetCorev2_remeshed_0.008/ShapeNetCore.v2/"
    # TODO, if you are generating the lmdb by your self, you must adopt it.
    parser.add_argument(
        "--mesh_path", default="path_to_remeshed_shapenet", type=str, required=True
    )
    parser.add_argument(
        "--marching_cube_result_dir",
        default="/path_to_marching_cube_result_dir/",
        type=str,
    )

    parser.add_argument("--points_to_sample", default=64, type=int)
    parser.add_argument("--query_number", default=1000, type=int)
    parser.add_argument("--value_range", default=1, type=int)
    parser.add_argument("--examples_per_epoch", default=1000, type=int)

    args = parser.parse_args()
    # write the checkpoints every 1000 steps
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="train_loss",
        filename="checkpoint-{epoch:03d}-{loss:.3f}",
        save_top_k=6,
        mode="min",
        verbose=True,
        every_n_train_steps=1000,
    )
    lr_Monitor = LearningRateMonitor(logging_interval="step")
    model = SDFtoSDF(
        latent_dim=args.latent_dim,
        resolution=args.resolution,
        target_resolution=args.target_resolution,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_val_dict_path=args.train_val_dict_path,
        lmdb_path=args.lmdb_path,
        mesh_path=args.mesh_path,
        points_to_sample=args.points_to_sample,
        query_number=args.query_number,
        value_range=args.value_range,
        examples_per_epoch=args.examples_per_epoch,
        marching_cube_result_dir=args.marching_cube_result_dir,
    )

    # configure the pytorch-lightning trainer.
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=-1,
        strategy=DDPStrategy(process_group_backend="gloo"),
        max_epochs=186,
        log_every_n_steps=100,
        detect_anomaly=True,
        callbacks=[checkpoint_callback, lr_Monitor],
        val_check_interval=10747,
        check_val_every_n_epoch=1,
        default_root_dir="/path_to_tensorboard_log_root/",
    )

    trainer.fit(model)


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    print("torch.cuda.device_count()", torch.cuda.device_count())
    print("torch.cuda.nccl.version()", torch.cuda.nccl.version())

    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
