import os

import argparse
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
import pytorch_lightning as pl

from train_no_empty_masking_custom_shapenet import (
    TransformerSDFtoSDFShapenetNormalizedNoEmptyMaskingCustom,
)


def main():
    parser = argparse.ArgumentParser()
    #  for SDFtoSDF
    parser.add_argument("--points_to_sample", default=1024, type=int)  #
    parser.add_argument("--query_number", default=1024, type=int)  #
    parser.add_argument("--examples_per_epoch", default=1000, type=int)  #

    parser.add_argument("--latent_dim", default=512, type=int)  # 512
    parser.add_argument("--resolution", default=128, type=int)
    parser.add_argument("--target_resolution", default=32, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--warmup_ratio", default=0.02, type=float)
    # --------------------------------------------
    parser.add_argument(
        "--train_lmdb_path",
        default="/graphics/scratch2/staff/zakeri/LMDBs/shapenetcorev2_SDF_SpanningMultiResVoxel32_128fullmesh_normalized_train/encoded_combined",  # dataset for full mesh with 128^3
        type=str,
    )

    # by mistake, evaluation lmdb is called validation in this pipeline. The validation is the first 100 objects in the training set used to monitor
    # training performance
    parser.add_argument(
        "--val_lmdb_path",
        default="/graphics/scratch2/staff/zakeri/LMDBs/shapenetcorev2_SDF_SpanningMultiResVoxel32_128fullmesh_ExcludingValSplit_normalized_val/_with_NonOptimizedLatentCodes/",  # dataset for full mesh with 128^3
        type=str,
    )

    parser.add_argument(
        "--mesh_path",
        default="/graphics/scratch2/staff/ruppert/scart/ShapeNetCorev2_remeshed_0.008/ShapeNetCore.v2/",
        type=str,
    )

    parser.add_argument("--value_range", default=1, type=int)

    parser.add_argument(
        "--vae_checkpoint_path",
        default="/graphics/scratch3/staff/zakeri/VAE_Checkpoint/checkpoint-epoch=193-loss=0.000.ckpt/",
        type=str,
    )
    parser.add_argument(
        "--marching_cube_result_dir",
        default="/graphics/scratch2/staff/zakeri/train_logs/Transformer/flash_attention/with_optimized_latent_codes/full_dataset/overfitting/clean_code/regular_cat_fulldataset_alternative_test3_normalized_shapenet_noEmptymasking_custom/marching_cube_results_0/",
        type=str,
    )

    parser.add_argument(
        "--transformer_checkpoint_path",  # no_empty_masking checkpoint is to initialize the no_empty_masking_custom model
        # default="/graphics/scratch2/staff/zakeri/train_logs/Transformer/flash_attention/with_optimized_latent_codes/full_dataset/overfitting/clean_code/regular_cat_fulldataset_alternative_test3_normalized_shapenet_noEmptymasking/lightning_logs/version_5/checkpoints/saved/checkpoint-epoch=791-loss=0.000.ckpt/",
        default="/graphics/scratch3/staff/zakeri/scratch2_coppied/train_logs/Transformer/flash_attention/with_optimized_latent_codes/full_dataset/overfitting/clean_code/regular_cat_fulldataset_alternative_test3_normalized_shapenet_noEmptymasking/lightning_logs/version_5/checkpoints/saved/checkpoint-epoch=791-loss=0.000.ckpt",
        type=str,
    )

    # hparams for transformer
    parser.add_argument(
        "--layers", default=20, type=int
    )  # layers: Number of transformer layers.
    parser.add_argument(
        "--dim_size", default=512 * 4, type=int
    )  # Dimensionality of latent space in transformer.
    parser.add_argument(
        "--heads", default=16, type=int
    )  # heads: Number of attention heads.
    parser.add_argument("--pre_trained", default=True, type=bool)
    parser.add_argument("--masking_ratio", default=0.60, type=float)

    # num_gpus = 3
    # num_train_steps = len(train_dataset) // (hparams.batch_size * num_gpus) * trainer.max_epochs
    # print("\n num_train_steps: ", num_train_steps)
    # num_warmup_steps = int(warmup_ratio * num_train_steps)
    # print("\n num_warmup_steps: ", num_warmup_steps)
    #
    parser.add_argument("--num_warmup_steps", default=1000, type=int)
    parser.add_argument("--num_training_steps", default=1000000, type=int)

    args = parser.parse_args()
    # write the checkpoints every 1000 steps
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="train_loss",
        filename="checkpoint-{epoch:03d}-{loss:.3f}",
        save_top_k=10,
        save_last=True,
        mode="min",
        verbose=True,
        every_n_train_steps=150,
    )
    lr_Monitor = LearningRateMonitor(logging_interval="step")

    model = TransformerSDFtoSDFShapenetNormalizedNoEmptyMaskingCustom(
        latent_dim=args.latent_dim,
        resolution=args.resolution,
        target_resolution=args.target_resolution,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        train_lmdb_path=args.train_lmdb_path,
        val_lmdb_path=args.val_lmdb_path,
        mesh_path=args.mesh_path,
        value_range=args.value_range,
        vae_checkpoint_path=args.vae_checkpoint_path,
        marching_cube_result_dir=args.marching_cube_result_dir,
        layers=args.layers,
        dim_size=args.dim_size,
        heads=args.heads,
        pre_trained=args.pre_trained,
        masking_ratio=args.masking_ratio,
        points_to_sample=args.points_to_sample,
        query_number=args.query_number,
        examples_per_epoch=args.examples_per_epoch,
        transformer_checkpoint_path=args.transformer_checkpoint_path,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_training_steps,
    )

    # configure the pytorch-lightning trainer.
    trainer = pl.Trainer(
        # args,
        accelerator="gpu",
        devices=-1,
        num_nodes=1,
        strategy=DDPStrategy(process_group_backend="NCCL"),
        max_epochs=2200,
        log_every_n_steps=200,
        detect_anomaly=False,
        callbacks=[checkpoint_callback, lr_Monitor],
        val_check_interval=10000,
        check_val_every_n_epoch=None,
        default_root_dir="/graphics/scratch2/staff/zakeri/tmp/pocslt_test/train_log/",
    )
    trainer.fit(model)
    print("CUDA_VISIBLE_DEVICES", os.environ["CUDA_VISIBLE_DEVICES"])


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

    print("torch.cuda.device_count()", torch.cuda.device_count())
    print("torch.cuda.nccl.version()", torch.cuda.nccl.version())
    torch.cuda.empty_cache()
    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
