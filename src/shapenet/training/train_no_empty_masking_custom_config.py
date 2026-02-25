import os
import sys
sys.path.append('..')
import argparse
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
import pytorch_lightning as pl

from train_no_empty_masking_custom_shapenet import TransformerSDFtoSDFShapenetNormalizedNoEmptyMaskingCustom
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
    # on scratch 2
    # parser.add_argument(
    #     "--train_lmdb_path",
    #     default="/graphics/scratch2/staff/zakeri/LMDBs/shapenetcorev2_SDF_SpanningMultiResVoxel32_128fullmesh_normalized_train/encoded_combined/",
    #
    #     type=str,
    # )
    # on Heracleum or cluster-gpu-02
    parser.add_argument(
        "--train_lmdb_path",
        default="/scratch/zakeri/shapenetcorev2_SDF_SpanningMultiResVoxel32_128fullmesh_normalized_train/encoded_combined/",  # dataset for full mesh with 128^3
        type=str,
    )

    # parser.add_argument(
    #     "--train_lmdb_path",
    #     default="/ceph/zakeri/shapenetcorev2_SDF_SpanningMultiResVoxel32_128fullmesh_normalized_train/encoded_combined/",
    #
    #     type=str,
    # )

    parser.add_argument(
        "--val_lmdb_path",
        default="/ceph/zakeri/shapenetcorev2_SDF_SpanningMultiResVoxel32_128fullmesh_ExcludingValSplit_normalized_val/_with_NonOptimizedLatentCodes/",  # dataset for full mesh with 128^3
        type=str,
    )

    parser.add_argument(
        "--mesh_path",
        default="/graphics/scratch2/staff/ruppert/scart/ShapeNetCorev2_remeshed_0.008/ShapeNetCore.v2/",
        type=str,
    )

    parser.add_argument("--value_range", default=1, type=int)

    parser.add_argument(
        "--checkpoint_path",
        default="/graphics/scratch2/staff/zakeri/train_logs/VAE/skip_connection/v403_64_2x2x2_noBNDecoder_shapenetcorev2_excluding_shapenetcorev1_validation_split/lightning_logs/version_0/checkpoints/saved/checkpoint-epoch=193-loss=0.000.ckpt/",
        type=str,
    )
    parser.add_argument(
        "--marching_cube_result_dir",
        default="/graphics/scratch2/staff/zakeri/train_logs/Transformer/flash_attention/with_optimized_latent_codes/full_dataset/overfitting/clean_code/regular_cat_fulldataset_alternative_test3_normalized_shapenet_noEmptymasking_custom/marching_cube_results_0/",
        type=str,
    )

    parser.add_argument(
        "--transformer_checkpoint_path",
        default="/graphics/scratch2/staff/zakeri/train_logs/Transformer/flash_attention/with_optimized_latent_codes/full_dataset/overfitting/clean_code/regular_cat_fulldataset_alternative_test3_normalized_shapenet_noEmptymasking/lightning_logs/version_5/checkpoints/saved/checkpoint-epoch=791-loss=0.000.ckpt/",
        type=str,
    )

    # hparams for transformer
    parser.add_argument("--layers", default=20, type=int)  # layers: Number of transformer layers.
    parser.add_argument("--dim_size", default=512 * 4, type=int)  # Dimensionality of latent space in transformer.
    parser.add_argument("--heads", default=16, type=int)  # heads: Number of attention heads.
    parser.add_argument("--pre_trained", default=True, type=bool)
    parser.add_argument("--masking_ratio", default=0.60, type=float)

    parser = pl.Trainer.add_argparse_args(parser)
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
        checkpoint_path=args.checkpoint_path,
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
    )

    # configure the pytorch-lightning trainer.
    trainer = pl.Trainer.from_argparse_args(
        args,
        accelerator="gpu",
        devices=-1,
        num_nodes=1,
        strategy=DDPStrategy(process_group_backend="NCCL"),  # NCCL tends to be unreliable for some reason
        max_epochs=2200,
        log_every_n_steps=200,
        detect_anomaly=False,
        callbacks=[checkpoint_callback, lr_Monitor],
        val_check_interval=10000,
        check_val_every_n_epoch=None,
        default_root_dir="/graphics/scratch2/staff/zakeri/train_logs/Transformer/flash_attention/with_optimized_latent_codes/full_dataset/overfitting/clean_code/regular_cat_fulldataset_alternative_test3_normalized_shapenet_noEmptymasking_custom/",
        # precision="bf16",
        # gradient_clip_val=0.5,
        # resume_from_checkpoint ="/graphics/scratch2/staff/zakeri/train_logs/Transformer/flash_attention/with_optimized_latent_codes/full_dataset/overfitting/clean_code/regular_cat_fulldataset_alternative_test3_normalized_shapenet_noEmptymasking_custom/lightning_logs/version_4/checkpoints/checkpoint-epoch=308-loss=0.000.ckpt"
        # resume_from_checkpoint="/graphics/scratch2/staff/zakeri/train_logs/Transformer/flash_attention/with_optimized_latent_codes/full_dataset/overfitting/clean_code/regular_cat_fulldataset_alternative_test3_normalized_shapenet_noEmptymasking_custom/lightning_logs/version_5/checkpoints/checkpoint-epoch=988-loss=0.000.ckpt"
        # resume_from_checkpoint="/graphics/scratch2/staff/zakeri/train_logs/Transformer/flash_attention/with_optimized_latent_codes/full_dataset/overfitting/clean_code/regular_cat_fulldataset_alternative_test3_normalized_shapenet_noEmptymasking_custom/lightning_logs/version_7/checkpoints/last.ckpt"
        # resume_from_checkpoint ="/graphics/scratch2/staff/zakeri/train_logs/Transformer/flash_attention/with_optimized_latent_codes/full_dataset/overfitting/clean_code/regular_cat_fulldataset_alternative_test3_normalized_shapenet_noEmptymasking_custom/lightning_logs/version_8/checkpoints/last.ckpt"
    )
    trainer.fit(model)
    print("CUDA_VISIBLE_DEVICES", os.environ["CUDA_VISIBLE_DEVICES"])
    # Custom masking----------------------------------------------------------------------
    # v0-v3 are trained from scratch
    # v4 is trained with initializiation from noEmptyMasking Transformer
    # v5 is the resume of v4 only change is the gpus 0,1
    # v6 is nothing
    # v7 is the resumed of v5 for another 100 epochs with no scheduling and lr=1e-5
    # v8 is the resume of v7 with 3gpus
    # # # v9 it seems v8 is not fully convergeed we training it for another 200 epochs only


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

    print("torch.cuda.device_count()", torch.cuda.device_count())
    print("torch.cuda.nccl.version()", torch.cuda.nccl.version())
    torch.cuda.empty_cache()
    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
