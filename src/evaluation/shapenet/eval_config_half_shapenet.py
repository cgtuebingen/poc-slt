import argparse
import os

from src.evaluation.shapenet.eval_shapenet import EVALShapenet
import torch

def main_half(eval_mode_dir: str, obj_dir: str, common_obj_dir: str, checkpoint_mode_path: str):
    parser = argparse.ArgumentParser()
    #  for SDFtoSDF
    parser.add_argument("--latent_dim", default=512, type=int)  # 512
    parser.add_argument("--resolution", default=128, type=int)
    parser.add_argument("--target_resolution", default=32, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument(
        "--mesh_path",
        default="/graphics/scratch2/staff/ruppert/scart/ShapeNetCorev2_remeshed_0.008/ShapeNetCore.v2/",
        type=str,
    )
    parser.add_argument("--points_to_sample", default=1024, type=int)
    parser.add_argument("--query_number", default=1000, type=int)
    parser.add_argument("--value_range", default=1, type=int)
    parser.add_argument("--examples_per_epoch", default=1000, type=int)

    parser.add_argument(
        "--vae_checkpoint_path",
        default="/graphics/scratch3/staff/zakeri/scratch2_coppied/train_logs/VAE/skip_connection/v403_64_2x2x2_noBNDecoder_shapenetcorev2_excluding_shapenetcorev1_validation_split/lightning_logs/version_0/checkpoints/saved/checkpoint-epoch=193-loss=0.000.ckpt",
        type=str,
    )

    parser.add_argument(
        "--lmdb_path",
        default="/graphics/scratch2/staff/zakeri/LMDBs/shapenetcorev2_SDF_SpanningMultiResVoxel32_128fullmesh_normalized_val/encoded_combined/_with_NonOptimizedLatentCodes_new",  # dataset for full mesh with 128^3
        type=str,
    )

    parser.add_argument(
        "--orig_mesh_bbx_path",
        default="/graphics/scratch2/staff/zakeri/all_mesh_file_names_shapenetCorev1_55/all_mesh_file_bbx.pkl",
        type=str,
    )
    parser.add_argument("--pre_trained", default=True, type=bool)

    # eval
    # for octant used in the paper: "front-bottom-right"
    parser.add_argument("--custom_mask_mode", default="bottom-half", type=str, required=False)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--num_samples", default=1000000, type=int)

    parser.add_argument("--min_range", type=int, default=0)
    parser.add_argument("--max_range", type=int, default=10)

    parser.add_argument(
        "--eval_dir",
        default=eval_mode_dir,
        type=str,
    )

    parser.add_argument(
        "--marching_cube_result_dir",
        default=obj_dir,
        type=str,
    )
    parser.add_argument(
        "--common_obj_dir",
        default=common_obj_dir,
        type=str,
    )

    parser.add_argument(
        "--checkpoint_path",
        default=checkpoint_mode_path,
        type=str,
    )

    args = parser.parse_args()

    eval_obj = EVALShapenet(
        latent_dim=args.latent_dim,
        resolution=args.resolution,
        target_resolution=args.target_resolution,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        lmdb_path=args.lmdb_path,
        mesh_path=args.mesh_path,
        points_to_sample=args.points_to_sample,
        query_number=args.query_number,
        value_range=args.value_range,
        examples_per_epoch=args.examples_per_epoch,
        eval_dir=args.eval_dir,
        checkpoint_path=args.checkpoint_path,
        marching_cube_result_dir=args.marching_cube_result_dir,
        common_obj_dir=args.common_obj_dir,
        orig_mesh_bbx_path=args.orig_mesh_bbx_path,
        pre_trained=args.pre_trained,
        custom_mask_mode=args.custom_mask_mode,
        vae_checkpoint_path=args.vae_checkpoint_path,
        device=args.device,
        num_samples=args.num_samples,
        min_range=args.min_range,
        max_range=args.max_range,
    )

    eval_obj.evaluate_val()


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    print("torch.cuda.device_count()", torch.cuda.device_count())
    print("torch.cuda.nccl.version()", torch.cuda.nccl.version())
    torch.cuda.empty_cache()
    torch.multiprocessing.set_sharing_strategy("file_system")

    version_root = "/graphics/scratch3/staff/zakeri/scratch2_coppied/train_logs/Transformer/flash_attention/with_optimized_latent_codes/full_dataset/overfitting/clean_code/regular_cat_fulldataset_alternative_test3_normalized_shapenet_noEmptymasking_custom/lightning_logs/version_9/"

    eval_root = os.path.join("/graphics/scratch2/staff/zakeri/tmp/", "eval")
    checkpoint_root = os.path.join(version_root, "checkpoints")

    eval_mode_dir = os.path.join(eval_root, "ev0", "bottom_half/")
    if not os.path.isdir(eval_mode_dir):
        os.makedirs(eval_mode_dir)

    eval_dir = os.path.join(eval_mode_dir, 'eval_dir/')
    if not os.path.isdir(eval_dir):
        os.mkdir(eval_dir)
    obj_dir = os.path.join(eval_mode_dir, "obj_dir/")
    if not os.path.isdir(obj_dir):
        os.mkdir(obj_dir)
    common_obj_dir = os.path.join(eval_mode_dir, 'common_obj_dir/')
    if not os.path.isdir(common_obj_dir):
        os.mkdir(common_obj_dir)
    # used for evaluation of transformer
    checkpoint_mode_path = os.path.join(checkpoint_root, "saved/" + "checkpoint-epoch=2133-loss=0.000.ckpt")

    main_half(eval_dir, obj_dir, common_obj_dir, checkpoint_mode_path)
