import argparse
import os

from src.evaluation.shapenet.eval_shapenet import EVALShapenet
import torch


def main_half(
    eval_mode_dir: str, obj_dir: str, common_obj_dir: str, checkpoint_mode_path: str
):
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
        # default="path_to_p_vae_checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--lmdb_path",
        # default="path_to_poc-slt_test_lmdb",  # dataset for full mesh with 128^3
        type=str,
        required=True
    )

    parser.add_argument(
        "--orig_mesh_bbx_path",
        default="path_to/data/Shapenet/all_mesh_file_bbx.pkl",
        type=str,
    )
    parser.add_argument("--pre_trained", default=True, type=bool)

    # eval
    # for octant used in the paper: "front-bottom-right"
    parser.add_argument(
        "--custom_mask_mode", default="bottom-half", type=str,  required=True
    )
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--num_samples", default=1000000, type=int)

    parser.add_argument("--min_range", type=int,  required=True)
    parser.add_argument("--max_range", type=int,  required=True)

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

    eval_root = os.path.join("path_to_eval_root", "eval")

    eval_mode_dir = os.path.join(eval_root, "ev1", "bottom_half/")  # TODO, change your version and MR here
    if not os.path.isdir(eval_mode_dir):
        os.makedirs(eval_mode_dir)

    checkpoint_dir = "path_to_poc-slt-checkpoint"

    eval_dir = os.path.join(eval_mode_dir, "eval_dir/")
    if not os.path.isdir(eval_dir):
        os.mkdir(eval_dir)
    obj_dir = os.path.join(eval_mode_dir, "obj_dir/")
    if not os.path.isdir(obj_dir):
        os.mkdir(obj_dir)
    common_obj_dir = os.path.join(eval_mode_dir, "common_obj_dir/")
    if not os.path.isdir(common_obj_dir):
        os.mkdir(common_obj_dir)

    main_half(eval_dir, obj_dir, common_obj_dir, checkpoint_dir)
