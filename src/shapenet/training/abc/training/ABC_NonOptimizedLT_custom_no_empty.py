import os
import sys
from typing import Tuple, Any, Union
sys.path.append("/home/zakeri/Documents/Codes/MyCodes/Proposal2/SDF_VAE/")
import torch
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

print("torch.cuda.device_count()", torch.cuda.device_count())
print("torch.cuda.nccl.version()", torch.cuda.nccl.version())
torch.cuda.empty_cache()
torch.multiprocessing.set_sharing_strategy("file_system")
from pytorch_lightning.strategies import DDPStrategy
from torch import nn
import pytorch_lightning as pl
import argparse
from Dataset.Dataset_Class_128fullmesh_ABC_with_NonOptimizedLatentCodes import ABCWITHNONOPTIMIZEDLATENTCODES
from Dataset.Dataset_Class_128fullmesh_ABC_with_NonOptimizedLatentCodes_Val import ABCWITHNONOPTIMIZEDLATENTCODESVAL
from torch.utils.tensorboard import SummaryWriter
from Experiments.stream32cube.vae_main_v1_64_2x2x2_32cubestream import SDFtoSDF
from transformers.optimization import get_cosine_schedule_with_warmup
from pytorch_lightning.callbacks import LearningRateMonitor
from Transformer.Attention.TransformerVisualizations import transformer_visualizations as tv
from Transformer.Attention.TransformerLoss import loss_helper_fns as l_fn
from Transformer.Attention.TransformerExperiments.clean_code.clean_code_common_files import plot_march_fns as pmt_fns
from Transformer.Attention.TransformerExperiments.clean_code.clean_code_common_files import sub_voxel_related_fns as pp_fns
from Transformer.Attention.TransformerExperiments.clean_code.clean_code_common_files.positional_encoder_class import MYPositionalEncoder3D
# ----------------------------------------------------------------------------------------------------------------------------------------------------
from Transformer.Attention.TransformerExperiments.clean_code.clean_code_common_files import encoder_decoder_loading as ed
from Transformer.Attention.TransformerExperiments.clean_code.clean_code_common_files import L1_loss_fns as L1_fn
from Transformer.Attention.TransformerExperiments.clean_code.clean_code_common_files import mask_huristic as gr_mask
from Transformer.Attention.TransformerExperiments.clean_code.clean_code_common_files.helper_fns import concatenate_for_given_dim
from Transformer.Attention.TransformerExperiments.clean_code.clean_code_experiments.fulldataset.completionstr2Experiments.ABC.ABC_NonOptimizedLT_no_empty import TransformerSDFtoSDFABCOUTSIDE as no_empty_trans
import numpy as np
# --------------------------------------------------------------------------------------------------------------------------------------------------------
class TransformerSDFtoSDFABCOUTSIDE(pl.LightningModule):
    def __init__(
        self,
        latent_dim: int,
        resolution: int,
        target_resolution: int,
        batch_size: int,
        val_batch_size: int,
        learning_rate: float,
        warmup_ratio: float,
        train_lmdb_path: str,
        val_lmdb_path: str,
        obj_dir: str,
        value_range: int,
        vae_checkpoint_path: str,
        marching_cube_result_dir: str,
        layers: int,
        dim_size: int,
        heads: int,
        pre_trained: bool,
        masking_ratio: torch.float32,
        transformer_checkpoint_path:str,
    ):
        super(TransformerSDFtoSDFABCOUTSIDE, self).__init__()
        self.save_hyperparameters()

        if self.hparams.pre_trained:
            print("\n pre_trained: ", pre_trained)

            pre_trained_model = SDFtoSDF.load_from_checkpoint(
                vae_checkpoint_path,
            ).to(self.device)
            pre_trained_model.freeze()
            pre_trained_model.train(False)
            # del SDFtoSDF
            self.fdecoder = ed.load_decoder_from_checkpoint(pre_trained_model, latent_dim)
            self.fdecoder.to(self.device)

            pre_trained_transformer_checkpoint = no_empty_trans.load_from_checkpoint(
                self.hparams.transformer_checkpoint_path,
            ).to(self.device)
            self.regular_transformer = pre_trained_transformer_checkpoint.regular_transformer
            del pre_trained_transformer_checkpoint
            print("\n regular transformer is initialized  from pretrained-transformer num_layer:", self.regular_transformer.num_layers, )
        self.l1_loss = nn.L1Loss(reduction="mean")

        number_of_sub_voxels = self.hparams.resolution // self.hparams.target_resolution
        self.number_of_sub_voxels = number_of_sub_voxels * number_of_sub_voxels * number_of_sub_voxels

        self.my_selected_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # this script

        # generator = torch.Generator(device=self.device)
        # generator.manual_seed(123)
        self.penc_channels = 8 * self.hparams.latent_dim
        self.positional_encoder_3d = MYPositionalEncoder3D(self.penc_channels)

        self.constant_learnable_mask_tensors = torch.nn.Parameter(torch.randn([self.hparams.latent_dim, 2, 2, 2], dtype=torch.float32, device=self.device))  # , generator=generator
        self.constant_learnable_mask_tensors_for_empty_voxels = torch.nn.Parameter(torch.randn([self.hparams.latent_dim, 2, 2, 2], dtype=torch.float32, device=self.device))  # , generator=generator

        self.mapping_down = nn.Linear(self.penc_channels + 8 * self.hparams.latent_dim, self.hparams.dim_size)  # FIXME check me
        self.mapping_up = nn.Linear(self.hparams.dim_size, 8 * self.hparams.latent_dim)  # FIXME check me
        self.redundant_mapping = nn.Linear(8 * self.hparams.latent_dim, 8 * self.hparams.latent_dim)

    def calculate_losses(
        self,
        masked_bool: torch.Tensor,
        non_masked_bool: torch.Tensor,
        transformer_output_sequence: torch.Tensor,  # output
        optimized_latent_codes_reshaped: torch.Tensor,  # gt
    ) -> dict:
        all_losses = self.l1_loss(transformer_output_sequence, optimized_latent_codes_reshaped)

        l1_loss_masked = torch.tensor(0.0).to(device=self.device)
        if torch.any(masked_bool):
            l1_loss_masked = self.l1_loss(transformer_output_sequence[masked_bool], optimized_latent_codes_reshaped[masked_bool])

        l1_loss_non_masked = torch.tensor(0.0).to(device=self.device)
        if torch.any(non_masked_bool):
            l1_loss_non_masked = self.l1_loss(transformer_output_sequence[non_masked_bool], optimized_latent_codes_reshaped[non_masked_bool])

        loss_dict = {"l1_loss_masked": l1_loss_masked, "l1_loss_non_masked": l1_loss_non_masked}
        # calculate scaled losses
        if True:
            transformer_output_sequence_shape = [transformer_output_sequence.shape[0], transformer_output_sequence.shape[1]]
            (l1_loss_scaled, l1_loss_masked_scaled, l1_loss_non_masked_scaled) = L1_fn.scale_losses_noEmptymasking(
                loss_dict, masked_bool, non_masked_bool, transformer_output_sequence_shape
            )
            loss_dict_scaled = {"l1_loss_masked": l1_loss_masked_scaled, "l1_loss_non_masked": l1_loss_non_masked_scaled}

        # weight losses
        if True:
            (l1_loss_w, l1_loss_masked_w, l1_loss_non_masked_w) = L1_fn.weight_losses_noEmptymasking(
                loss_dict_scaled, weight_masked=1.0, weight_non_masked=1.0
            )

        return {
            "l1_loss_masked": l1_loss_masked_w,
            "l1_loss_non_masked": l1_loss_non_masked_w,
            "l1_loss": l1_loss_w,
            "all_losses": all_losses,
        }
    def assign_constant_learnable_tensors(self, optimized_latent_codes: torch.Tensor, bool_dict: dict) -> torch.Tensor:
        masked_optimized_latent_codes = optimized_latent_codes.clone()

        mask_all_bool = bool_dict["mask_all_bool"]

        masked_optimized_latent_codes[mask_all_bool] = self.constant_learnable_mask_tensors

        return masked_optimized_latent_codes.clone()

    def call_transformer_and_mapping_layers(self, transformer_input_sequence: torch.Tensor) -> torch.Tensor:
        transformer_input_sequence = transformer_input_sequence.clone()

        transformer_input_sequence_down = self.mapping_down(transformer_input_sequence)
        transformer_output_sequence = self.regular_transformer(transformer_input_sequence_down)
        transformer_output_sequence_up = self.mapping_up(transformer_output_sequence)

        return transformer_output_sequence_up.clone()

    def forward(self, sub_voxels: torch.Tensor, non_optimized_latent_codes: torch.Tensor, mask_all_bool) -> Union[tuple[Any, Any, Any], Any]:
        # This part is the same for training and validation
        mask_all_bool = mask_all_bool.clone()
        batch_size = sub_voxels.shape[0]
        masked_non_optimized_latent_codes = self.assign_constant_learnable_tensors(non_optimized_latent_codes.clone(), bool_dict={"mask_all_bool": mask_all_bool})
        # -----------------------------------------------------------------------------------------------------------------------------
        # Prep for passing the masked_optimized_latent_codes to transformer--------------------------------------------------------
        # masked_optimized_latent_codes size : [B, SeqLen, 512, 2, 2, 2] --> [B, 64, 512, 2, 2, 2] --> reshape: [B, 64, 4096]
        masked_non_optimized_non_latent_codes_reshaped = masked_non_optimized_latent_codes.reshape([batch_size, self.number_of_sub_voxels, 8 * self.hparams.latent_dim])
        # redundant mapping-------------------------------------------------------------------------------------------------------
        masked_non_optimized_non_latent_codes_reshaped_mapped = self.redundant_mapping(masked_non_optimized_non_latent_codes_reshaped)
        assert masked_non_optimized_non_latent_codes_reshaped.shape == masked_non_optimized_non_latent_codes_reshaped_mapped.shape
        # Positional Embeder----------------------------------------------------------------------------------------------------
        z_positionally_encoded_re = self.positional_encoder_3d(shape_of_positions=[batch_size, 4, 4, 4, self.penc_channels])
        # Adding latent code with positional embedding-----------------------------------------------------------------------------
        assert z_positionally_encoded_re.shape == masked_non_optimized_non_latent_codes_reshaped_mapped.shape
        # CAT---------
        transformer_input_sequence = concatenate_for_given_dim(z_positionally_encoded_re, masked_non_optimized_non_latent_codes_reshaped_mapped, cat_dim=2)

        # Transformer ----------------------------------------------------------------------------------------------------
        transformer_output_sequence = self.call_transformer_and_mapping_layers(transformer_input_sequence)

        return (transformer_output_sequence, masked_non_optimized_latent_codes)
    def generate_all_masking(self, sub_voxels: torch.Tensor, object_indices: torch.Tensor, mesh_file_name):
        batch_size = object_indices.shape[0]
        masking_choice = np.random.choice(32, 1)

        empty_sub_voxels_bool, non_empty_sub_voxels_bool = pp_fns.extract_outside_and_non_outside_voxels(sub_voxels.clone(), self.number_of_sub_voxels, self.hparams.target_resolution)
        # empty_indices = []
        if not torch.all(torch.any(non_empty_sub_voxels_bool, dim=1)):
            pair = {"object_index": object_indices.item(), "mesh_file_name": mesh_file_name}
            print("\nall voxels are empty", pair)
            breakpoint()

        # generate masked bool------------------------------------------------------------------------------------------------------------
        # 2st method:
        mask_all_bool = gr_mask.mask_heuristic(masking_choice, self.hparams.masking_ratio, batch_size,  self.number_of_sub_voxels, self.hparams.target_resolution, self.device)

        # just for return
        masked_bool = mask_all_bool.clone()
        # actual non_masked_bool:
        non_masked_bool = torch.logical_and(non_empty_sub_voxels_bool, torch.logical_not(mask_all_bool))
        # num_non_masked_bool = torch.count_nonzero(non_masked_bool, dim=1)
        return (mask_all_bool, masked_bool, non_masked_bool)
    def fwd(self, batch: list, train: bool) -> Tuple[dict,  Tuple]:
        (
            keys,
            object_indices,
            obj_file_names,
            gt_sdf_full_voxel,
            # std_copy,
            # var_copy,
            non_optimized_latent_codes,
        ) = batch

        batch_size = object_indices.shape[0]
        # -----------
        sub_voxels = pp_fns.sub_divide_gt_and_normalize(gt_sdf_full_voxel.clone(), self.number_of_sub_voxels, self.hparams.target_resolution)
        # generate empty and non-empty bool-----------------------------------------------------------------------------------------------
        mask_all_bool, masked_bool, non_masked_bool = self.generate_all_masking(sub_voxels, object_indices, obj_file_names)

        # FORWARD CAlL----------------------------------------------------------------------------------------------------------------------------
        # if I want this to run , my val_batch and my train_batch need to be the same.
        (transformer_output_sequence_up, masked_non_optimized_latent_codes) = self.forward(sub_voxels, non_optimized_latent_codes, mask_all_bool)
        # for loss calculation
        non_optimized_latent_codes_reshaped = non_optimized_latent_codes.reshape([batch_size, self.number_of_sub_voxels, 8 * self.hparams.latent_dim])
        assert transformer_output_sequence_up.shape == non_optimized_latent_codes_reshaped.shape

        # calculate losses:
        loss_dict = self.calculate_losses(masked_bool, non_masked_bool, transformer_output_sequence_up, non_optimized_latent_codes_reshaped)
        loss = loss_dict["l1_loss"]

        # log losses
        if train:
            loss_log = l_fn.create_log_losses_for_given_dict(loss_dict, stage="training")

            self.log_dict(loss_log, batch_size=self.hparams.batch_size, sync_dist=True)
            self.log("train_loss", loss, batch_size=self.hparams.batch_size, sync_dist=True)

        else:
            loss_log = l_fn.create_log_losses_for_given_dict(loss_dict, stage="val")

            self.log_dict(loss_log, batch_size=self.hparams.val_batch_size, sync_dist=True)
            self.log("val_loss", loss, batch_size=self.hparams.val_batch_size, sync_dist=True)

        loss_dict = {"loss": loss}

        return (loss_dict, (masked_non_optimized_latent_codes, transformer_output_sequence_up, sub_voxels))

    def training_step(self, batch: list, batch_idx: int) -> dict:
        # (
        #     keys,
        #     object_indices,
        #     obj_file_names,
        #     gt_sdf_full_voxel,
        #     # std_copy,
        #     # var_copy,
        #     non_optimized_latent_codes
        # ) = batch
        loss_dict,  stuff = self.fwd(batch, True)

        # masked_non_optimized_latent_codes, transformer_output_sequence_up, sub_voxels = stuff
        return loss_dict

    def validation_step(self, batch: list, batch_idx: int) -> None:
        self.fdecoder.eval()
        assert not self.fdecoder.batchNorm3d5.track_running_stats
        assert not self.fdecoder.batchNorm3d6.track_running_stats
        assert not self.fdecoder.batchnorm3d7.track_running_stats
        assert not self.fdecoder.training

        (
            keys,
            object_indices,
            obj_file_names,
            gt_sdf_full_voxel,
            # std_copy,
            # var_copy,
            non_optimized_latent_codes,
        ) = batch

        batch_size = object_indices.shape[0]
        assert object_indices.shape == (self.hparams.val_batch_size,)
        loss_dict, stuff = self.fwd(batch, False)

        masked_non_optimized_latent_codes, transformer_output_sequence_up, sub_voxels = stuff

        # visualization and tensorboard---------------------------
        dict_arguments_for_vis = {
            "non_optimized_latent_codes": non_optimized_latent_codes,
            "masked_non_optimized_latent_codes": masked_non_optimized_latent_codes,
            "transformer_output_sequence_up": transformer_output_sequence_up,
            "sub_voxels": sub_voxels,
        }

        dict_arguments_of_variables = {
            "number_of_sub_voxels": self.number_of_sub_voxels,
            "latent_dim": self.hparams.latent_dim,
            "target_resolution": self.hparams.target_resolution,
            "resolution": self.hparams.resolution,
        }

        self.plot_march_and_login_tensorboard(dict_arguments_for_vis, dict_arguments_of_variables, object_indices, batch_size)

    def plot_march_and_login_tensorboard(self, dict_arguments_for_vis: dict, dict_arguments_of_variables: dict, object_indices: torch.Tensor, batch_size: int) -> None:
        data_dict_for_vis = pmt_fns.generate_data_for_plottingv2(dict_arguments_for_vis, dict_arguments_of_variables, self.fdecoder)
        for b in range(batch_size):
            selected_index = object_indices[b].detach().cpu().item()
            if selected_index in self.my_selected_indices:
                collected_data_dict_for_plotting = pmt_fns.collect_generated_data_for_plottingv3(data_dict_for_vis, self.hparams.resolution, batch_idx=b)
                plots = tv.generate_plot_for_given_dict_of_items(collected_data_dict_for_plotting, self.hparams.resolution, number_of_slices=2, plot_scale_factor=2, plot_range=2)
                self.login_to_tensorboard(plots, selected_index, number_of_slices=2)

    def login_to_tensorboard(self, plots: list, selected_index: int, number_of_slices: int):
        # plot and log slices
        for sl in range(0, number_of_slices, 1):
            # plot everything
            plot = tv.plot_for_given_dict_of_items(sl, plots).to(self.device)
            plot = plot.to(self.device)
            # show in tensorboard
            self.logger.experiment.add_image(
                "mesh-Id-{}_slice-{}".format(selected_index, sl),
                plot,
                self.global_step,
            )

    def setup(self, stage: str) -> None:
        train_empty_list_file = "/graphics/scratch2/staff/zakeri/LMDBs/ABC_128cube_100KLMDB_Train_cuda/_with_NonOptimizedLatentCodes/empty_indices"
        self.train_dataset = ABCWITHNONOPTIMIZEDLATENTCODES(self.hparams.obj_dir, self.hparams.train_lmdb_path, train_empty_list_file, self.hparams.value_range, self.hparams.resolution)

        print("\n setup: train_dataset len: ", len(self.train_dataset))

        val_empty_list_file = "/graphics/scratch2/staff/zakeri/LMDBs/ABC_128cube_5KLMDB_Test_cuda/_WithnonOptimizedLatentCodes/empty_indices"
        self.val_dataset = ABCWITHNONOPTIMIZEDLATENTCODESVAL(self.hparams.obj_dir, self.hparams.val_lmdb_path, val_empty_list_file, self.hparams.value_range, self.hparams.resolution)

        print("\n setup: val_dataset len: ", len(self.val_dataset))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=30,
            pin_memory=False,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.hparams.val_batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=False,
            drop_last=True,
        )

    def configure_optimizers(self):
        # we exclude fdecoer params from optimizer because it fucks them up, yes you head me!
        dont_train_those = []
        for k, _ in self.fdecoder.named_parameters():
            dont_train_those.append(k)
        params = []

        for k, v in self.named_parameters():
            if k not in dont_train_those:
                params.append(v)
        # print(params)

        optimizer = torch.optim.AdamW(
            params,
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.99),
            weight_decay=0.05,
        )
        num_gpus = 3
        num_train_steps = len(self.train_dataset) // (self.hparams.batch_size * num_gpus) * self.trainer.max_epochs
        print("\n num_train_steps: ", num_train_steps)
        num_warmup_steps = int(self.hparams.warmup_ratio * num_train_steps)
        print("\n num_warmup_steps: ", num_warmup_steps)

        lr_scheduler = {
            "scheduler": get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_train_steps,
                num_cycles=0.5,
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]


def main():
    parser = argparse.ArgumentParser()
    #  for SDFtoSDF
    parser.add_argument("--latent_dim", default=512, type=int)  # 512
    parser.add_argument("--resolution", default=128, type=int)
    parser.add_argument("--target_resolution", default=32, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--warmup_ratio", default=0.02, type=float)
    # on Heracleum--------------------------------------------
    # TODO Chaneme
    parser.add_argument(
        "--train_lmdb_path",
        default="/graphics/scratch2/staff/zakeri/LMDBs/ABC_128cube_100KLMDB_Train_cuda/_with_NonOptimizedLatentCodes/",  # dataset for full mesh with 128^3
        type=str,
    )
    #
    parser.add_argument(
        "--val_lmdb_path",
        default="/graphics/scratch2/staff/zakeri/LMDBs/ABC_128cube_5KLMDB_Test_cuda/_WithnonOptimizedLatentCodes/",  # dataset for full mesh with 128^3
        type=str,
    )

    # cluster-00
    # parser.add_argument(
    #     "--train_lmdb_path",
    #     default="/scratch/zakeri/ABC/ABC_128cube_100KLMDB_Train_cuda/_with_NonOptimizedLatentCodes/",  # dataset for full mesh with 128^3
    #     type=str,
    # )
    #
    # parser.add_argument(
    #     "--val_lmdb_path",
    #     default="/scratch/zakeri/ABC/ABC_128cube_5KLMDB_Test_cuda/_WithnonOptimizedLatentCodes/",  # dataset for full mesh with 128^3
    #     type=str,
    # )

    # parser.add_argument(
    #     "--lmdb_path",
    #     default="/graphics/scratch2/staff/zakeri/LMDBs/ABC_128cube_100KLMDB_combined/ABC_128cube_100KLMDB_combined_with_NonOptimizedLatentCodes/",  # dataset for full mesh with 128^3
    #     type=str,
    # )

    parser.add_argument(
        "--obj_dir",
        default="/graphics/scratch/datasets/ABC/obj/",
        type=str,
    )

    parser.add_argument("--value_range", default=1, type=int)

    parser.add_argument(
        "--vae_checkpoint_path",
        default="/graphics/scratch2/staff/zakeri/train_logs/VAE/skip_connection/v403_64_2x2x2_noBNDecoder_shapenetcorev2_excluding_shapenetcorev1_validation_split/lightning_logs/version_0/checkpoints/saved/checkpoint-epoch=193-loss=0.000.ckpt/",
        type=str,
    )

    parser.add_argument(
        "--marching_cube_result_dir",
        default="/graphics/scratch2/staff/zakeri/train_logs/Transformer/flash_attention/with_optimized_latent_codes/full_dataset/overfitting/clean_code/regular_cat_fulldataset_alternative_test3_ABC_custom_noEmpty/marching_cube_results_0/",
        type=str,
    )
    # ABC no_empty checkpoint
    parser.add_argument(
        "--transformer_checkpoint_path",
        default="/graphics/scratch2/staff/zakeri/train_logs/Transformer/flash_attention/with_optimized_latent_codes/full_dataset/overfitting/clean_code/regular_cat_fulldataset_alternative_test3_ABC_noEmpty/lightning_logs/version_1/checkpoints/saved/checkpoint-epoch=684-loss=0.000.ckpt",
        type=str,
    )
    # hparams for transformer
    parser.add_argument("--layers", default=20, type=int)  # layers: Number of transformer layers.
    parser.add_argument("--dim_size", default=512 * 4, type=int)  # Dimensionality of latent space in transformer.
    parser.add_argument("--heads", default=16, type=int)  # heads: Number of attention heads.
    parser.add_argument("--pre_trained", default=True, type=bool)
    parser.add_argument("--masking_ratio", default=0.40, type=float)

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    # write the checkpoints every 1000 steps
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="train_loss",
        filename="checkpoint-{epoch:03d}-{loss:.3f}",
        save_top_k=5,
        mode="min",
        verbose=True,
        every_n_train_steps=500,
    )
    lr_Monitor = LearningRateMonitor(logging_interval="step")
    model = TransformerSDFtoSDFABCOUTSIDE(
        latent_dim=args.latent_dim,
        resolution=args.resolution,
        target_resolution=args.target_resolution,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        train_lmdb_path=args.train_lmdb_path,
        val_lmdb_path=args.val_lmdb_path,
        obj_dir=args.obj_dir,
        value_range=args.value_range,
        vae_checkpoint_path=args.vae_checkpoint_path,
        marching_cube_result_dir=args.marching_cube_result_dir,
        layers=args.layers,
        dim_size=args.dim_size,
        heads=args.heads,
        pre_trained=args.pre_trained,
        masking_ratio=args.masking_ratio,
        transformer_checkpoint_path=args.transformer_checkpoint_path,
    )
    # configure the pytorch-lightning trainer.
    trainer = pl.Trainer.from_argparse_args(
        args,
        accelerator="gpu",
        devices=-1,
        num_nodes=1,
        strategy=DDPStrategy(process_group_backend="NCCl"),  # NCCL tends to be unreliable for some reason
        max_epochs=2000,
        log_every_n_steps=100,
        detect_anomaly=True,
        callbacks=[checkpoint_callback, lr_Monitor],
        val_check_interval=10000,
        check_val_every_n_epoch=None,
        default_root_dir="/graphics/scratch2/staff/zakeri/train_logs/Transformer/flash_attention/with_optimized_latent_codes/full_dataset/overfitting/clean_code/regular_cat_fulldataset_alternative_test3_ABC_custom_noEmpty/",
        # precision="bf16",
        # gradient_clip_val=0.5,
        #resume_from_checkpoint="/graphics/scratch2/staff/zakeri/train_logs/Transformer/flash_attention/with_optimized_latent_codes/full_dataset/overfitting/clean_code/regular_cat_fulldataset_alternative_test3_ABC_custom_noEmpty/lightning_logs/version_0/checkpoints/saved/checkpoint-epoch=881-loss=0.000.ckpt" # v1
        #resume_from_checkpoint="/graphics/scratch2/staff/zakeri/train_logs/Transformer/flash_attention/with_optimized_latent_codes/full_dataset/overfitting/clean_code/regular_cat_fulldataset_alternative_test3_ABC_custom_noEmpty/lightning_logs/version_1/checkpoints/checkpoint-epoch=884-loss=0.000.ckpt" #v2
        # resume_from_checkpoint="/graphics/scratch2/staff/zakeri/train_logs/Transformer/flash_attention/with_optimized_latent_codes/full_dataset/overfitting/clean_code/regular_cat_fulldataset_alternative_test3_ABC_custom_noEmpty/lightning_logs/version_2/checkpoints/checkpoint-epoch=889-loss=0.000.ckpt"
        resume_from_checkpoint="/graphics/scratch2/staff/zakeri/train_logs/Transformer/flash_attention/with_optimized_latent_codes/full_dataset/overfitting/clean_code/regular_cat_fulldataset_alternative_test3_ABC_custom_noEmpty/lightning_logs/version_3/checkpoints/checkpoint-epoch=892-loss=0.000.ckpt"
    )
    trainer.fit(model)
    print("CUDA_VISIBLE_DEVICES", os.environ["CUDA_VISIBLE_DEVICES"])
    # v0
    # v1 is the resume of v0
    # v2 is the resume of v1 so v3

if __name__ == "__main__":
    main()
