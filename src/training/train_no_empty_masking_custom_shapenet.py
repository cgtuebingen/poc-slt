from typing import Tuple, Any, Union
import torch
import numpy as np

from torch import nn
import pytorch_lightning as pl


from src.dataset.shapenet_train_dataset import (
    ShapeNetcorev1NormalizedTrainWithNonOptimizedLatentCodes,
)  # train

from src.dataset.shapenet_eval_dataset import (
    ShapeNetcorev1NormalizedValWithNonOptimizedLatentCodes,
)  # test

from src.p_vae.pvae import SDFtoSDF

from src.utils import transformer_visualizations as tv
from src.utils import loss_helper_fns as l_fn
from src.utils import plot_march_fns as pmt_fns
from src.utils import sub_voxel_related_fns as pp_fns
from src.utils.positional_encoder_class import MYPositionalEncoder3D

from src.utils import encoder_decoder_loading as ed
from src.utils import mask_huristic as gr_mask
from src.utils.helper_fns import concatenate_for_given_dim

from src.training.train_no_empty_masking_shapenet import (
    TransformerSDFtoSDFShapenetNormalized,
)


# --------------------------------------------------------------------------------------------------------------------------------------------------------
class TransformerSDFtoSDFShapenetNormalizedNoEmptyMaskingCustom(pl.LightningModule):
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
        mesh_path: str,
        value_range: int,
        vae_checkpoint_path: str,
        marching_cube_result_dir: str,
        layers: int,
        dim_size: int,
        heads: int,
        pre_trained: bool,
        masking_ratio: torch.float32,
        points_to_sample: int,
        query_number: int,
        examples_per_epoch: int,
        transformer_checkpoint_path: str,
        num_warmup_steps: int = 1000,
        num_training_steps: int = 1000000,
        **kwargs: dict  # gobble up unused parameters
    ):
        super(
            TransformerSDFtoSDFShapenetNormalizedNoEmptyMaskingCustom, self
        ).__init__()
        self.save_hyperparameters()

        if self.hparams.pre_trained:
            print("\n pre_trained: ", pre_trained)

            pre_trained_model = SDFtoSDF.load_from_checkpoint(
                vae_checkpoint_path,
            ).to(self.device)
            pre_trained_model.freeze()
            pre_trained_model.train(False)
            # del SDFtoSDF
            self.fdecoder = ed.load_decoder_from_checkpoint(
                pre_trained_model, latent_dim
            )
            self.fdecoder.to(self.device)

            # initialize the transformer from pretrained transformer trained with random-masking AND No empty masking
            # if you want to train from scratch , remove the initialization and take the model from scratch from no_empty_transformer or TransformerSDFtoSDFShapenetNormalized

            pre_trained_transformer_checkpoint = (
                TransformerSDFtoSDFShapenetNormalized.load_from_checkpoint(
                    transformer_checkpoint_path,
                    vae_checkpoint_path=vae_checkpoint_path,
                    transformer_checkpoint_path=transformer_checkpoint_path,
                    **{
                        "mesh_path": None,
                        "points_to_sample": None,
                        "query_number": None,
                        "examples_per_epoch": None,
                    },
                    map_location="cpu",
                    strict=False
                ).to(self.device)
            )
            self.regular_transformer = (
                pre_trained_transformer_checkpoint.regular_transformer
            )
            del pre_trained_transformer_checkpoint
            print(
                "\n regular transformer is initialized  from pretrained-transformer num_layer:",
                self.regular_transformer.num_layers,
            )

        self.l1_loss = nn.L1Loss(reduction="mean")

        number_of_sub_voxels = self.hparams.resolution // self.hparams.target_resolution
        self.number_of_sub_voxels = (
            number_of_sub_voxels * number_of_sub_voxels * number_of_sub_voxels
        )

        self.my_selected_indices = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
        ]  # only for the purpose of visualization and save time

        # generator = torch.Generator(device=self.device)
        # generator.manual_seed(123)
        self.penc_channels = 8 * self.hparams.latent_dim
        self.positional_encoder_3d = MYPositionalEncoder3D(self.penc_channels)

        self.constant_learnable_mask_tensors = torch.nn.Parameter(
            torch.randn(
                [self.hparams.latent_dim, 2, 2, 2],
                dtype=torch.float32,
                device=self.device,
            )
        )  # , generator=generator

        self.mapping_down = nn.Linear(
            self.penc_channels + 8 * self.hparams.latent_dim, self.hparams.dim_size
        )
        self.mapping_up = nn.Linear(self.hparams.dim_size, 8 * self.hparams.latent_dim)
        self.redundant_mapping = nn.Linear(
            8 * self.hparams.latent_dim, 8 * self.hparams.latent_dim
        )

    def calculate_losses(
        self,
        masked_bool: torch.Tensor,
        non_masked_bool: torch.Tensor,
        transformer_output_sequence: torch.Tensor,  # output
        optimized_latent_codes_reshaped: torch.Tensor,  # gt
    ) -> dict:
        all_losses = self.l1_loss(
            transformer_output_sequence, optimized_latent_codes_reshaped
        )

        l1_loss_masked = torch.tensor(0.0).to(device=self.device)
        if torch.any(masked_bool):
            l1_loss_masked = self.l1_loss(
                transformer_output_sequence[masked_bool],
                optimized_latent_codes_reshaped[masked_bool],
            )

        l1_loss_non_masked = torch.tensor(0.0).to(device=self.device)
        if torch.any(non_masked_bool):
            l1_loss_non_masked = self.l1_loss(
                transformer_output_sequence[non_masked_bool],
                optimized_latent_codes_reshaped[non_masked_bool],
            )

        loss_dict = {
            "l1_loss_masked": l1_loss_masked,
            "l1_loss_non_masked": l1_loss_non_masked,
        }
        # calculate scaled losses
        if True:
            transformer_output_sequence_shape = [
                transformer_output_sequence.shape[0],
                transformer_output_sequence.shape[1],
            ]
            l1_loss_scaled, l1_loss_masked_scaled, l1_loss_non_masked_scaled = (
                l_fn.scale_losses_noEmptymasking(
                    loss_dict,
                    masked_bool,
                    non_masked_bool,
                    transformer_output_sequence_shape,
                )
            )
            loss_dict_scaled = {
                "l1_loss_masked": l1_loss_masked_scaled,
                "l1_loss_non_masked": l1_loss_non_masked_scaled,
            }

        # weight losses
        if True:
            l1_loss_w, l1_loss_masked_w, l1_loss_non_masked_w = (
                l_fn.weight_losses_noEmptymasking(
                    loss_dict_scaled, weight_masked=1.0, weight_non_masked=1.0
                )
            )

        return {
            "l1_loss_masked": l1_loss_masked_w,
            "l1_loss_non_masked": l1_loss_non_masked_w,
            "l1_loss": l1_loss_w,
            "all_losses": all_losses,
        }

    def assign_constant_learnable_tensors(
        self, optimized_latent_codes: torch.Tensor, bool_dict: dict
    ) -> torch.Tensor:
        masked_optimized_latent_codes = optimized_latent_codes.clone()

        mask_all_bool = bool_dict["mask_all_bool"]

        masked_optimized_latent_codes[mask_all_bool] = (
            self.constant_learnable_mask_tensors
        )

        return masked_optimized_latent_codes.clone()

    def call_transformer_and_mapping_layers(
        self, transformer_input_sequence: torch.Tensor
    ) -> torch.Tensor:
        transformer_input_sequence = transformer_input_sequence.clone()

        transformer_input_sequence_down = self.mapping_down(transformer_input_sequence)
        transformer_output_sequence = self.regular_transformer(
            transformer_input_sequence_down
        )
        transformer_output_sequence_up = self.mapping_up(transformer_output_sequence)

        return transformer_output_sequence_up.clone()

    def forward(
        self,
        sub_voxels: torch.Tensor,
        non_optimized_latent_codes: torch.Tensor,
        mask_all_bool,
    ) -> Union[tuple[Any, Any, Any], Any]:
        # This part is the same for train and validation
        mask_all_bool = mask_all_bool.clone()
        batch_size = sub_voxels.shape[0]
        masked_non_optimized_latent_codes = self.assign_constant_learnable_tensors(
            non_optimized_latent_codes.clone(),
            bool_dict={"mask_all_bool": mask_all_bool},
        )
        # -----------------------------------------------------------------------------------------------------------------------------
        # Prep for passing the masked_optimized_latent_codes to transformer--------------------------------------------------------
        # masked_optimized_latent_codes size : [B, SeqLen, 512, 2, 2, 2] --> [B, 64, 512, 2, 2, 2] --> reshape: [B, 64, 4096]
        masked_non_optimized_non_latent_codes_reshaped = (
            masked_non_optimized_latent_codes.reshape(
                [batch_size, self.number_of_sub_voxels, 8 * self.hparams.latent_dim]
            )
        )
        masked_non_optimized_non_latent_codes_reshaped_mapped = self.redundant_mapping(
            masked_non_optimized_non_latent_codes_reshaped
        )
        assert (
            masked_non_optimized_non_latent_codes_reshaped.shape
            == masked_non_optimized_non_latent_codes_reshaped_mapped.shape
        )
        z_positionally_encoded_re = self.positional_encoder_3d(
            shape_of_positions=[batch_size, 4, 4, 4, self.penc_channels]
        )
        assert (
            z_positionally_encoded_re.shape
            == masked_non_optimized_non_latent_codes_reshaped_mapped.shape
        )
        transformer_input_sequence = concatenate_for_given_dim(
            z_positionally_encoded_re,
            masked_non_optimized_non_latent_codes_reshaped_mapped,
            cat_dim=2,
        )
        transformer_output_sequence = self.call_transformer_and_mapping_layers(
            transformer_input_sequence
        )

        return (transformer_output_sequence, masked_non_optimized_latent_codes)

    def generate_all_masking(
        self, sub_voxels: torch.Tensor, object_indices: torch.Tensor, mesh_file_name
    ):
        batch_size = object_indices.shape[0]
        masking_choice = np.random.choice(32, 1)

        empty_sub_voxels_bool, non_empty_sub_voxels_bool = (
            pp_fns.extract_outside_and_non_outside_voxels(
                sub_voxels.clone(),
                self.number_of_sub_voxels,
                self.hparams.target_resolution,
            )
        )
        if not torch.all(torch.any(non_empty_sub_voxels_bool, dim=1)):
            pair = {
                "object_index": object_indices.item(),
                "mesh_file_name": mesh_file_name,
            }
            print("\nall voxels are empty", pair)
            breakpoint()

        mask_all_bool = gr_mask.mask_heuristic(
            masking_choice,
            self.hparams.masking_ratio,
            batch_size,
            self.number_of_sub_voxels,
            self.hparams.target_resolution,
            self.device,
        )

        # just for return
        masked_bool = mask_all_bool.clone()
        non_masked_bool = torch.logical_and(
            non_empty_sub_voxels_bool, torch.logical_not(mask_all_bool)
        )
        return (mask_all_bool, masked_bool, non_masked_bool)

    def fwd(self, batch: list, train: bool) -> Tuple[dict, Tuple]:
        if train:
            (
                object_indices,
                mesh_file_name,
                gt_sdf_full_voxel,
                std,
                var,
                non_optimized_latent_codes,
                folder_name,
                sub_folder_name,
            ) = batch

            batch_size = object_indices.shape[0]
        else:
            (
                object_indices,
                mesh_file_name,
                gt_sdf_full_voxel,
                std,
                var,
                non_optimized_latent_codes,
                folder_name,
                sub_folder_name,
            ) = batch
            batch_size = object_indices.shape[0]

        # -----------
        sub_voxels = pp_fns.sub_divide_gt_and_normalize(
            gt_sdf_full_voxel.clone(),
            self.number_of_sub_voxels,
            self.hparams.target_resolution,
        )
        mask_all_bool, masked_bool, non_masked_bool = self.generate_all_masking(
            sub_voxels, object_indices, mesh_file_name
        )

        # FORWARD CAlL----------------------------------------------------------------------------------------------------------------------------
        transformer_output_sequence_up, masked_non_optimized_latent_codes = (
            self.forward(sub_voxels, non_optimized_latent_codes, mask_all_bool)
        )
        # for loss calculation
        non_optimized_latent_codes_reshaped = non_optimized_latent_codes.reshape(
            [batch_size, self.number_of_sub_voxels, 8 * self.hparams.latent_dim]
        )
        assert (
            transformer_output_sequence_up.shape
            == non_optimized_latent_codes_reshaped.shape
        )

        # calculate losses:
        loss_dict = self.calculate_losses(
            masked_bool,
            non_masked_bool,
            transformer_output_sequence_up,
            non_optimized_latent_codes_reshaped,
        )
        loss = loss_dict["l1_loss"]

        # log losses
        if train:
            loss_log = l_fn.create_log_losses_for_given_dict(loss_dict, stage="train")

            self.log_dict(loss_log, batch_size=self.hparams.batch_size, sync_dist=True)
            self.log(
                "train_loss", loss, batch_size=self.hparams.batch_size, sync_dist=True
            )

        else:
            loss_log = l_fn.create_log_losses_for_given_dict(loss_dict, stage="val")

            self.log_dict(
                loss_log, batch_size=self.hparams.val_batch_size, sync_dist=True
            )
            self.log(
                "val_loss", loss, batch_size=self.hparams.val_batch_size, sync_dist=True
            )

        loss_dict = {"loss": loss}

        return (
            loss_dict,
            (
                non_optimized_latent_codes_reshaped,
                masked_non_optimized_latent_codes,
                transformer_output_sequence_up,
                sub_voxels,
            ),
        )

    def training_step(self, batch: list, batch_idx: int) -> dict:
        # (
        #     object_indices, mesh_file_name, gt_sdf_full_voxel, std, var, non_optimized_latent_codes, folder_name, sub_folder_name,
        # ) = batch
        loss_dict, stuff = self.fwd(batch, True)

        # masked_non_optimized_latent_codes, transformer_output_sequence_up, sub_voxels = stuff
        return loss_dict

    def validation_step(self, batch: list, batch_idx: int) -> None:
        self.fdecoder.eval()
        assert not self.fdecoder.batchNorm3d5.track_running_stats
        assert not self.fdecoder.batchNorm3d6.track_running_stats
        assert not self.fdecoder.batchnorm3d7.track_running_stats
        assert not self.fdecoder.training

        (
            object_indices,
            mesh_file_name,
            gt_sdf_full_voxel,
            std,
            var,
            non_optimized_latent_codes,
            folder_name,
            sub_folder_name,
        ) = batch

        batch_size = object_indices.shape[0]
        assert object_indices.shape == (self.hparams.val_batch_size,)
        loss_dict, stuff = self.fwd(batch, False)

        (
            non_optimized_latent_codes_reshaped,
            masked_non_optimized_latent_codes,
            transformer_output_sequence_up,
            sub_voxels,
        ) = stuff

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

        self.plot_march_and_login_tensorboard(
            dict_arguments_for_vis,
            dict_arguments_of_variables,
            object_indices,
            batch_size,
        )

    def plot_march_and_login_tensorboard(
        self,
        dict_arguments_for_vis: dict,
        dict_arguments_of_variables: dict,
        object_indices: torch.Tensor,
        batch_size: int,
    ) -> None:
        data_dict_for_vis = pmt_fns.generate_data_for_plottingv2(
            dict_arguments_for_vis, dict_arguments_of_variables, self.fdecoder
        )
        for b in range(batch_size):
            selected_index = object_indices[b].detach().cpu().item()
            if selected_index in self.my_selected_indices:
                collected_data_dict_for_plotting = (
                    pmt_fns.collect_generated_data_for_plottingv3(
                        data_dict_for_vis, self.hparams.resolution, batch_idx=b
                    )
                )
                plots = tv.generate_plot_for_given_dict_of_items(
                    collected_data_dict_for_plotting,
                    self.hparams.resolution,
                    number_of_slices=2,
                    plot_scale_factor=2,
                    plot_range=[-2, 2],
                )
                self.login_to_tensorboard(plots, selected_index, number_of_slices=2)

    def login_to_tensorboard(
        self, plots: list, selected_index: int, number_of_slices: int
    ):
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
        self.train_dataset = ShapeNetcorev1NormalizedTrainWithNonOptimizedLatentCodes(
            self.hparams.mesh_path,
            self.hparams.points_to_sample,
            self.hparams.query_number,
            self.hparams.train_lmdb_path,
            self.hparams.value_range,
            self.hparams.resolution,
            self.hparams.examples_per_epoch,
        )

        print("\n setup: train_dataset len: ", len(self.train_dataset))

        # by mistake, we called the eval/test dataset as val dataset, while actual validation dataset is the first 100 objects in train dataset
        # that is used to evaluate the performance of training like below.
        # self.val_dataset = ShapeNetcorev1NormalizedTrainWithNonOptimizedLatentCodes(
        #     self.hparams.mesh_path, self.hparams.points_to_sample, self.hparams.query_number, self.hparams.train_lmdb_path, self.hparams.value_range, self.hparams.resolution, self.hparams.examples_per_epoch
        # )
        # self.val_dataset.len = 100
        self.val_dataset = ShapeNetcorev1NormalizedValWithNonOptimizedLatentCodes(
            self.hparams.mesh_path,
            self.hparams.points_to_sample,
            self.hparams.query_number,
            self.hparams.val_lmdb_path,
            self.hparams.value_range,
            self.hparams.resolution,
            self.hparams.examples_per_epoch,
        )

        print("\n setup: val_dataset len: ", len(self.val_dataset))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=30,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.hparams.val_batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

    def configure_optimizers(self):
        # we exclude fdecoer params from optimizer.
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

        # if you take the script as it is with initialization, you do not need below warm-up,
        # if you decide to train from very scratch and not use the model initialization, then uncomment below section.

        # num_gpus = 3
        # num_train_steps = len(self.train_dataset) // (self.hparams.batch_size * num_gpus) * self.trainer.max_epochs
        # print("\n num_train_steps: ", num_train_steps)
        # num_warmup_steps = int(self.hparams.warmup_ratio * num_train_steps)
        # print("\n num_warmup_steps: ", num_warmup_steps)
        #
        # lr_scheduler = {
        #     "scheduler": get_cosine_schedule_with_warmup(
        #         optimizer,
        #         num_warmup_steps=num_warmup_steps,
        #         num_training_steps=num_train_steps,
        #         num_cycles=0.5,
        #     ),
        #     "interval": "step",
        #     "frequency": 1,
        # }
        return [optimizer]  # , [lr_scheduler]
