import os
import sys

sys.path.append("....")

import torch
from torch import nn
import pytorch_lightning as pl
import numpy as np
import src.p_vae.distribution
import src.p_vae.vae_loss
from src.utils.plot_voxel import plot_v

import src.p_vae.pvae_model as sc
import src.p_vae.pvae_dataset as dt_new

from torch.optim.lr_scheduler import CosineAnnealingLR
import src.p_vae.normalizations as norm


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class SDFtoSDF(pl.LightningModule):
    def __init__(
        self,
        latent_dim: int,
        target_resolution,
        resolution: int,
        batch_size: int,
        learning_rate: float,
        train_val_dict_path: str,
        lmdb_path: str,
        mesh_path: str,
        points_to_sample: int,
        query_number: int,
        value_range: int,
        examples_per_epoch: int,
        marching_cube_result_dir: str,
    ):
        super(SDFtoSDF, self).__init__()
        self.save_hyperparameters()

        self.encoder = sc.VSEncoder(latent_dim)
        # self.dropout = nn.Dropout(0.1)  # change this, help to prevent from over fitting
        self.decoder = sc.VSDecoder(latent_dim)  # 2 * 2 * 2

        self.l1 = nn.L1Loss()
        self.vae_loss = src.p_vae.vae_loss.VAELoss()

    def forward(self, x_in: torch.Tensor, mode: str) -> torch.Tensor:
        x_encoded = self.encoder(x_in)
        x_encoded_reshaped = x_encoded.view(
            self.hparams.batch_size, 2 * self.hparams.latent_dim, 2, 2, 2
        )

        B, Ch, D, H, W = x_encoded_reshaped.shape

        x_encoded_p = x_encoded_reshaped.permute([0, 2, 3, 4, 1])
        x_encoded_reshaped = x_encoded_p.reshape([B * D * H * W, Ch])

        self.posterior = p_vae.distribution.Distribution(x_encoded_reshaped)
        if mode == "training":
            x_distribution_sample = self.posterior.sample()
            B_new, Ch_new = x_distribution_sample.shape
            x_distribution_sample_reshaped = x_distribution_sample.reshape(
                [B, Ch_new, D, H, W]
            )
            y = self.decoder(x_distribution_sample_reshaped)
            return y
        elif mode == "val":
            x_distribution_sample = self.posterior.mode()
            B_new, Ch_new = x_distribution_sample.shape
            x_distribution_sample_reshaped = x_distribution_sample.reshape(
                [B, Ch_new, D, H, W]
            )
            y = self.decoder(x_distribution_sample_reshaped)
            return y

    def training_step(self, batch: list, batch_idx: int) -> dict:
        (
            object_indicis,
            mesh_file_name,
            gt_sdf_voxels_transformed,
            distance_per_voxels,
            left,
            x_offset,
            top,
            y_offset,
            front,
            z_offset_copy,
            folder,
            label,
            dataset_index,
            sub_folder_index,
        ) = batch

        # normalization to scale everything to 1
        normalized_gt_sdf_voxels_transformed = (
            norm.normalize_voxels_to_their_distance_with_batch(
                gt_sdf_voxels_transformed,
                distance_per_voxels,
                self.hparams.batch_size,
                self.hparams.target_resolution,
            ).to(self.device)
        )
        normalized_gt_sdf_voxels_transformed_reshaped = (
            normalized_gt_sdf_voxels_transformed
        ).unsqueeze(
            1
        )  # FIXME

        predicted = self.forward(
            normalized_gt_sdf_voxels_transformed_reshaped, mode="training"
        )

        loss, loss_log_dict = self.vae_loss(
            predicted,
            normalized_gt_sdf_voxels_transformed_reshaped,
            self.posterior,
            self.global_step,
        )

        self.log("train_loss", loss, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log_dict(loss_log_dict, batch_size=self.hparams.batch_size, sync_dist=True)

        return {"loss": loss}

    def validation_step(self, batch: list, batch_idx: int) -> None:
        (
            object_indicis,
            mesh_file_name,
            gt_sdf_voxels_transformed,
            distance_per_voxels,
            left,
            x_offset,
            top,
            y_offset,
            front,
            z_offset_copy,
            folder,
            label,
            dataset_index,
            sub_folder_index,
        ) = batch

        # normalization to scale everything to 1
        normalized_gt_sdf_voxels_transformed = (
            norm.normalize_voxels_to_their_distance_with_batch(
                gt_sdf_voxels_transformed,
                distance_per_voxels,
                self.hparams.batch_size,
                self.hparams.target_resolution,
            ).to(self.device)
        )
        normalized_gt_sdf_voxels_transformed_reshaped = (
            normalized_gt_sdf_voxels_transformed
        ).unsqueeze(
            1
        )  # FIXME

        predicted = self.forward(
            normalized_gt_sdf_voxels_transformed_reshaped, mode="val"
        )

        loss, loss_log_dict = self.vae_loss(
            predicted,
            normalized_gt_sdf_voxels_transformed_reshaped,
            self.posterior,
            self.global_step,
        )
        # loss_per_voxel = self.l2_per_batch(predicted, normalized_gt_sdf_voxels_transformed_reshaped)

        self.log("val_loss", loss, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log_dict(loss_log_dict, batch_size=self.hparams.batch_size, sync_dist=True)

        # # plotting
        number_of_slices = 1
        # make a  plot from slices of gt sdf for transformed voxel grid
        for b in range(0, self.hparams.batch_size, 2):
            my_selected_meshes = [17, 13, 24, 31]

            selected_index = object_indicis[b].detach()

            if selected_index in my_selected_meshes:
                gt_sdf_voxel = normalized_gt_sdf_voxels_transformed_reshaped[
                    b, :, :, :, :
                ]
                gt_sdf_voxel_ = gt_sdf_voxel.squeeze(0)
                gt_sdf_voxel_ = np.asarray(gt_sdf_voxel_.cpu()).astype(float)

                (
                    gt_sdf_voxel_transformed_image_list,
                    gt_sdf_voxel_transformed_title_list,
                ) = plot_v(
                    gt_sdf_voxel_,
                    number_of_slices,
                    self.hparams.target_resolution,
                    "gt_sdf_voxel_transformed",
                    plot_range=[-1, 1],
                )
                # ----
                predicted_ = predicted[b, :, :, :, :]
                predicted_ = predicted_.squeeze(0)
                predicted_ = np.asarray(predicted_.cpu()).astype(float)

                # marching cube
                # m_cube_fns.make_mcubes_from_voxels(predicted_, self.current_epoch, self.global_step, b, '-predicted-', self.hparams.marching_cube_result_dir)
                # m_cube_fns.make_mcubes_from_voxels(gt_sdf_voxel_, self.current_epoch, self.global_step, b, '-gt-', self.hparams.marching_cube_result_dir)

                predicted_image_list, predicted_title_list = plot_v(
                    predicted_,
                    number_of_slices,
                    self.hparams.target_resolution,
                    "sdf_voxel_reconstructed",
                    plot_range=[-1, 1],
                )

                # ----
                diff = abs(np.subtract(predicted_, gt_sdf_voxel_))
                diff_ = np.asarray(diff).astype(float)
                diff_image_list, diff_title_list = plot_v(
                    diff_,
                    number_of_slices,
                    self.hparams.target_resolution,
                    "diff",
                    plot_range=[-1, 1],
                )
                # ----
                del predicted_
                del gt_sdf_voxel_

                for sl in range(0, number_of_slices, 1):
                    gt_sdf_voxel_slice = gt_sdf_voxel_transformed_image_list[sl]

                    predicted_image_slice = predicted_image_list[sl]

                    image_re = torch.as_tensor(predicted_image_slice[:, :, :3]).permute(
                        [2, 0, 1]
                    )
                    del predicted_image_slice

                    image_gt = torch.as_tensor(gt_sdf_voxel_slice[:, :, :3]).permute(
                        [2, 0, 1]
                    )
                    del gt_sdf_voxel_slice
                    diff_image_slice = diff_image_list[sl]
                    image_diff = torch.as_tensor(diff_image_slice[:, :, :3]).permute(
                        [2, 0, 1]
                    )
                    del diff_image_slice
                    plot = torch.cat(
                        [
                            image_re.to(device=self.device),
                            image_gt.to(device=self.device),
                            image_diff.to(device=self.device),
                        ],
                        -1,
                    )
                    del image_re
                    del image_gt
                    del image_diff
                    # show in tensorboard
                    self.logger.experiment.add_image(
                        "mesh-Id{}_slice{}".format(selected_index, sl),
                        plot,
                        self.global_step,
                    )

    def setup(self, stage: str) -> None:
        train_dict_path = os.path.join(
            self.hparams.train_val_dict_path, "train_dataset_dict"
        )
        train_dict = torch.load(train_dict_path)

        val_dict_path = os.path.join(
            self.hparams.train_val_dict_path, "val_dataset_dict"
        )
        val_dict = torch.load(val_dict_path)

        val_lmdb_path = os.path.join(self.hparams.lmdb_path, "shapenet_val")
        train_lmdb_path = os.path.join(self.hparams.lmdb_path, "shapenet_train")

        # setup data from dataset class
        self.val_dataset = dt_new.ReadLMDBDataset(
            val_dict,
            self.hparams.mesh_path,
            self.hparams.target_resolution,
            self.hparams.points_to_sample,
            self.hparams.query_number,
            val_lmdb_path,
            self.hparams.value_range,
            self.hparams.resolution,
            self.hparams.examples_per_epoch,
        )
        print("\n setup: val_dataset len: ", len(self.val_dataset))
        self.train_dataset = dt_new.ReadLMDBDataset(
            train_dict,
            self.hparams.mesh_path,
            self.hparams.target_resolution,
            self.hparams.points_to_sample,
            self.hparams.query_number,
            train_lmdb_path,
            self.hparams.value_range,
            self.hparams.resolution,
            self.hparams.examples_per_epoch,
        )

        print("\n setup: train_dataset len: ", len(self.train_dataset))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=False,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=False,
            drop_last=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        max_steps = int(
            self.trainer.max_epochs
            * len(self.train_dataset)
            / (self.hparams.batch_size * 2)
        )
        print("\nmax_steps", max_steps)
        scheduler = {
            "scheduler": CosineAnnealingLR(
                optimizer, T_max=max_steps, eta_min=0, last_epoch=-1
            )
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
