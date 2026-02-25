import sys
import os
#
# if __name__ == "__main__":
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
import torch

print("torch.cuda.device_count()", torch.cuda.device_count())
print("torch.cuda.nccl.version()", torch.cuda.nccl.version())
torch.cuda.empty_cache()
torch.multiprocessing.set_sharing_strategy("file_system")
from pytorch_lightning.strategies import DDPStrategy
from torch import nn
import pytorch_lightning as pl
import argparse
import numpy as np

sys.path.append("//")

from Networks import distribution
from Loss import vae_loss

from Augmentation.Augmentation_Visualization import voxel_plot  # , create_scatter_plot

from Networks import skip_connection_5_2x2x2_noBNDecoder as sc
from Dataset import Dataset_Class_32stream_sdf_plus_pc_shapenetcorev2_excluding_shapenetcrev1_validation_split as dt_new

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
# from Visualization import m_cube_fns
from Helpers import normalizations as norm
from pytorch_lightning.callbacks import LearningRateMonitor

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

        self.encoder = sc.VSEncoderv9(latent_dim)
        # self.dropout = nn.Dropout(0.1)  # change this, help to prevent from over fitting
        self.decoder = sc.VSDecoderv9(latent_dim)  # 2 * 2 * 2

        self.l1 = nn.L1Loss()
        # FIXME add VAE loss here!
        self.vae_loss = vae_loss.VAELoss()

        # self.l2_per_batch = vae_loss.LossPerVoxel()  # only L1

    def forward(self, x_in: torch.Tensor, mode: str) -> torch.Tensor:
        x_encoded = self.encoder(x_in)
        # print("\n x_encoded: ", x_encoded.shape)
        x_encoded_reshaped = x_encoded.view(self.hparams.batch_size, 2 * self.hparams.latent_dim, 2, 2, 2)

        B, Ch, D, H, W = x_encoded_reshaped.shape

        x_encoded_p = x_encoded_reshaped.permute([0, 2, 3, 4, 1])
        x_encoded_reshaped = x_encoded_p.reshape([B * D * H * W, Ch])

        # x_encoded_re = einops.rearrange("B Ch D H W" -> "(B D H W) Ch", x_encoded_reshaped)
        # print("\n x_encoded_reshaped: ", x_encoded_reshaped.shape)

        self.posterior = distribution.Distribution(x_encoded_reshaped)
        if mode == "training":
            x_distribution_sample = self.posterior.sample()
            # print("\n x_distribution_sample: ", x_distribution_sample.shape)
            B_new, Ch_new = x_distribution_sample.shape
            x_distribution_sample_reshaped = x_distribution_sample.reshape([B, Ch_new, D, H, W])
            # print("\n x_distribution_sample_reshaped: ", x_distribution_sample_reshaped.shape)
            y = self.decoder(x_distribution_sample_reshaped)
            return y
        elif mode == "val":
            x_distribution_sample = self.posterior.mode()
            # print("\n x_distribution_sample: ", x_distribution_sample.shape)
            B_new, Ch_new = x_distribution_sample.shape
            x_distribution_sample_reshaped = x_distribution_sample.reshape([B, Ch_new, D, H, W])
            # print("\n x_distribution_sample_reshaped: ", x_distribution_sample_reshaped.shape)
            y = self.decoder(x_distribution_sample_reshaped)
            return y

    def training_step(self, batch: list, batch_idx: int) -> dict:
        object_indicis, mesh_file_name, gt_sdf_voxels_transformed, distance_per_voxels, left, x_offset, top, y_offset, front, z_offset_copy, folder, label, dataset_index, sub_folder_index = batch  # run some tests:
        # run some tests:
        # if (len(gt_sdf_voxels_transformed.shape) < 0 or len(gt_sdf_voxels_transformed.shape) == 0):
        #     raise "\n training, invalid gt_sdf_voxels_transformed dim<0"
        # if (gt_sdf_voxels_transformed is None or len(gt_sdf_voxels_transformed.shape) is None):
        #     raise "\n training, invalid gt_sdf_voxels_transformed dim is None"

        # self.logger.experiment.add_histogram('GT SDF Voxel', gt_sdf_voxels_transformed, self.current_epoch, bins='auto')

        # normalization to scale everything to 1
        normalized_gt_sdf_voxels_transformed = norm.normalize_voxels_to_their_distance_with_batch(gt_sdf_voxels_transformed, distance_per_voxels, self.hparams.batch_size, self.hparams.target_resolution).to(self.device)
        # self.logger.experiment.add_histogram('Normalized GT SDF VOXEl', normalized_gt_sdf_voxels_transformed, self.current_epoch, bins='auto')
        normalized_gt_sdf_voxels_transformed_reshaped = (normalized_gt_sdf_voxels_transformed).unsqueeze(1)  # FIXME

        predicted = self.forward(normalized_gt_sdf_voxels_transformed_reshaped, mode="training")

        loss, loss_log_dict = self.vae_loss(predicted, normalized_gt_sdf_voxels_transformed_reshaped, self.posterior, self.global_step)

        self.log("train_loss", loss, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log_dict(loss_log_dict, batch_size=self.hparams.batch_size, sync_dist=True)

        return {"loss": loss}

    def validation_step(self, batch: list, batch_idx: int) -> None:
        object_indicis, mesh_file_name, gt_sdf_voxels_transformed, distance_per_voxels, left, x_offset, top, y_offset, front, z_offset_copy, folder, label, dataset_index, sub_folder_index = batch  # run some tests:
        # if (len(gt_sdf_voxels_transformed.shape) < 0 or len(gt_sdf_voxels_transformed.shape) == 0):
        #     raise "\n validation, invalid gt_sdf_voxels_transformed dim < 0"
        # if (gt_sdf_voxels_transformed is None or len(gt_sdf_voxels_transformed.shape) is None):
        #     raise "\n validation, invalid gt_sdf_voxels_transformed dim is None"

        # normalization to scale everything to 1
        normalized_gt_sdf_voxels_transformed = norm.normalize_voxels_to_their_distance_with_batch(gt_sdf_voxels_transformed, distance_per_voxels, self.hparams.batch_size, self.hparams.target_resolution).to(self.device)
        normalized_gt_sdf_voxels_transformed_reshaped = (normalized_gt_sdf_voxels_transformed).unsqueeze(1)  # FIXME

        predicted = self.forward(normalized_gt_sdf_voxels_transformed_reshaped, mode="val")

        loss, loss_log_dict = self.vae_loss(predicted, normalized_gt_sdf_voxels_transformed_reshaped, self.posterior, self.global_step)
        # loss_per_voxel = self.l2_per_batch(predicted, normalized_gt_sdf_voxels_transformed_reshaped)

        self.log("val_loss", loss, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log_dict(loss_log_dict, batch_size=self.hparams.batch_size, sync_dist=True)

        # # plotting
        number_of_slices = 1
        # make a  plot from slices of gt sdf for transformed voxel grid
        for b in range(0, self.hparams.batch_size, 2):
            # my_selected_meshes = [17, 13, 12, 24, 31, 36, 41, 43, 44, 45, 53, 56, 58]
            my_selected_meshes = [17, 13, 24, 31]

            selected_index = object_indicis[b].detach()
            # print("\n validation_epoch_end selected_index:", selected_index)

            if selected_index in my_selected_meshes:
                # print("\n validation_epoch_end selected_index:", selected_index)
                gt_sdf_voxel = normalized_gt_sdf_voxels_transformed_reshaped[b, :, :, :, :]
                # print("\n validation_epoch_end gt_sdf_voxel:", gt_sdf_voxel.shape)
                gt_sdf_voxel_ = gt_sdf_voxel.squeeze(0)
                # gt_sdf_voxel_cpu = gt_sdf_voxel_.cpu()
                # print("\n validation_epoch_end gt_sdf_voxel_:", gt_sdf_voxel_.shape)
                gt_sdf_voxel_ = np.asarray(gt_sdf_voxel_.cpu()).astype(float)
                # gt_sdf_voxel_copy = np.array(gt_sdf_voxel_cpu, copy=True, dtype=float)
                # print("\n validation_epoch_end min gt_sdf_voxel_ :", np.min(gt_sdf_voxel_), "max: ", np.max(gt_sdf_voxel_))
                gt_sdf_voxel_transformed_image_list, gt_sdf_voxel_transformed_title_list = voxel_plot(gt_sdf_voxel_, number_of_slices, self.hparams.target_resolution, "gt_sdf_voxel_transformed", plot_range=1)
                # print("\n validation_epoch_end gt_sdf_voxel_transformed_image_list:", type(gt_sdf_voxel_transformed_image_list[0].shape), ", len: ", len(gt_sdf_voxel_transformed_image_list))
                # print("\n validation_epoch_end gt_sdf_voxel_transformed_title_list:", type(gt_sdf_voxel_transformed_title_list[0]), ", len: ", len(gt_sdf_voxel_transformed_title_list))
                # ----
                predicted_ = predicted[b, :, :, :, :]
                predicted_ = predicted_.squeeze(0)
                # predicted_cpu = predicted_.cpu()
                # predicted_copy = np.array(predicted_cpu, copy=True, dtype=float)
                predicted_ = np.asarray(predicted_.cpu()).astype(float)

                # marching cube
                # m_cube_fns.make_mcubes_from_voxels(predicted_, self.current_epoch, self.global_step, b, '-predicted-', self.hparams.marching_cube_result_dir)
                # m_cube_fns.make_mcubes_from_voxels(gt_sdf_voxel_, self.current_epoch, self.global_step, b, '-gt-', self.hparams.marching_cube_result_dir)

                # print("\n validation_epoch_end min predicted_ :", np.min(predicted_), "max: ", np.max(predicted_))
                predicted_image_list, predicted_title_list = voxel_plot(predicted_, number_of_slices, self.hparams.target_resolution, "sdf_voxel_reconstructed", plot_range=1)
                # print("\n validation_epoch_end predicted_ predicted_image_list:", type(predicted_image_list[0].shape), ", len: ", len(predicted_image_list))
                # print("\n validation_epoch_end predicted_ predicted_title_list:", type(predicted_title_list[0]), ", len: ", len(predicted_title_list))

                # ----
                diff = abs(np.subtract(predicted_, gt_sdf_voxel_))
                diff_ = np.asarray(diff).astype(float)
                diff_image_list, diff_title_list = voxel_plot(diff_, number_of_slices, self.hparams.target_resolution, "diff", plot_range=1)
                # ----
                del predicted_
                del gt_sdf_voxel_

                for sl in range(0, number_of_slices, 1):
                    gt_sdf_voxel_slice = gt_sdf_voxel_transformed_image_list[sl]
                    # print("\n validation_epoch_end gt_sdf_voxel_slice type:", type(gt_sdf_voxel_slice), ", shape:", gt_sdf_voxel_slice.shape)

                    predicted_image_slice = predicted_image_list[sl]
                    # print("\n validation_epoch_end y_image_slice type:", type(y_image_slice), ", shape:", y_image_slice.shape)

                    image_re = torch.as_tensor(predicted_image_slice[:, :, :3]).permute([2, 0, 1])
                    # print("\n validation_epoch_end image_re type:", type(image_re), ", shape:", image_re.shape)
                    del predicted_image_slice

                    image_gt = torch.as_tensor(gt_sdf_voxel_slice[:, :, :3]).permute([2, 0, 1])
                    # print("\n validation_epoch_end image_gt type:", type(image_gt), ", shape:", image_gt.shape)
                    del gt_sdf_voxel_slice
                    diff_image_slice = diff_image_list[sl]
                    image_diff = torch.as_tensor(diff_image_slice[:, :, :3]).permute([2, 0, 1])
                    del diff_image_slice
                    plot = torch.cat([image_re.cuda(), image_gt.cuda(), image_diff.cuda()], -1)
                    # print("\n validation_epoch_end plot type:", type(plot), ", shape:", plot.shape)
                    del image_re
                    del image_gt
                    del image_diff
                    # show in tensorboard
                    self.logger.experiment.add_image("mesh-Id{}_slice{}".format(selected_index, sl), plot, self.global_step)
        # return {"loss_per_voxel": loss_per_voxel, "distance_per_voxels": distance_per_voxels}

    # def validation_epoch_end(self, outputs) -> None:
    #     # print(outputs['loss_per_voxel'])
    #
    #     loss_per_voxel_tensor = torch.cat([x['loss_per_voxel'] for x in outputs], dim=0)
    #     distance_per_voxels_tensor = torch.cat([y['distance_per_voxels'] for y in outputs], dim=0)
    #
    #     points = torch.cat([loss_per_voxel_tensor.to(self.device), distance_per_voxels_tensor.to(self.device)], dim=1)
    #
    #     loss_vs_distance_per_voxel_plot = create_scatter_plot(points)
    #     # plot = torch.as_tensor(loss_vs_distance_per_voxel_plot[:, :, :3])
    #     # plot_ = plot.permute(2, 0, 1)
    #     # .detach().cpu().numpy()
    #     self.logger.experiment.add_figure(
    #         "epoch{}".format(self.current_epoch), loss_vs_distance_per_voxel_plot, self.global_step
    #     )

    def setup(self, stage: str) -> None:
        train_dict_path = os.path.join(self.hparams.train_val_dict_path, "train_dataset_dict")
        train_dict = torch.load(train_dict_path)

        val_dict_path = os.path.join(self.hparams.train_val_dict_path, "val_dataset_dict")
        val_dict = torch.load(val_dict_path)

        val_lmdb_path = os.path.join(self.hparams.lmdb_path, "shapenetcorev2Excludingcorev1validation_SDF_SpanningMultiResVoxelPLUSPC32_64_val")
        # print("\n setup, val_lmdb_path: ", val_lmdb_path)
        train_lmdb_path = os.path.join(self.hparams.lmdb_path, "shapenetcorev2Excludingcorev1validation_SDF_SpanningMultiResVoxelPLUSPC32_64_train")
        # print("\n setup, train_lmdb_path: ", train_lmdb_path)

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
        # self.val_dataset.len = 200
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
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=8, pin_memory=False, drop_last=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=4, pin_memory=False, drop_last=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        max_steps = int(self.trainer.max_epochs * len(self.train_dataset) / (self.hparams.batch_size * 2))
        print("\nmax_steps", max_steps)
        scheduler = {"scheduler": CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=0, last_epoch=-1, verbose=True)}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dim", default=512, type=int)
    parser.add_argument("--resolution", default=128, type=int)
    parser.add_argument("--target_resolution", default=32, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)

    parser.add_argument("--train_val_dict_path", default="/graphics/scratch2/staff/zakeri/LMDBs/ShapeNetCorev2_remeshed_0.008_train_val_dictionaries/", type=str)
    # parser.add_argument('--lmdb_path', default="/graphics/scratch2/staff/zakeri/LMDBs/shapenetcorev2_SDF_SpanningMultiResVoxelPLUSPC32_64/", type=str) # old training with shapnetcorev2

    # on Hercaleum:
    # parser.add_argument("--lmdb_path", default="/scratch/zakeri/shapenetcorev2Excludingcorev1validation_SDF_SpanningMultiResVoxelPLUSPC32_64/", type=str)
    parser.add_argument("--lmdb_path", default="/graphics/scratch2/staff/zakeri/LMDBs/shapenetcorev2Excludingcorev1validation_SDF_SpanningMultiResVoxelPLUSPC32_64/", type=str)

    parser.add_argument("--mesh_path", default="/graphics/scratch2/staff/zakeri/LMDBs/ShapeNetCorev2_remeshed_0.008/ShapeNetCore.v2/", type=str)
    parser.add_argument(
        "--marching_cube_result_dir",
        default="/graphics/scratch2/staff/zakeri/train_logs/VAE/skip_connection/v403_64_2x2x2_noBNDecoder_shapenetcorev2_excluding_shapenetcorev1_validation_split/marching_cube_results_0/",
        type=str,
    )

    parser.add_argument("--points_to_sample", default=64, type=int)
    parser.add_argument("--query_number", default=1000, type=int)
    parser.add_argument("--value_range", default=1, type=int)
    parser.add_argument("--examples_per_epoch", default=1000, type=int)

    parser = pl.Trainer.add_argparse_args(parser)
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
    # print("\n main, model: ", model)
    # configure the pytorch-lightning trainer.
    trainer = pl.Trainer.from_argparse_args(
        args,
        accelerator="gpu",
        devices=-1,
        # strategy="ddp",  # NCCL tends to be unreliable for some reason
        # strategy=DDPStrategy(process_group_backend="NCCL"),  # NCCL tends to be unreliable for some reason
        strategy=DDPStrategy(process_group_backend="gloo"),  # NCCL tends to be unreliable for some reason
        max_epochs=186,
        log_every_n_steps=100,
        detect_anomaly=True,
        callbacks=[checkpoint_callback, lr_Monitor],
        val_check_interval=10747,  #7165
        check_val_every_n_epoch=1,
        default_root_dir="/graphics/scratch2/staff/zakeri/train_logs/VAE/skip_connection/v403_64_2x2x2_noBNDecoder_shapenetcorev2_excluding_shapenetcorev1_validation_split/",
        # resume_from_checkpoint="/graphics/scratch2/staff/zakeri/train_logs/VAE/skip_connection/v403_64_2x2x2_noBNDecoder_newestLMDB/lightning_logs/version_1/checkpoints/checkpoint-epoch=078-loss=0.000.ckpt", # v2 v403_64_2x2x2_noBNDecoder_newestLMDB
        # resume_from_checkpoint=""
    )

    trainer.fit(model)
    # v0 max_epochs=250, #7165, max_epochs=250
    # v3 is resume of v0 from epoch 181 for 5 epochs just to let scheduleler goes to zero, max_epochs=5


# if __name__ == "__main__":
#     main()
