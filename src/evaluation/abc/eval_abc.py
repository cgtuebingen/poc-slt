import os
import sys
root = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), "../../../"))
from typing import Tuple, Any, Union
import torch
from tqdm import tqdm
from src.utils import sub_voxel_related_fns as pp_fns
from src.utils.positional_encoder_class import MYPositionalEncoder3D
import pickle
# ----------------------------------------------------------------------------------------------------------------------------------------------------
from src.utils import encoder_decoder_loading as ed
from src.training.train_no_empty_masking_custom_abc import (
    TransformerSDFtoSDFABCOUTSIDE,
)
from src.evaluation.abc.calculate_bbx_abc_and_normalize_them_to_shapenet_scale import (
    normalize_abc_compatible_with_shapenet,
)
from src.utils.custom_cutting import custom_mask
from src.evaluation.abc.dataset_abc import setup_dataset
from src.unused_code.common_fns_abc import (
    march_voxels_and_write_objs,
    evaluate,
    write_evaluation_result,
    march_gt_and_mask_only_and_write_objs,
)

from src.p_vae.pvae import SDFtoSDF


# --------------------------------------------------------------------------------------------------------------------------------------------------------
class EVALABC:
    def __init__(
        self,
        latent_dim: int,
        resolution: int,
        target_resolution: int,
        val_batch_size: int,
        val_lmdb_path: str,
        obj_dir: str,
        value_range: int,
        eval_dir: str,
        checkpoint_path: str,
        marching_cube_result_dir: str,
        pre_trained: bool,
        custom_mask_mode: str,
        vae_checkpoint_path: str,
        device: str,
        num_samples: int,
        min_range: int,
        max_range: int,
        common_obj_dir: str,
    ):
        self.latent_dim = latent_dim
        self.resolution = resolution
        self.target_resolution = target_resolution

        self.val_batch_size = val_batch_size

        self.val_lmdb_path = val_lmdb_path
        self.obj_dir = obj_dir

        self.checkpoint_path = checkpoint_path
        self.marching_cube_result_dir = marching_cube_result_dir
        self.pre_trained = pre_trained
        self.custom_mask_mode = custom_mask_mode
        self.vae_checkpoint_path = vae_checkpoint_path

        self.value_range = value_range
        self.eval_dir = eval_dir

        self.range_to_evaluate = [min_range, max_range]

        self.device = device
        self.num_samples = num_samples
        self.common_obj_dir = common_obj_dir

        ABC_mesh_file_name_to_bbx_path = os.path.join(root, "data", "ABC",  "mesh_file_names_to_bbx", "data.pkl")
        with open(ABC_mesh_file_name_to_bbx_path, "rb") as f:
            self.mesh_file_name_to_bbx = pickle.load(f)
        # self.mesh_file_name_to_bbx = torch.load(ABC_mesh_file_name_to_bbx_path, weights_only=False)

        if self.pre_trained:
            self.pre_trained_transformer = (
                TransformerSDFtoSDFABCOUTSIDE.load_from_checkpoint(
                    checkpoint_path=self.checkpoint_path,
                    vae_checkpoint_path=self.vae_checkpoint_path,
                    transformer_checkpoint_path=self.checkpoint_path,
                ).to(self.device)
            )
            self.pre_trained_transformer.eval()
            self.pre_trained_transformer.train(False)

            self.frozen_regular_transformer = (
                self.pre_trained_transformer.regular_transformer
            )
            self.frozen_regular_transformer.eval()
            self.frozen_regular_transformer.train(False)

            self.frozen_mapping_down = self.pre_trained_transformer.mapping_down
            self.frozen_mapping_down.eval()
            self.frozen_mapping_down.train(False)

            self.frozen_mapping_up = self.pre_trained_transformer.mapping_up
            self.frozen_mapping_up.eval()
            self.frozen_mapping_up.train(False)

            self.frozen_redundant_mapping = (
                self.pre_trained_transformer.redundant_mapping
            )
            self.frozen_redundant_mapping.eval()
            self.frozen_redundant_mapping.train(False)

            self.frozen_constant_learnable_mask_tensors = (
                self.pre_trained_transformer.constant_learnable_mask_tensors
            )

            # vae model
            pre_trained_model_vae = SDFtoSDF.load_from_checkpoint(
                self.vae_checkpoint_path
            ).to(self.device)
            pre_trained_model_vae.eval()
            pre_trained_model_vae.train(False)

            self.fencoder = ed.load_encoder_from_checkpoint(
                pre_trained_model_vae, latent_dim
            )
            self.fdecoder = ed.load_decoder_from_checkpoint(
                pre_trained_model_vae, latent_dim
            )

        number_of_sub_voxels = self.resolution // self.target_resolution
        self.number_of_sub_voxels = (
            number_of_sub_voxels * number_of_sub_voxels * number_of_sub_voxels
        )

        self.target_resolution = target_resolution
        self.resolution = resolution

        self.penc_channels = 8 * self.latent_dim
        self.positional_encoder_3d = MYPositionalEncoder3D(self.penc_channels)

    def call_transformer_and_mapping_layers(
        self, transformer_input_sequence: torch.Tensor
    ) -> torch.Tensor:
        transformer_input_sequence = transformer_input_sequence.clone()

        transformer_input_sequence_down = self.frozen_mapping_down(
            transformer_input_sequence
        )
        transformer_output_sequence = self.frozen_regular_transformer(
            transformer_input_sequence_down
        )
        transformer_output_sequence_up = self.frozen_mapping_up(
            transformer_output_sequence
        )

        return transformer_output_sequence_up.clone()

    def assign_constant_learnable_tensors(
        self, optimized_latent_codes: torch.Tensor, bool_dict: dict
    ) -> torch.Tensor:
        masked_optimized_latent_codes = optimized_latent_codes.clone()

        mask_all_bool = bool_dict["mask_all_bool"]

        masked_optimized_latent_codes[mask_all_bool] = (
            self.frozen_constant_learnable_mask_tensors
        )

        return masked_optimized_latent_codes.clone()

    def forward_ABC(
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
                [batch_size, self.number_of_sub_voxels, 8 * self.latent_dim]
            )
        )
        masked_non_optimized_non_latent_codes_reshaped_mapped = (
            self.frozen_redundant_mapping(
                masked_non_optimized_non_latent_codes_reshaped
            )
        )
        assert (
            masked_non_optimized_non_latent_codes_reshaped.shape
            == masked_non_optimized_non_latent_codes_reshaped_mapped.shape
        )
        z_positionally_encoded_re = self.positional_encoder_3d(
            shape_of_positions=[batch_size, 4, 4, 4, self.penc_channels]
        ).to(device=self.device)
        assert (
            z_positionally_encoded_re.shape
            == masked_non_optimized_non_latent_codes_reshaped_mapped.shape
        )

        transformer_input_sequence = torch.cat(
            (
                z_positionally_encoded_re,
                masked_non_optimized_non_latent_codes_reshaped_mapped,
            ),
            dim=2,
        ).to(device=self.device)
        transformer_output_sequence = self.call_transformer_and_mapping_layers(
            transformer_input_sequence
        )

        return (transformer_output_sequence, masked_non_optimized_latent_codes)

    def fwd_ABC(self, batch: list, train: bool) -> Tuple:
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

        sub_voxels = pp_fns.sub_divide_gt_and_normalize(
            gt_sdf_full_voxel.clone(), self.number_of_sub_voxels, self.target_resolution
        )
        empty_sub_voxels_bool, non_empty_sub_voxels_bool = (
            pp_fns.extract_outside_and_non_outside_voxels(
                sub_voxels.clone(), self.number_of_sub_voxels, self.target_resolution
            )
        )

        if not torch.all(torch.any(non_empty_sub_voxels_bool, dim=1)):
            pair = {
                "key": keys,
                "object_index": object_indices.item(),
                "obj_file_name": obj_file_names,
            }
            print("\nall voxels are empty", pair)
            breakpoint()

        mask_all_bool = custom_mask(
            self.custom_mask_mode,
            self.target_resolution,
            batch_size,
            self.number_of_sub_voxels,
            given_device=self.device,
        )

        # FORWARD CAlL----------------------------------------------------------------------------------------------------------------------------
        transformer_output_sequence_up, masked_non_optimized_latent_codes = (
            self.forward_ABC(sub_voxels, non_optimized_latent_codes, mask_all_bool)
        )

        return (
            masked_non_optimized_latent_codes,
            transformer_output_sequence_up,
            sub_voxels,
        )

    def evaluate_val(self):

        val_dataset = setup_dataset(
            self.obj_dir, self.val_lmdb_path, self.value_range, self.resolution
        )

        min_range = self.range_to_evaluate[0]
        max_range = self.range_to_evaluate[1]
        print("\n  min_range:", min_range, ", max: ", max_range)
        print(
            "\n  self.val_dataset len:",
        )
        for i in tqdm(range(min_range, max_range, 1), desc="Val Samples"):
            val_sample = val_dataset[i]
            (
                keys,
                object_indices,
                obj_file_names,
                gt_sdf_full_voxel,
                # std_copy,
                # var_copy,
                non_optimized_latent_codes,
            ) = val_sample

            # adding batch
            object_indices = torch.tensor([object_indices]).to(device=self.device)
            gt_sdf_full_voxel = gt_sdf_full_voxel.unsqueeze(0).to(device=self.device)
            non_optimized_latent_codes = non_optimized_latent_codes.unsqueeze(0).to(
                device=self.device
            )

            val_batch = [
                keys,
                object_indices,
                obj_file_names,
                gt_sdf_full_voxel,
                # std_copy,
                # var_copy,
                non_optimized_latent_codes,
            ]

            batch_size = object_indices.shape[0]
            assert object_indices.shape == (self.val_batch_size,)

            stuff = self.fwd_ABC(val_batch, False)

            (
                masked_non_optimized_latent_codes,
                transformer_output_sequence_up,
                sub_voxels,
            ) = stuff
            # visualization and tensorboard---------------------------
            transformer_output_sequence_up_reshaped = (
                transformer_output_sequence_up.reshape(
                    [
                        batch_size,
                        self.number_of_sub_voxels,
                        self.latent_dim,
                        2,
                        2,
                        2,
                    ]
                )
                .to(device="cuda")
                .to(dtype=torch.float32)
            )

            # here is where we prep data for eval so normalizing ABC compatible with shapenet
            hausdorff_scale, chamfer_scale = normalize_abc_compatible_with_shapenet(
                self.mesh_file_name_to_bbx, obj_file_names
            )
            dict_arguments_for_vis = {
                "non_optimized_latent_codes": non_optimized_latent_codes,
                "masked_non_optimized_latent_codes": masked_non_optimized_latent_codes,
                "transformer_output_sequence_up": transformer_output_sequence_up_reshaped,
                "sub_voxels": sub_voxels,
            }

            dict_arguments_of_variables = {
                "number_of_sub_voxels": self.number_of_sub_voxels,
                "latent_dim": self.latent_dim,
                "target_resolution": self.target_resolution,
                "resolution": self.resolution,
                "hausdorff_scale": hausdorff_scale,
                "chamfer_scale": chamfer_scale,
                "object_index": object_indices,
                "obj_file_name": obj_file_names,
                "marching_cube_result_dir": self.marching_cube_result_dir,
                "common_obj_dir": self.common_obj_dir,
                "num_samples": self.num_samples,
            }
            march_gt_and_mask_only_and_write_objs(
                dict_arguments_for_vis,
                dict_arguments_of_variables,
                object_indices,
                self.fdecoder,
            )
            dict_arguments_for_eval, collected_data_dict_for_plotting = (
                march_voxels_and_write_objs(
                    dict_arguments_for_vis,
                    dict_arguments_of_variables,
                    object_indices,
                    self.fdecoder,
                )
            )

            eval_results = evaluate(
                dict_arguments_for_eval,
                dict_arguments_of_variables,
                collected_data_dict_for_plotting,
            )
            write_evaluation_result(eval_results, self.eval_dir)
