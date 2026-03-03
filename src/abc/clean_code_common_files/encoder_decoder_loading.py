import torch
from Networks import frozen_encoder_decoder as fe
from Networks import frozen_encoder_decoder_16cube as ae_16c
from einops import rearrange
from Helpers.vae_specific_fns import sample_from_distribution, prepare_encoded_voxel_for_sampling

def load_encoder_from_checkpoint(pre_trained_model, latent_dim):
    params_encoder = pre_trained_model.encoder
    params_encoder.freeze()
    params_encoder.train(False)
    fencoder = fe.FrozenVSEncoderv9(params_encoder.to(pre_trained_model.device), latent_dim).to(pre_trained_model.device).train(False)
    fencoder.train(False)
    fencoder.freeze()
    # it is sequential, and the element 1 is always the batch_norm
    fencoder.convblock1.conv[1].track_running_stats = False
    fencoder.convblock1_1.conv[1].track_running_stats = False
    fencoder.convblock2.conv[1].track_running_stats = False
    fencoder.convblock2_1.conv[1].track_running_stats = False
    fencoder.convblock3.conv[1].track_running_stats = False
    fencoder.convblock3_1.conv[1].track_running_stats = False
    fencoder.convblock4.conv[1].track_running_stats = False
    fencoder.convblock4_1.conv[1].track_running_stats = False
    fencoder.convblock5.conv[1].track_running_stats = False
    fencoder.convblock5_1.conv[1].track_running_stats = False
    fencoder.convblock6.conv[1].track_running_stats = False
    fencoder.convblock6_1.conv[1].track_running_stats = False
    fencoder.convblock7.conv[1].track_running_stats = False
    fencoder.convblock7_1.conv[1].track_running_stats = False
    fencoder.convblock8.conv[1].track_running_stats = False
    fencoder.convblock8_1.conv[1].track_running_stats = False
    fencoder.convblock9.conv[1].track_running_stats = False
    fencoder.convblock9_1.conv[1].track_running_stats = False
    fencoder.convblock10.conv[1].track_running_stats = False
    fencoder.convblock10_1.conv[1].track_running_stats = False
    print(" encoder is restored from pre-trained VEncoder and is frozen")
    return fencoder


def load_decoder_from_checkpoint(pre_trained_model, latent_dim):
    params_decoder = pre_trained_model.decoder
    params_decoder.freeze()
    params_decoder.train(False)
    fdecoder = fe.FrozenVSDecoderv9(params_decoder.to(pre_trained_model.device), latent_dim).to(pre_trained_model.device).train(False)
    fdecoder.train(False)
    fdecoder.freeze()
    fdecoder.batchNorm3d5.track_running_stats = False
    fdecoder.batchNorm3d6.track_running_stats = False
    fdecoder.batchnorm3d7.track_running_stats = False
    # print(" decoder is restored from pre-trained VEncoder and is frozen")
    return fdecoder


def load_decoder16cube_from_checkpoint(pre_trained_model, latent_dim):
    params_decoder = pre_trained_model.decoder
    fdecoder = ae_16c.VSFDecoderv10(params_decoder, latent_dim)
    fdecoder.freeze()
    fdecoder.train(False)
    fdecoder.batchNorm3d4.track_running_stats = False
    fdecoder.batchNorm3d5.track_running_stats = False
    fdecoder.batchNorm3d6.track_running_stats = False
    fdecoder.batchnorm3d7.track_running_stats = False
    print("\n decoder is restored from pre-trained VEncoder and is not frozen")
    return fdecoder

def load_encoder16cube_from_checkpoint(pre_trained_model, latent_dim):
    params_encoder = pre_trained_model.encoder
    fencoder = ae_16c.VSFEncoderv10(params_encoder, latent_dim)
    fencoder.freeze()
    fencoder.train(False)
    # it is sequential, and the element 1 is always the batch_norm
    fencoder.convblock1.conv[1].track_running_stats = False
    fencoder.convblock1_1.conv[1].track_running_stats = False
    fencoder.convblock2.conv[1].track_running_stats = False
    fencoder.convblock2_1.conv[1].track_running_stats = False
    fencoder.convblock3.conv[1].track_running_stats = False
    fencoder.convblock3_1.conv[1].track_running_stats = False
    fencoder.convblock4.conv[1].track_running_stats = False
    fencoder.convblock4_1.conv[1].track_running_stats = False
    fencoder.convblock5.conv[1].track_running_stats = False
    fencoder.convblock5_1.conv[1].track_running_stats = False
    fencoder.convblock6.conv[1].track_running_stats = False
    fencoder.convblock6_1.conv[1].track_running_stats = False
    fencoder.convblock7.conv[1].track_running_stats = False
    fencoder.convblock7_1.conv[1].track_running_stats = False
    fencoder.convblock8.conv[1].track_running_stats = False
    fencoder.convblock8_1.conv[1].track_running_stats = False
    fencoder.convblock9.conv[1].track_running_stats = False
    fencoder.convblock9_1.conv[1].track_running_stats = False
    fencoder.convblock10.conv[1].track_running_stats = False
    fencoder.convblock10_1.conv[1].track_running_stats = False
    fencoder.convblock11.conv[1].track_running_stats = False
    fencoder.convblock11_1.conv[1].track_running_stats = False
    print(" encoder is restored from pre-trained VEncoder and is frozen")
    return fencoder

def load_encoder8cube_from_checkpoint(pre_trained_model, latent_dim):
    from Networks.stream8cube import skip_connection_4_noBNDecoder_1x1x1 as sc8
    params_encoder = pre_trained_model.encoder
    fencoder = sc8.VSEncoderv10(params_encoder, latent_dim) #CHNAGEME
    fencoder.freeze()
    fencoder.train(False)
    print("\n encoder is restored from pre-trained VEncoder and is frozen")
    # it is sequential, and the element 1 is always the batch_norm
    fencoder.convblock1.conv[1].track_running_stats = False
    fencoder.convblock1_1.conv[1].track_running_stats = False
    fencoder.convblock2.conv[1].track_running_stats = False
    fencoder.convblock2_1.conv[1].track_running_stats = False
    fencoder.convblock3.conv[1].track_running_stats = False
    fencoder.convblock3_1.conv[1].track_running_stats = False
    fencoder.convblock4.conv[1].track_running_stats = False
    fencoder.convblock4_1.conv[1].track_running_stats = False
    fencoder.convblock5.conv[1].track_running_stats = False
    fencoder.convblock5_1.conv[1].track_running_stats = False
    fencoder.convblock6.conv[1].track_running_stats = False
    fencoder.convblock6_1.conv[1].track_running_stats = False
    fencoder.convblock7.conv[1].track_running_stats = False
    fencoder.convblock7_1.conv[1].track_running_stats = False
    fencoder.convblock8.conv[1].track_running_stats = False
    fencoder.convblock8_1.conv[1].track_running_stats = False
    fencoder.convblock9.conv[1].track_running_stats = False
    fencoder.convblock9_1.conv[1].track_running_stats = False
    fencoder.convblock10.conv[1].track_running_stats = False
    fencoder.convblock10_1.conv[1].track_running_stats = False
    print(" encoder is restored from pre-trained VEncoder and is frozen")
    return fencoder

def get_encoded_latent_code(self, sub_voxels: torch.Tensor) -> torch.Tensor:
    batch_size = sub_voxels.shape[0]
    #  Prep for Encoding--------------------------------------------------------------------------------------------------------------------
    # TODO : I was the reason for fucking you. dimension broadcasting, nasty python
    # Prepare sub_voxels for encoding and encode them to extract non-optimized_latent_codes-----------------------------------------------
    non_optimized_latent_codes = self.prep_sub_voxels_and_encode(sub_voxels)
    # Prepare for forward----------------------------------------------------------------------------------------------------------------
    non_optimized_latent_codes_reshaped = non_optimized_latent_codes.reshape(batch_size, self.number_of_sub_voxels, self.hparams.latent_dim, 2, 2, 2)
    assert non_optimized_latent_codes_reshaped.shape == (batch_size, self.number_of_sub_voxels, self.hparams.latent_dim, 2, 2, 2)
    # Non-Optimized latent code collect------------------------------------------------------------------------------------------------------------
    # decoder--------------------------
    with torch.no_grad():
        decoded_non_optimized_latent_codes = self.fdecoder(non_optimized_latent_codes_reshaped).to(self.device)  # [128, 512, 2, 2, 2] -> [128, 1, 32, 32, 32]
    decoded_non_optimized_latent_codes_reshaped = decoded_non_optimized_latent_codes.reshape(
        [batch_size, self.number_of_sub_voxels, self.hparams.target_resolution, self.hparams.target_resolution, self.hparams.target_resolution]
    )
    # del decoded_latent_codes_reshaped
    collected_sub_voxels_decoded_non_optimized = collect_sub_voxels_to_voxel_with_batch(decoded_non_optimized_latent_codes_reshaped, self.hparams.resolution)
    assert collected_sub_voxels_decoded_non_optimized.shape == (batch_size, self.hparams.resolution, self.hparams.resolution, self.hparams.resolution)
    return non_optimized_latent_codes_reshaped

# def decode_selected_sub_voxels(self, sub_voxels: torch.Tensor, optimized_latent_codes: torch.Tensor, decoder_bool: bool) -> tuple:
#     selected_sub_voxels_for_decoding = sub_voxels[decoder_bool]
#     selected_sub_voxels_for_decoding_un_squeezed = selected_sub_voxels_for_decoding.unsqueeze(0)
#     number_of_selected_sub_voxels_for_decoding = selected_sub_voxels_for_decoding.shape[0]
#     selected_latent_codes_for_decoding = optimized_latent_codes[decoder_bool]
#     with torch.no_grad():
#         decoded_selected_latent_codes_for_decoding = self.fdecoder(selected_latent_codes_for_decoding).to(self.device)
#     # FIXME this need to be rechecked, checked
#     decoded_selected_latent_codes_for_decoding_reshaped = decoded_selected_latent_codes_for_decoding.reshape(
#         [1, number_of_selected_sub_voxels_for_decoding, self.hparams.target_resolution, self.hparams.target_resolution, self.hparams.target_resolution]
#     )
#     assert selected_sub_voxels_for_decoding_un_squeezed.shape == decoded_selected_latent_codes_for_decoding_reshaped.shape
#     return (decoded_selected_latent_codes_for_decoding_reshaped, selected_sub_voxels_for_decoding_un_squeezed)