from src.utils import frozen_encoder_decoder as fe


def load_encoder_from_checkpoint(pre_trained_model, latent_dim):
    params_encoder = pre_trained_model.encoder
    params_encoder.freeze()
    params_encoder.train(False)
    fencoder = (
        fe.FVSEncoder(params_encoder.to(pre_trained_model.device), latent_dim)
        .to(pre_trained_model.device)
        .train(False)
    )
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
    fdecoder = (
        fe.FVSDecoder(params_decoder.to(pre_trained_model.device), latent_dim)
        .to(pre_trained_model.device)
        .train(False)
    )
    fdecoder.train(False)
    fdecoder.freeze()
    fdecoder.batchNorm3d5.track_running_stats = False
    fdecoder.batchNorm3d6.track_running_stats = False
    fdecoder.batchnorm3d7.track_running_stats = False
    return fdecoder
