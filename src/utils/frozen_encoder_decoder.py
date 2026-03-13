import pytorch_lightning as pl
import torch
from torch import nn

from src.utils import network_helper_fns as sc


class FVSEncoder(pl.LightningModule):
    """Encoder module"""

    def __init__(self, params_encoder, latent_dim: int):
        super(FVSEncoder, self).__init__()
        self.latent_dim = latent_dim

        if params_encoder is not None:

            self.convblock1 = params_encoder.convblock1
            self.convblock1_1 = params_encoder.convblock1_1
            self.convblock2 = params_encoder.convblock2
            self.convblock2_1 = params_encoder.convblock2_1
            self.maxpool2 = params_encoder.maxpool2
            self.sk2 = params_encoder.sk2

            self.convblock3 = params_encoder.convblock3
            self.convblock3_1 = params_encoder.convblock3_1
            self.convblock4 = params_encoder.convblock4
            self.convblock4_1 = params_encoder.convblock4_1
            # mapping conv
            self.conv_map4 = params_encoder.conv_map4
            self.maxpool4 = params_encoder.maxpool4
            self.sk4 = params_encoder.sk4
            self.convblock5 = params_encoder.convblock5
            self.convblock5_1 = params_encoder.convblock5_1
            self.convblock6 = params_encoder.convblock6
            self.convblock6_1 = params_encoder.convblock6_1
            # mapping conv
            self.conv_map6 = params_encoder.conv_map6
            self.maxpool6 = params_encoder.maxpool6
            self.sk6 = params_encoder.sk6

            self.convblock7 = params_encoder.convblock7
            self.convblock7_1 = params_encoder.convblock7_1
            self.convblock8 = params_encoder.convblock8
            self.convblock8_1 = params_encoder.convblock8_1

            # mapping conv
            self.conv_map8 = params_encoder.conv_map8
            self.maxpool8 = params_encoder.maxpool8
            self.sk8 = params_encoder.sk8
            self.convblock9 = params_encoder.convblock9
            self.convblock9_1 = params_encoder.convblock9_1
            self.convblock10 = params_encoder.convblock10
            self.convblock10_1 = params_encoder.convblock10_1
            # mapping conv
            self.conv_map10 = params_encoder.conv_map10
            self.conv11 = params_encoder.conv11

        else:
            self.convblock1 = sc.ConvBlock(
                ch_in=1, ch_out=32, k_size=3, s_size=1, p_size=1
            )
            self.convblock1_1 = sc.ConvBlock(
                ch_in=32, ch_out=32, k_size=3, s_size=1, p_size=1
            )

            self.convblock2 = sc.ConvBlock(
                ch_in=32, ch_out=32, k_size=3, s_size=1, p_size=1
            )
            self.convblock2_1 = sc.ConvBlock(
                ch_in=32, ch_out=32, k_size=3, s_size=1, p_size=1
            )

            self.maxpool2 = nn.MaxPool3d(3, stride=2, padding=1)
            self.sk2 = nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1)
            # ----------------------------------------------------------------------
            self.convblock3 = sc.ConvBlock(
                ch_in=32, ch_out=64, k_size=3, s_size=1, p_size=1
            )
            self.convblock3_1 = sc.ConvBlock(
                ch_in=64, ch_out=64, k_size=3, s_size=1, p_size=1
            )

            self.convblock4 = sc.ConvBlock(
                ch_in=64, ch_out=64, k_size=3, s_size=1, p_size=1
            )
            self.convblock4_1 = sc.ConvBlock(
                ch_in=64, ch_out=64, k_size=3, s_size=1, p_size=1
            )

            # mapping conv
            self.conv_map4 = nn.Conv3d(32, 64, kernel_size=1, stride=1, padding=0)
            self.maxpool4 = nn.MaxPool3d(3, stride=2, padding=1)
            self.sk4 = nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1)
            # -----------------------------------------------------------------------
            self.convblock5 = sc.ConvBlock(
                ch_in=64, ch_out=128, k_size=3, s_size=1, p_size=1
            )
            self.convblock5_1 = sc.ConvBlock(
                ch_in=128, ch_out=128, k_size=3, s_size=1, p_size=1
            )

            self.convblock6 = sc.ConvBlock(
                ch_in=128, ch_out=128, k_size=3, s_size=1, p_size=1
            )
            self.convblock6_1 = sc.ConvBlock(
                ch_in=128, ch_out=128, k_size=3, s_size=1, p_size=1
            )
            # mapping conv
            self.conv_map6 = nn.Conv3d(64, 128, kernel_size=1, stride=1, padding=0)
            self.maxpool6 = nn.MaxPool3d(3, stride=2, padding=1)
            self.sk6 = nn.Conv3d(128, 128, kernel_size=3, stride=2, padding=1)
            # -----------------------------------------------------------------------
            self.convblock7 = sc.ConvBlock(
                ch_in=128, ch_out=256, k_size=3, s_size=1, p_size=1
            )
            self.convblock7_1 = sc.ConvBlock(
                ch_in=256, ch_out=256, k_size=3, s_size=1, p_size=1
            )

            self.convblock8 = sc.ConvBlock(
                ch_in=256, ch_out=256, k_size=3, s_size=1, p_size=1
            )
            self.convblock8_1 = sc.ConvBlock(
                ch_in=256, ch_out=256, k_size=3, s_size=1, p_size=1
            )

            # mapping conv
            self.conv_map8 = nn.Conv3d(128, 256, kernel_size=1, stride=1, padding=0)
            self.maxpool8 = nn.MaxPool3d(3, stride=2, padding=1)
            self.sk8 = nn.Conv3d(256, 256, kernel_size=3, stride=2, padding=1)
            # -----------------------------------------------------------------------
            self.convblock9 = sc.ConvBlock(
                ch_in=256, ch_out=512, k_size=3, s_size=1, p_size=1
            )
            self.convblock9_1 = sc.ConvBlock(
                ch_in=512, ch_out=512, k_size=3, s_size=1, p_size=1
            )

            self.convblock10 = sc.ConvBlock(
                ch_in=512, ch_out=512, k_size=3, s_size=1, p_size=1
            )
            self.convblock10_1 = sc.ConvBlock(
                ch_in=512, ch_out=512, k_size=3, s_size=1, p_size=1
            )

            # mapping conv
            self.conv_map10 = nn.Conv3d(256, 512, kernel_size=1, stride=1, padding=0)
            # -----------------------------------------------------------------------
            self.conv11 = nn.Conv3d(512, 1024, kernel_size=1, stride=1, padding=0)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        block1 = self.convblock1(x_in)
        block1_1 = self.convblock1_1(block1)
        # -----------------------------------------
        block2 = self.convblock2(block1_1)
        # -----------------------------------------
        block2_1 = self.convblock2_1(block2)
        # -----------------------------------------
        block = block2_1 + x_in
        # -----------------------------------------
        block2_m = self.maxpool2(block)
        skip2 = self.sk2(block)
        # -----------------------------------------
        block = skip2 + block2_m
        # -----------------------------------------
        block3 = self.convblock3(block)
        block3_1 = self.convblock3_1(block3)
        # -----------------------------------------
        block4 = self.convblock4(block3_1)
        block4_1 = self.convblock4_1(block4)
        # -----------------------------------------
        # mapping block to the size pf block4_1 channel
        block_mapped = self.conv_map4(block)
        block = block_mapped + block4_1
        # -----------------------------------------
        block4_m = self.maxpool4(block)
        skip4 = self.sk4(block)
        # -----------------------------------------
        block = skip4 + block4_m
        # -----------------------------------------
        block5 = self.convblock5(block)
        block5_1 = self.convblock5_1(block5)
        # -----------------------------------------
        block6 = self.convblock6(block5_1)
        block6_1 = self.convblock6_1(block6)

        # mapping block to the size pf block4_1 channel
        block_mapped = self.conv_map6(block)
        block = block_mapped + block6_1

        block6_m = self.maxpool6(block)
        skip6 = self.sk6(block)
        # -----------------------------------------
        block = skip6 + block6_m
        # -----------------------------------------
        block7 = self.convblock7(block)
        block7_1 = self.convblock7_1(block7)
        # -----------------------------------------
        block8 = self.convblock8(block7_1)
        block8_1 = self.convblock8_1(block8)

        # mapping block to the size pf block4_1 channel
        block_mapped = self.conv_map8(block)
        block = block_mapped + block8_1

        block8_m = self.maxpool8(block)
        skip8 = self.sk8(block)
        # -----------------------------------------
        block = skip8 + block8_m

        block9 = self.convblock9(block)
        block9_1 = self.convblock9_1(block9)

        block10 = self.convblock10(block9_1)
        block10_1 = self.convblock10_1(block10)
        # -----------------------------------------
        # mapping block to the size pf block4_1 channel
        block_mapped = self.conv_map10(block)
        block = block_mapped + block10_1

        block11 = self.conv11(block)
        x_out = block11
        return x_out


class FVSDecoder(pl.LightningModule):
    """Decoder Module"""

    def __init__(self, params_decoder, latent_dim: int):
        super(FVSDecoder, self).__init__()

        self.latent_dim = latent_dim
        if params_decoder is not None:
            self.conv7 = params_decoder.conv7
            self.batchnorm3d7 = params_decoder.batchnorm3d7
            self.relu7 = params_decoder.relu7
            self.sk7 = params_decoder.sk7
            self.upsample7 = params_decoder.upsample7

            self.conv6 = params_decoder.conv6
            self.batchNorm3d6 = params_decoder.batchNorm3d6
            self.relu6 = params_decoder.relu6
            self.sk6 = params_decoder.sk6
            self.upsample6 = params_decoder.upsample6

            self.conv5 = params_decoder.conv5
            self.batchNorm3d5 = params_decoder.batchNorm3d5
            self.relu5 = params_decoder.relu5
            self.sk5 = params_decoder.sk5
            self.upsample5 = params_decoder.upsample5

            self.conv4 = params_decoder.conv4
            self.relu4 = params_decoder.relu4
            self.sk4 = params_decoder.sk4
            self.upsample4 = params_decoder.upsample4

            self.linear_down = params_decoder.linear_down
        else:
            self.conv7 = nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1)
            self.batchnorm3d7 = nn.BatchNorm3d(512)
            self.relu7 = nn.ReLU()
            self.sk7 = nn.ConvTranspose3d(
                512, 512, kernel_size=3, stride=2, padding=1, output_padding=1
            )
            self.upsample7 = nn.Upsample(
                scale_factor=2, mode="trilinear", align_corners=False
            )

            self.conv6 = nn.Conv3d(512, 256, kernel_size=3, stride=1, padding=1)
            self.batchNorm3d6 = nn.BatchNorm3d(256)
            self.relu6 = nn.ReLU()
            self.sk6 = nn.ConvTranspose3d(
                256, 256, kernel_size=3, stride=2, padding=1, output_padding=1
            )
            self.upsample6 = nn.Upsample(
                scale_factor=2, mode="trilinear", align_corners=False
            )

            self.conv5 = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)
            self.batchNorm3d5 = nn.BatchNorm3d(256)
            self.relu5 = nn.ReLU()
            self.sk5 = nn.ConvTranspose3d(
                256, 256, kernel_size=3, stride=2, padding=1, output_padding=1
            )
            self.upsample5 = nn.Upsample(
                scale_factor=2, mode="trilinear", align_corners=False
            )

            self.conv4 = nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1)
            self.relu4 = nn.ReLU()
            self.sk4 = nn.ConvTranspose3d(
                128, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            )
            self.upsample4 = nn.Upsample(
                scale_factor=2, mode="trilinear", align_corners=False
            )

            self.linear_down = nn.Conv3d(128, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x_view = x_in.view(-1, 512, 2, 2, 2)  # 8*8*8 for 128?^3 and 2*2*2 for 32^3

        b7 = self.conv7(x_view)
        b7 = self.batchnorm3d7(b7)
        skip_7 = self.sk7(b7)
        u7 = self.upsample7(b7)
        b7_1 = skip_7 + u7
        u7_1 = self.relu7(b7_1)
        # -----------------------------------------
        b6 = self.conv6(u7_1)
        b6 = self.batchNorm3d6(b6)
        skip6 = self.sk6(b6)
        u6 = self.upsample6(b6)
        b6_1 = skip6 + u6
        u6_1 = self.relu6(b6_1)
        # -----------------------------------------
        b5 = self.conv5(u6_1)
        b5 = self.batchNorm3d5(b5)
        skip_5 = self.sk5(b5)
        u5 = self.upsample5(b5)
        b5_1 = u5 + skip_5
        u5_1 = self.relu5(b5_1)
        # -----------------------------------------
        b4 = self.conv4(u5_1)
        skip_4 = self.sk4(b4)
        u4 = self.upsample4(b4)
        b4_1 = skip_4 + u4
        u4_1 = self.relu4(b4_1)
        # -----------------------------------------
        result4_mapped = self.linear_down(u4_1)
        x_out = result4_mapped
        return x_out
