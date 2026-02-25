import pytorch_lightning as pl
import torch
from torch import nn
import sys
sys.path.append("../../")
# from Networks import autoencoder as ae
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class ConvBlock(pl.LightningModule):
    def __init__(self, ch_in: int, ch_out: int, k_size: int, s_size: int = 1, p_size: int = 1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=k_size, stride=s_size, padding=p_size),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x_out = self.conv(x_in)
        return x_out
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class UpConv(pl.LightningModule):
    """ Reduce the number of features by 2 using Conv with kernel size 1x1x1 and double the spatial dimension using 3D trilinear up-sampling"""
    def __init__(self, ch_in, ch_out, k_size=1, scale=2, align_corners=False):
        super(UpConv, self).__init__()

        self.up_conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=k_size),
            nn.Upsample(scale_factor=scale, mode='trilinear', align_corners=align_corners),
        )
    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x_out = self.up_conv(x_in)
        return x_out
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class VSEncoderv9(pl.LightningModule):
    """ Encoder module """

    def __init__(self, latent_dim: int):
        super(VSEncoderv9, self).__init__()
        self.latent_dim = latent_dim

        self.convblock1 = ConvBlock(ch_in=1, ch_out=32, k_size=3, s_size=1, p_size=1)
        self.convblock1_1 = ConvBlock(ch_in=32, ch_out=32, k_size=3, s_size=1, p_size=1)

        self.convblock2 = ConvBlock(ch_in=32, ch_out=32, k_size=3, s_size=1, p_size=1)
        self.convblock2_1 = ConvBlock(ch_in=32, ch_out=32, k_size=3, s_size=1, p_size=1)

        self.maxpool2 = nn.MaxPool3d(3, stride=2, padding=1)
        self.sk2 = nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1)
        #----------------------------------------------------------------------
        self.convblock3 = ConvBlock(ch_in=32, ch_out=64, k_size=3, s_size=1, p_size=1)
        self.convblock3_1 = ConvBlock(ch_in=64, ch_out=64, k_size=3, s_size=1, p_size=1)

        self.convblock4 = ConvBlock(ch_in=64, ch_out=64, k_size=3, s_size=1, p_size=1)
        self.convblock4_1 = ConvBlock(ch_in=64, ch_out=64, k_size=3, s_size=1, p_size=1)

        # mapping conv
        self.conv_map4 = nn.Conv3d(32, 64, kernel_size=1, stride=1, padding=0)
        self.maxpool4 = nn.MaxPool3d(3, stride=2, padding=1)
        self.sk4 = nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1)
        #-----------------------------------------------------------------------
        self.convblock5 = ConvBlock(ch_in=64, ch_out=128, k_size=3, s_size=1, p_size=1)
        self.convblock5_1 = ConvBlock(ch_in=128, ch_out=128, k_size=3, s_size=1, p_size=1)

        self.convblock6 = ConvBlock(ch_in=128, ch_out=128, k_size=3, s_size=1, p_size=1)
        self.convblock6_1 = ConvBlock(ch_in=128, ch_out=128, k_size=3, s_size=1, p_size=1)
        # mapping conv
        self.conv_map6 = nn.Conv3d(64, 128, kernel_size=1, stride=1, padding=0)

        self.maxpool6 = nn.MaxPool3d(3, stride=2, padding=1)
        self.sk6 = nn.Conv3d(128, 128, kernel_size=3, stride=2, padding=1)
        #-----------------------------------------------------------------------
        self.convblock7 = ConvBlock(ch_in=128, ch_out=256, k_size=3, s_size=1, p_size=1)
        self.convblock7_1 = ConvBlock(ch_in=256, ch_out=256, k_size=3, s_size=1, p_size=1)

        self.convblock8 = ConvBlock(ch_in=256, ch_out=256, k_size=3, s_size=1, p_size=1)
        self.convblock8_1 =ConvBlock(ch_in=256, ch_out=256, k_size=3, s_size=1, p_size=1)

        # mapping conv
        self.conv_map8 = nn.Conv3d(128, 256, kernel_size=1, stride=1, padding=0)
        self.maxpool8 = nn.MaxPool3d(3, stride=2, padding=1)
        self.sk8 = nn.Conv3d(256, 256, kernel_size=3, stride=2, padding=1)
        #-----------------------------------------------------------------------
        self.convblock9 = ConvBlock(ch_in=256, ch_out=512, k_size=3, s_size=1, p_size=1)
        self.convblock9_1 = ConvBlock(ch_in=512, ch_out=512, k_size=3, s_size=1, p_size=1)

        self.convblock10 = ConvBlock(ch_in=512, ch_out=512, k_size=3, s_size=1, p_size=1)
        self.convblock10_1 = ConvBlock(ch_in=512, ch_out=512, k_size=3, s_size=1, p_size=1)

        # mapping conv
        self.conv_map10 = nn.Conv3d(256, 512, kernel_size=1, stride=1, padding=0)

        #-----------------------------------------------------------------------
        self.conv11 = nn.Conv3d(512, 1024, kernel_size=1, stride=1, padding=0)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        # print("\n Encoder x_in: ", x_in.shape)
        block1 = self.convblock1(x_in)
        # print("\n Encoder block1 ", block1.shape)
        block1_1 = self.convblock1_1(block1)
        # print("\n Encoder block1_1 ", block1_1.shape)
        #-----------------------------------------
        block2 = self.convblock2(block1_1)
        # print("\n Encoder block2: ", block2.shape)
        #-----------------------------------------
        block2_1 = self.convblock2_1(block2)
        # print("\n Encoder block2_1: ", block2_1.shape)
        #-----------------------------------------
        block = block2_1 + x_in
        #-----------------------------------------
        block2_m = self.maxpool2(block)
        skip2 = self.sk2(block)
        #-----------------------------------------
        block = skip2 + block2_m
        #-----------------------------------------
        block3 = self.convblock3(block)
        # print("\n Encoder block3: ", block3.shape)
        block3_1 = self.convblock3_1(block3)
        # print("\n Encoder block3_1: ", block3_1.shape)
        #-----------------------------------------
        block4 = self.convblock4(block3_1)
        # print("\n Encoder block4: ", block4.shape)
        block4_1 = self.convblock4_1(block4)
        # print("\n Encoder block4_1: ", block4_1.shape)
        #-----------------------------------------
        # mapping block to the size pf block4_1 channel
        block_mapped = self.conv_map4(block)
        block = block_mapped + block4_1
        # print("\n Encoder block= block1 + block2: ", block.shape)
        #-----------------------------------------
        block4_m = self.maxpool4(block)
        # print("\n Encoder block4_m: ", block4_m.shape)
        skip4 = self.sk4(block)
        # print("\n Encoder skip4: ", skip4.shape)
        #-----------------------------------------
        block = skip4 + block4_m
        # print("\n Encoder block = skip4 + block4_m: ", block.shape)
        #-----------------------------------------
        block5 = self.convblock5(block)
        # print("\n Encoder block5: ", block5.shape)
        block5_1 = self.convblock5_1(block5)
        # print("\n Encoder block5_1: ", block5_1.shape)
        #-----------------------------------------
        block6 = self.convblock6(block5_1)
        # print("\n Encoder block6: ", block6.shape)
        block6_1 = self.convblock6_1(block6)
        # print("\n Encoder block6_1: ", block6_1.shape)

        # mapping block to the size pf block4_1 channel
        block_mapped = self.conv_map6(block)
        block = block_mapped + block6_1

        block6_m = self.maxpool6(block)
        skip6 = self.sk6(block)
        # -----------------------------------------
        block = skip6 + block6_m
        # -----------------------------------------
        block7 = self.convblock7(block)
        # print("\n Encoder block7: ", block7.shape)
        block7_1 = self.convblock7_1(block7)
        # print("\n Encoder block7_1: ", block7_1.shape)
        # -----------------------------------------
        block8 = self.convblock8(block7_1)
        # print("\n Encoder block8: ", block8.shape)
        block8_1 = self.convblock8_1(block8)

        # mapping block to the size pf block4_1 channel
        block_mapped = self.conv_map8(block)
        block = block_mapped + block8_1

        block8_m = self.maxpool8(block)
        skip8 = self.sk8(block)
        # -----------------------------------------
        block = skip8 + block8_m

        block9 = self.convblock9(block)
        # print("\n Encoder block9: ", block9.shape)
        block9_1 = self.convblock9_1(block9)

        block10 = self.convblock10(block9_1)
        # print("\n Encoder block10: ", block10.shape)
        block10_1 = self.convblock10_1(block10)
        # -----------------------------------------
        # mapping block to the size pf block4_1 channel
        block_mapped = self.conv_map10(block)
        block = block_mapped + block10_1

        block11 = self.conv11(block)
        x_out = block11
        return x_out

class VSDecoderv9(pl.LightningModule):
    """ Decoder Module """

    def __init__(self, latent_dim: int):
        super(VSDecoderv9, self).__init__()
        self.latent_dim = latent_dim
        # self.linear_up = nn.Linear(self.latent_dim, int(128 * 2 * 2 * 2))  #

        # self.up_conv7 = ae.UpConv(ch_in=512, ch_out=512, k_size=1, scale=2)
        self.conv7 = nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1)
        self.batchnorm3d7 = nn.BatchNorm3d(512)
        self.relu7 = nn.ReLU()
        self.sk7 = nn.ConvTranspose3d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample7 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        # self.conv7_1 = nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1)
        # self.relu7_1 = nn.ReLU()

        # self.up_conv6 = ae.UpConv(ch_in=512, ch_out=256, k_size=1, scale=2)
        self.conv6 = nn.Conv3d(512, 256, kernel_size=3, stride=1, padding=1)
        self.batchNorm3d6 = nn.BatchNorm3d(256)
        self.relu6 = nn.ReLU()
        self.sk6 = nn.ConvTranspose3d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample6 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        # self.conv6_1 = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)
        # self.relu6_1 = nn.ReLU()

        # self.up_conv5 = ae.UpConv(ch_in=256, ch_out=256, k_size=1, scale=2)
        self.conv5 = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)
        self.batchNorm3d5 = nn.BatchNorm3d(256)
        self.relu5 = nn.ReLU()
        self.sk5 = nn.ConvTranspose3d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample5 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        # self.conv5_1 = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)
        # self.relu5_1 = nn.ReLU()

        # self.up_conv4 = ae.UpConv(ch_in=256, ch_out=128, k_size=1, scale=2)
        self.conv4 = nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.sk4 = nn.ConvTranspose3d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.linear_down = nn.Conv3d(128, 1, kernel_size=1, stride=1, padding=0)
    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        # print("\n Decoder  x_in: ", x_in.shape)

        x_view = x_in.view(-1, 512, 2, 2, 2)  # 8*8*8 for 128?^3 and 2*2*2 for 32^3
        # print("\n Decoder  x_view: ", x_view.shape)

        b7 = self.conv7(x_view)
        b7 = self.batchnorm3d7(b7)
        # print("\n Decoder 7 b7: ", b7.shape)
        skip_7 = self.sk7(b7)
        # print("\n Decoder  7 skip_7: ", skip_7.shape)
        u7 = self.upsample7(b7)
        b7_1 = skip_7 + u7
        # print("\n Decoder 7 u7: ", u7.shape)
        u7_1 = self.relu7(b7_1)
        # -----------------------------------------
        b6 = self.conv6(u7_1)
        b6 = self.batchNorm3d6(b6)
        # print("\n Decoder 6 b6: ", b6.shape)
        skip6 = self.sk6(b6)
        # print("\n Decoder 6 skip6: ", skip6.shape)
        u6 = self.upsample6(b6)
        # print("\n Decoder 6 u6: ", u6.shape)
        b6_1 = skip6 + u6
        u6_1 = self.relu6(b6_1)
        # -----------------------------------------
        b5 = self.conv5(u6_1)
        b5 = self.batchNorm3d5(b5)
        # print("\n Decoder 5 b5: ", b5.shape)
        skip_5 = self.sk5(b5)
        # print("\n Decoder 5 skip_5: ", skip_5.shape)
        u5 = self.upsample5(b5)
        # print("\n Decoder 5 u5: ", u5.shape)
        b5_1 = u5 + skip_5
        u5_1 = self.relu5(b5_1)
        # -----------------------------------------
        b4 = self.conv4(u5_1)
        # print("\n Decoder b 4: ", b.shape)
        skip_4 = self.sk4(b4)
        # print("\n Decoder 4 skip_: ", skip_.shape)
        u4 = self.upsample4(b4)
        # print("\n Decoder 4 u: ", u.shape)
        b4_1 = skip_4 + u4
        u4_1 = self.relu4(b4_1)
        # -----------------------------------------
        result4_mapped = self.linear_down(u4_1)
        # print("\n Decoder x_out: ", x_out.shape)
        x_out = result4_mapped
        return x_out


