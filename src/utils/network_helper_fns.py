import torch
from torch import nn
import pytorch_lightning as pl

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