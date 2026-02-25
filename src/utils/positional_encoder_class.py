import torch
import pytorch_lightning as pl
from positional_encodings.torch_encodings import (
    PositionalEncoding3D,
)
from einops import rearrange

class MYPositionalEncoder3D(pl.LightningModule):
    def __init__(self, channels):
        super(MYPositionalEncoder3D, self).__init__()
        self.positional_encoder = PositionalEncoding3D(channels)

    def forward(self, shape_of_positions: list) -> torch.Tensor:
        # B, SqL_l, Ch = masked_optimized_latent_codes_reshaped.shape
        batch_size = shape_of_positions[0]
        D = shape_of_positions[1]
        H = shape_of_positions[2]
        W = shape_of_positions[3]
        Ch = shape_of_positions[4]

        z = torch.zeros((batch_size, D, H, W, Ch), dtype=torch.float32, device=self.device)
        z_positionally_encoded = self.positional_encoder(z)
        del z
        z_positionally_encoded_re = rearrange(z_positionally_encoded, "B D H W Ch -> B (D H W) Ch").to(self.device)
        del z_positionally_encoded

        return z_positionally_encoded_re



