import torch
from typing import Tuple
from torchtyping import TensorType
import src.p_vae.distribution


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(torch.nn.functional.relu(1.0 - logits_real))
    loss_fake = torch.mean(torch.nn.functional.relu(1.0 - logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


class VAELoss(torch.nn.Module):
    def __init__(self, kulbak_leibler_weight: float = 1e-6, logvar_init: float = 0.0):
        super().__init__()
        self.logvar = torch.nn.Parameter(torch.ones(size=()) * logvar_init)
        self.hinge_loss = hinge_d_loss
        self.kulbak_leibler_weight = kulbak_leibler_weight

    def forward(
        self,
        gt_sequence: TensorType[
            "Batch", "BlockDuration", "BlockHeight", "BlockWidth", "RGB"
        ],
        pred_sequence: TensorType[
            "Batch", "BlockDuration", "BlockHeight", "BlockWidth", "RGB"
        ],
        posterior: src.p_vae.distribution.Distribution,
        global_step: int,
        log_prefix: str = "training",
    ) -> Tuple[TensorType[1], dict["str", "torch.Tensor"]]:

        reconstruction_loss = torch.abs(gt_sequence - pred_sequence).mean()  # L1
        loss_log: dict = {f"{log_prefix}/l1_loss": reconstruction_loss.detach()}
        # likelihood

        # -log([0, 1]) ==> [0, inf]

        negative_log_likelihood_loss = (
            reconstruction_loss / torch.exp(self.logvar) + self.logvar
        )  # L1 loss as main loss, converts it into a negative log likelihood
        kullback_leibler_loss = posterior.KL()
        kullback_leibler_loss = (
            torch.sum(kullback_leibler_loss) / kullback_leibler_loss.shape[0]
        )
        loss_log[f"{log_prefix}/kullback_leibler_loss"] = kullback_leibler_loss.detach()
        loss_log[f"{log_prefix}/logvar"] = self.logvar.detach()
        loss_log[f"{log_prefix}/nll_loss"] = negative_log_likelihood_loss.detach()
        loss_log[f"{log_prefix}/reconstruction_loss"] = reconstruction_loss.detach()

        loss = (
            negative_log_likelihood_loss
            + kullback_leibler_loss * self.kulbak_leibler_weight
        )
        # loss_log[f"{log_prefix}/loss"] = loss
        return loss, loss_log


class LossPerVoxel(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self,
        gt_sequence: TensorType["Batch", "BlockDuration", "BlockHeight", "BlockWidth"],
        pred_sequence: TensorType[
            "Batch", "BlockDuration", "BlockHeight", "BlockWidth"
        ],
    ) -> TensorType["Batch", 1]:

        reconstruction_loss_per_batch = torch.empty([gt_sequence.shape[0], 1])
        for b in range(gt_sequence.shape[0]):
            reconstruction_loss_per_batch[b] = torch.abs(
                gt_sequence[b] - pred_sequence[b]
            ).mean()  # L1

        return reconstruction_loss_per_batch
