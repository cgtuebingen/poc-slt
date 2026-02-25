import torch
from torchtyping import TensorType


class Distribution:
    """
    A Gaussian distribution.
        This class represents the given parameter tensor as gaussian distribution, which can be sampled easily.
        The parameters are split into the mean and variance of the distribution.
    Args:
        parameters:
            The parameter tensor of shape [Batch, 2 x Channels]
    """

    def __init__(self, parameters: TensorType["Batch", "TokenSize"]):
        """
        Args:
         parameters:
          The parameter tensor of shape [Batch, 2 x Channels]
        """
        self.parameters = parameters
        self.mean: TensorType["Batch", "DistSize"]
        self.logvar: TensorType["Batch", "DistSize"]
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(
            self.logvar, -30, 20
        )  # torch.clamp(self.logvar, -30, 20) was used for sdf
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self) -> TensorType["Batch", "DistSize"]:
        """
        Randomly sample the distribution.
        Returns:
              Tensor of shape [Batch, Channels].
        """
        rand_ = torch.randn(
            self.mean.shape, dtype=self.parameters.dtype, device=self.parameters.device
        )
        return self.mean + self.std * torch.randn(
            self.mean.shape, dtype=self.parameters.dtype, device=self.parameters.device
        )

    def KL(self, other=None) -> TensorType["Batch"]:
        """
        Computes the Kulbak Leibler Divergence between this distribution and another Distribution.
        This can be used during training of a VAE to sample the latent representation.
        Args:
            other:
             (Optional) Another distribution to which the KL divergence is computed.
              If `other` is not given the KL divergence to the standard normal distribution is returned.
        """
        if other is None:
            return 0.5 * torch.sum(
                self.mean * self.mean + self.var - 1 - self.logvar, dim=[1]
            )
        else:
            return 0.5 * torch.sum(
                (self.mean - other.mean) * (self.mean - other.mean) / other.var
                + self.var / other.var
                - 1
                - self.logvar
                + other.logvar,
                dim=[1],
            )

    def mode(self) -> TensorType["Batch", "DistSize"]:
        """
        Get the mean of the distribution.
        This is typically used during inference to get the latent representation.
        """
        return self.mean
