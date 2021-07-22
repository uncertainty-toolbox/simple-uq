""" Basic implementation of a Probabilistic Neural Network (PNN). This is a
neural network that outputs the mean and variance of a standard normal.
"""
from typing import Callable, NoReturn, Sequence, Tuple

import numpy as np
from pytorch_lightning import LightningModule
import torch
import torch.nn.functional as F

from simple_uq.util.mlp import MLP


class PNN(LightningModule):
    """
    A probabilistic neural network implemented as a two headed neuural
    network. The two heads output the mean and logvariance of a multi-variate
    norma.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        # Parameters for the encoder network.
        encoder_hidden_sizes: Sequence[int],
        encoder_output_dim: int,
        # Parameters for mean and logvar heads.
        mean_hidden_sizes: Sequence[int],
        logvar_hidden_sizes: Sequence[int],
        hidden_activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        learning_rate: float = 1e-3,
    ):
        """Constructor.
        Args:
            input_dim: Dimension of input data.
            output_dim: Dimesnion of data outputted.
            hidden_activation: Hidden activation function.
            encoder_hidden_sizes: List of the hidden sizes for the encoder.
            encoder_output_dim: Dimension of the data outputted by the encoder.
            mean_hidden_sizes: List of hidden sizes for mean head.
            logvar_hidden_sizes: List of hidden sizes for logvar head.
        """
        super().__init__()
        self._learning_rate = learning_rate
        self.encoder = MLP(
            input_dim=input_dim,
            output_dim=encoder_output_dim,
            hidden_sizes=encoder_hidden_sizes,
            hidden_activation=hidden_activation,
        )
        self.mean_head = MLP(
            input_dim=encoder_output_dim,
            output_dim=output_dim,
            hidden_sizes=mean_hidden_sizes,
            hidden_activation=hidden_activation,
        )
        self.logvar_head = MLP(
            input_dim=encoder_output_dim,
            output_dim=output_dim,
            hidden_sizes=logvar_hidden_sizes,
            hidden_activation=hidden_activation,
        )

    def get_mean_and_standard_deviation(
        self,
        x_data: np.ndarray,
        device: str = "cpu",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the mean and standard deviation prediction.
        Args:
            x_data: The data in numpy ndarray form.
            device: The device to use. Should be the same as the device
                the model is currently on.
        Returns: Mean and standard deviation as ndarrays
        """
        with torch.no_grad():
            mean, logvar = self.forward(torch.Tensor(x_data, device=device))
        mean = mean.numpy()
        std = (logvar / 2).exp().numpy()
        return mean, std

    def forward(
        self,
        x_data: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the mean and standard deviation prediction.
        Args:
            x_data: The data in tensor form.
        Returns: Mean and log variance as tensors.
        """
        latent = self.encoder(x_data)
        return self.mean_head(latent), self.logvar_head(latent)

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Do a training step.
        Args:
            batch: The x and y data to train on.
            batch_idx: Index of he batch.
        Returns: The loss.
        """
        x_data, y_data = batch
        mean, logvar = self.forward(x_data)
        loss = torch.mean(self.compute_nll(mean, logvar, y_data))
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> NoReturn:
        """Do a validation step.
        Args:
            batch: The x and y data to train on.
            batch_idx: Index of he batch.
        """
        x_data, y_data = batch
        mean, logvar = self.forward(x_data)
        loss = torch.mean(self.compute_nll(mean, logvar, y_data))
        self.log("validation_loss", loss)

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> NoReturn:
        """Do a validation step.
        Args:
            batch: The x and y data to train on.
            batch_idx: Index of he batch.
        """
        x_data, y_data = batch
        mean, logvar = self.forward(x_data)
        loss = torch.mean(self.compute_nll(mean, logvar, y_data))
        self.log("test_loss", loss)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer.
        Returns: Optimizer
        """
        return torch.optim.Adam(self.parameters(), lr=self._learning_rate)

    def compute_nll(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the loss as negative log likelihood.
        Args:
            mean: The mean prediction for labels.
            logvar: The logvariance prediction for labels.
            labels: The observed labels of the data.
        Returns: The negative log likelihood of each point.
        """
        sqdiffs = (mean - labels) ** 2
        return torch.exp(-logvar) * sqdiffs + logvar
