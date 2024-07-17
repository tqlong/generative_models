from typing import Tuple
import torch
from torch import nn
from functools import reduce


class SimpleEncoder(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 32,
    ) -> None:
        """Initialize a `SimpleEncoder` module.

        :param input_size: The number of input features.
        :param lin1_size: The number of output features of the first linear layer.
        :param lin2_size: The number of output features of the second linear layer.
        :param lin3_size: The number of output features of the third linear layer.
        :param output_size: The number of output features of the final linear layer.
        """
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.ReLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin3_size),
            nn.BatchNorm1d(lin3_size),
            nn.ReLU(),
        )
        self.mu = nn.Linear(lin3_size, output_size)
        self.logvar = nn.Linear(lin3_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        batch_size = x.shape[0]

        # (batch, 1, width, height) -> (batch, 1*width*height)
        x = x.view(batch_size, -1)

        feature = self.feature(x)
        return self.mu(feature), self.logvar(feature)


class SimpleDecoder(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        input_size: int = 32,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_shape: Tuple[int, int] = (1, 28, 28),
    ) -> None:
        """Initialize a `SimpleDecoder` module.

        :param input_size: The number of input features.
        :param lin1_size: The number of output features of the first linear layer.
        :param lin2_size: The number of output features of the second linear layer.
        :param lin3_size: The number of output features of the third linear layer.
        :param output_size: The number of output features of the final linear layer.
        """
        super().__init__()

        output_size = reduce(lambda x, y: x * y, output_shape)
        self.input_size = input_size
        self.output_shape = output_shape
        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.ReLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin3_size),
            nn.BatchNorm1d(lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        batch_size = x.shape[0]

        # (batch, 1, width, height) -> (batch, 1*width*height)
        x = x.view(batch_size, -1)

        out = self.model(x)
        return out.view(batch_size, *self.output_shape)


if __name__ == "__main__":
    model = SimpleDecoder()
    x = torch.randn(10, 32)
    out = model(x)
    print(out.shape)

    # model = SimpleEncoder()
    # x = torch.randn(10, 28, 28)
    # mu, logvar = model(x)
    # print(mu.shape, logvar.shape)
