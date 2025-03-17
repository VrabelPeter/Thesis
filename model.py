import typing as tt

import torch
from torch import nn


class DQN(nn.Module):
    """The CNN architecture from the 2015 publication."""

    def __init__(self, input_shape: tt.Tuple[int, ...], n_actions: int) -> None:
        """Initialize the network.

        Args:
            `input_shape`: The shape of the input tensor (C, H, W).
            `n_actions`: The number of actions the agent can take.
        """
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        size = self.conv(torch.zeros(1, *input_shape)).size()[-1]
        self.fc = nn.Sequential(
            nn.Linear(size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        """Apply the network to the input tensor.

        Args:
            `x`: The input tensor of shape (B, C, H, W).
                Represents a batch of preprocessed images.

        Returns:
            The output tensor of shape (B, n_actions)
            corresponding to the Q-values of the actions.
        """
        return self.fc(self.conv(x))
