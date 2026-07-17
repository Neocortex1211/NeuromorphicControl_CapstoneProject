import torch
import torch.nn as nn


class CerebellumNet(nn.Module):
    """
    Feedforward neural network approximating cerebellar
    feedforward torque corrections.

    Inputs (6)
    ----------
    q1
    q2
    dq1
    dq2
    q1_des
    q2_des

    Outputs (2)
    -----------
    Δτ1
    Δτ2
    """

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.network(x)