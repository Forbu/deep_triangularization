"""
In this module we will create the torch module that will optimize the linear layers of the network.
And also the module that will optimize the triangularization of the network.
"""

import math
from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT

import torch
from torch import nn

import lightning as pl

from deep_triangularization.layers import Triangle

class MLP_dense(nn.Module):
    """
    Simple MLP with dense layers.
    """

    def __init__(self, in_dim, out_dim, hidden_dim=128, num_layers=5):
        super(MLP_dense, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList(
            [
                nn.Linear(in_dim, hidden_dim),
                *[nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)],
                nn.Linear(hidden_dim, out_dim),
            ]
        )

    def forward(self, input):

        for layer in self.layers[:-1]:
            input = nn.functional.relu(layer(input))

        return self.layers[-1](input)
    

    
class MLP_triangular(pl.LightningModule):
    """
    MLP with triangular layers (hidden layers only).
    """
    
        def __init__(self, in_dim, out_dim, hidden_dim=128, num_layers=5):
            super(MLP_triangular, self).__init__()
            self.in_dim = in_dim
            self.out_dim = out_dim
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
    
            self.layers = nn.ModuleList(
                [
                    nn.Linear(in_dim, hidden_dim),
                    *[Triangle(hidden_dim, hidden_dim) for _ in range(num_layers - 2)],
                    nn.Linear(hidden_dim, out_dim),
                ]
            )
    
        def forward(self, input):
    
            for layer in self.layers[:-1]:
                input = nn.functional.relu(layer(input))
    
            return self.layers[-1](input)
