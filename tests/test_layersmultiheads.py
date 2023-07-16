"""
Simple test to check that the triangular layers are working as expected.
"""

import pytest

import torch
from torch import nn

from deep_triangularization.layers_multiheads import HeadLinear

DIM_IN = 32
BATCH_SIZE = 16
DIM_OUT = 64

# init fixtures
@pytest.fixture
def input():
    return torch.rand(BATCH_SIZE, DIM_IN)


def test_triangle(input):
    """
    Testing the triangular layer.
    """
    layer = HeadLinear(DIM_IN, DIM_OUT, nb_head=4)

    # check that the forward pass works
    output = layer(input)

    assert output.shape == (BATCH_SIZE, DIM_OUT)
