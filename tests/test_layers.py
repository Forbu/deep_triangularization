"""
Simple test to check that the triangular layers are working as expected.
"""

import pytest

import torch
from torch import nn

from deep_triangularization.layers import Triangle

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
    layer = Triangle(DIM_IN, DIM_OUT)

    # check that the layer is initialized correctly
    assert layer.weight.shape == (DIM_OUT, DIM_IN)
    assert layer.bias.shape == (DIM_OUT,)
    assert layer.mask.shape == (DIM_OUT, DIM_IN)

    # check that the forward pass works
    output = layer(input)

    assert output.shape == (BATCH_SIZE, DIM_OUT)