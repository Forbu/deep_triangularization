"""
Script to compare 2D convolutions
"""

import sys
sys.path.append("../")

# add the path to the deep_triangularization package
from deep_triangularization.layers_multiheads import HeadLinear
from torch import nn
import torch
import time



def compare_conv2d():

    hidden_dim = 1024

    # initialize the input
    x = torch.randn(10000, hidden_dim)

    # initialize the convolutions
    layer1 = nn.Linear(hidden_dim, hidden_dim)
    layer2 = HeadLinear(hidden_dim, 16, random_rows=False)

    with torch.no_grad():

        # now we compare the forward pass
        start = time.time()
        out1 = layer1(x)
        end = time.time()
        print("Time for conv1: ", end - start)

        start = time.time()
        out2 = layer2(x)
        end = time.time()
        print("Time for conv2: ", end - start)



compare_conv2d()
