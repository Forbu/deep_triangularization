"""
Script to compare 2D convolutions
"""

import sys
sys.path.append("../")

# add the path to the deep_triangularization package
from deep_triangularization.layers_multiheads import BlockDiagonalConv2d, BlockDiagonalConv2d_v2
from torch import nn
import torch
import time



def compare_conv2d():

    # initialize the input
    x = torch.randn(64, 64, 60, 60)

    # initialize the convolutions
    conv1 = nn.Conv2d(64, 64, 3, padding=1)
    conv2 = BlockDiagonalConv2d(64, 64, 3, 8)
    conv3 = BlockDiagonalConv2d_v2(64, 64, 3, 8)

    with torch.no_grad():

        # now we compare the forward pass
        start = time.time()
        out1 = conv1(x)
        end = time.time()
        print("Time for conv1: ", end - start)

        start = time.time()
        out2 = conv2(x)
        end = time.time()
        print("Time for conv2: ", end - start)

        start = time.time()
        out3 = conv3(x)
        end = time.time()
        print("Time for conv3: ", end - start)


compare_conv2d()
