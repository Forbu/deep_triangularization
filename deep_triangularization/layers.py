"""
In this module we will create the torch module that will optimize the linear layers of the network.
"""
import math

import torch
from torch import nn

from einops import rearrange


class Triangle(nn.Module):
    """
    Simple class that instead of doing a linear transformation, does a triangularization.
    """

    def __init__(self, in_dim, out_dim, upper=True, bias=True, random_rows=True):
        super(Triangle, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.Tensor(out_dim, in_dim))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_dim))

        self.upper = upper

        # compute a triangular mask
        mask = (
            torch.triu(torch.ones(out_dim, in_dim), diagonal=0)
            if upper
            else torch.tril(torch.ones(out_dim, in_dim), diagonal=0)
        )

        # if we want to randomize the rows of the mask
        if random_rows:
            mask = mask[torch.randperm(out_dim)]

        # we register the mask as a buffer so that it is moved to the device along with the module
        self.register_buffer("mask", mask)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # we apply the triangular mask to the weight matrix
        weight = self.weight * self.mask

        if self.bias is None:
            return nn.functional.linear(input, weight)
        else:
            return nn.functional.linear(input, weight, self.bias)


class HeadLinear(nn.Module):
    """
    HeadLinear class which performe diagonal block multiplication in a cleaver way
    """

    def __init__(self, hidden_dim, nb_head, random_rows=True):
        """
        args:
            hidden_dim: int, the hidden dimension of the input
            nb_head: int, the number of head to use

        """
        super(HeadLinear, self).__init__()
        self.hidden_dim = hidden_dim
        self.nb_head = nb_head
        self.random_rows = random_rows

        assert hidden_dim % nb_head == 0, "hidden_dim must be divisible by nb_head"

        # if we want to randomize the rows of the mask
        if random_rows:
            # we register a random permutation of the rows as a buffer so that it is moved to the device along with the module
            permutation = torch.randperm(hidden_dim)
            inverse_permutation = torch.argsort(permutation)

            self.register_buffer("permutation", permutation)
            self.register_buffer("inverse_permutation", inverse_permutation)

        self.weights = nn.Parameter(torch.randn(
            nb_head, hidden_dim//nb_head, hidden_dim//nb_head))

    def forward(self, x):
        """
        forward pass of the HeadLinear module
        """
        if self.random_rows:
            # we apply the permutation to the input
            x = x[:, self.permutation]

        x = rearrange(x, 'b (nb_head d) -> b nb_head d', nb_head=self.nb_head)
        out = torch.einsum('bmd,mdk->bmk', x, self.weights)
        out = rearrange(out, 'b nb_head k -> b (nb_head k)')

        if self.random_rows:
            # we apply the inverse permutation to the output
            out = out[:, self.inverse_permutation]

        return out
