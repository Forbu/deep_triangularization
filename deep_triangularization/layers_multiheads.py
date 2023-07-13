import math

import torch
from torch import nn

from einops import rearrange, repeat


class HeadLinear(nn.Module):
    """
    HeadLinear class which performe diagonal block multiplication in a clever way
    """

    def __init__(self, hidden_dim, nb_head, random_rows=True):
        """
        args:
            hidden_dim: int, the hidden dimension of the input
            nb_head: int, the number of head to use
            random_rows: bool, if True, the rows of the input are randomly permuted

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

        self.weights = nn.Parameter(
            torch.randn(nb_head, hidden_dim // nb_head, hidden_dim // nb_head)
        )

        self.bias = nn.Parameter(torch.zeros(hidden_dim))

        # we initialize the weights and bias
        self.reset_parameters()

    def reset_parameters(self):
        """
        This method is used to initialize the weights and bias
        """
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        forward pass of the HeadLinear module
        """
        if self.random_rows:
            # we apply the permutation to the input
            x = x[:, self.permutation]

        x = rearrange(x, "b (nb_head d) -> b nb_head d", nb_head=self.nb_head)
        out = torch.einsum("bmd,mdk->bmk", x, self.weights)
        out = rearrange(out, "b nb_head k -> b (nb_head k)")

        # if self.random_rows:
        # we apply the inverse permutation to the output
        #   out = out[:, self.inverse_permutation]

        return out + self.bias.view(1, -1)


class BlockDiagonalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, m):
        super(BlockDiagonalConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.m = m
        self.weights = nn.Parameter(
            torch.randn(
                m, out_channels // m, in_channels // m, kernel_size, kernel_size
            )
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = rearrange(x, "b (c m) h w -> b m c h w", m=self.m)
        x = torch.zeros_like(x)

        for i in range(self.m):
            x[:, i, :, :, :] = nn.functional.conv2d(
                x[:, i, :, :, :],
                self.weights[i],
                bias=None,
                stride=1,
                padding=self.kernel_size // 2,
                dilation=1,
                groups=1,
            )

        x = rearrange(x, "b m c h w -> b (c m) h w", m=self.m)

        return x + self.bias.view(1, -1, 1, 1)


def head_conv2d(x, weights, bias, kernel_size):
    """
    convolution of a head using functional conv2d
    x: (b, c, h, w)
    weights: (c, c, kernel_size, kernel_size)
    bias: (c)

    """

    result = nn.functional.conv2d(
        x, weights, bias=bias, stride=1, padding=kernel_size // 2, dilation=1, groups=1
    )

    return result


head_conv2d_vmap = torch.vmap(head_conv2d, in_dims=(0, 0, 0, None), out_dims=0)


class BlockDiagonalConv2d_v2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, m):
        super(BlockDiagonalConv2d_v2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.m = m
        self.weights = nn.Parameter(
            torch.randn(
                m, out_channels // m, in_channels // m, kernel_size, kernel_size
            )
        )

        self.bias = nn.Parameter(torch.zeros(m, out_channels // m))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = rearrange(x, "b (c m) h w -> m b c h w", m=self.m)

        res = head_conv2d_vmap(x, self.weights, self.bias, self.kernel_size)

        res = rearrange(res, "m b c h w -> b (c m) h w", m=self.m)

        return res
