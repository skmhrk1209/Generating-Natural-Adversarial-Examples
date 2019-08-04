import torch
from torch import nn
import numpy as np


class Linear(nn.Module):

    def __init__(self, in_features, out_features, bias, variance_scale, weight_scale):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if weight_scale:
            self.std = 1.0
            self.scale = np.sqrt(variance_scale / in_features)
        else:
            self.std = np.sqrt(variance_scale / in_features)
            self.scale = 1.0
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=self.std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, inputs):
        return nn.functional.linear(
            input=inputs,
            weight=self.weight * self.scale,
            bias=self.bias
        )

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


class Embedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, variance_scale, weight_scale):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        if weight_scale:
            self.std = 1.0
            self.scale = np.sqrt(variance_scale / num_embeddings)
        else:
            self.std = np.sqrt(variance_scale / num_embeddings)
            self.scale = 1.0
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=self.std)

    def forward(self, inputs):
        return nn.functional.embedding(
            input=inputs,
            weight=self.weight * self.scale
        )

    def extra_repr(self):
        return f'num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}'


class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, bias, variance_scale, weight_scale):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        if weight_scale:
            self.std = 1.0
            self.scale = np.sqrt(variance_scale / in_channels / kernel_size / kernel_size)
        else:
            self.std = np.sqrt(variance_scale / in_channels / kernel_size / kernel_size)
            self.scale = 1.0
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=self.std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, inputs):
        return nn.functional.conv2d(
            input=inputs,
            weight=self.weight * self.scale,
            bias=self.bias,
            padding=self.padding,
            stride=self.stride
        )

    def extra_repr(self):
        return (f'in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, '
                f'padding={self.padding}, stride={self.stride}, bias={self.bias is not None}')


class ConvTranspose2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, bias, variance_scale, weight_scale):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size, kernel_size))
        if weight_scale:
            self.std = 1.0
            self.scale = np.sqrt(variance_scale / in_channels / kernel_size / kernel_size)
        else:
            self.std = np.sqrt(variance_scale / in_channels / kernel_size / kernel_size)
            self.scale = 1.0
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=self.std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, inputs):
        return nn.functional.conv_transpose2d(
            input=inputs,
            weight=self.weight * self.scale,
            bias=self.bias,
            padding=self.padding,
            stride=self.stride
        )

    def extra_repr(self):
        return (f'in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, '
                f'padding={self.padding}, stride={self.stride}, bias={self.bias is not None}')


class PixelNorm(nn.Module):

    def __init__(self, epsilon=1e-12):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, inputs):
        norm = torch.mean(inputs ** 2, dim=1, keepdim=True)
        norm = torch.sqrt(norm + self.epsilon)
        outputs = inputs / norm
        return outputs


class BatchStd(nn.Module):

    def __init__(self, groups, epsilon=1e-12):
        super().__init__()
        self.groups = groups
        self.epsilon = epsilon

    def forward(self, inputs):
        outputs = inputs.reshape(self.groups, -1, *inputs.shape[1:])
        outputs -= torch.mean(outputs, dim=0, keepdim=True)
        outputs = torch.mean(outputs ** 2, dim=0)
        outputs = torch.sqrt(outputs + self.epsilon)
        outputs = torch.mean(outputs, dim=(1, 2, 3), keepdim=True)
        outputs = outputs.repeat(self.groups, 1, *inputs.shape[2:])
        return outputs


class LearnedConstant(nn.Module):

    def __init__(self, num_channels, resolution):
        super().__init__()
        self.constant = nn.Parameter(torch.ones(1, num_channels, resolution, resolution))

    def forward(self, inputs):
        outputs = self.constant.repeat(inputs.shape[0], *(1 for _ in self.constant.shape[1:]))
        return outputs


class LearnedNoise(nn.Module):

    def __init__(self, num_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, inputs):
        noises = torch.randn(inputs.shape[0], 1, *inputs.shape[2:]).to(inputs.device)
        outputs = inputs + noises * self.weight
        return outputs


class AdaptiveInstanceNorm(nn.Module):

    def __init__(self, in_channels, out_channels, bias, variance_scale, weight_scale):
        super().__init__()
        self.instance_norm2d = nn.InstanceNorm2d(
            num_features=out_channels,
            affine=False
        )
        self.conv2d = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels * 2,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=bias,
            variance_scale=variance_scale,
            weight_scale=weight_scale
        )

    def forward(self, inputs, styles):
        outputs = self.instance_norm2d(inputs)
        gamma, beta = torch.chunk(self.conv2d(styles), chunks=2, dim=1)
        outputs = outputs * gamma + beta
        return outputs
