import torch
from torch import nn
import numpy as np
import functools
from collections import OrderedDict
from modules import *


class Generator(nn.Module):

    def __init__(self, latent_size, mapping_layers, min_resolution, max_resolution, max_channels, min_channels, out_channels):

        super().__init__()

        num_layers = int(np.log2(max_resolution // min_resolution))
        def num_channels(n): return min(max_channels, min_channels << (num_layers - n))

        self.mapping_network = nn.ModuleDict(OrderedDict(
            conv_blocks=nn.ModuleList([
                nn.ModuleDict(OrderedDict(
                    conv2d=Conv2d(
                        in_channels=latent_size,
                        out_channels=latent_size,
                        kernel_size=1,
                        padding=0,
                        stride=1,
                        bias=True,
                        variance_scale=2,
                        weight_scale=True
                    ),
                    leaky_relu=nn.LeakyReLU(0.2)
                ))
                for i in range(mapping_layers)
            ])
        ))

        self.synthesis_network = nn.ModuleDict(OrderedDict(
            conv_block=nn.ModuleDict(OrderedDict(
                first=nn.ModuleDict(OrderedDict(
                    leaned_constant=LearnedConstant(
                        num_channels=num_channels(0),
                        resolution=min_resolution
                    ),
                    learned_noise=LearnedNoise(
                        num_channels=num_channels(0)
                    ),
                    leaky_relu=nn.LeakyReLU(0.2),
                    adaptive_instance_norm=AdaptiveInstanceNorm(
                        in_channels=latent_size,
                        out_channels=num_channels(0),
                        bias=True,
                        variance_scale=1,
                        weight_scale=True
                    )
                )),
                second=nn.ModuleDict(OrderedDict(
                    conv2d=Conv2d(
                        in_channels=num_channels(0),
                        out_channels=num_channels(0),
                        kernel_size=3,
                        padding=1,
                        stride=1,
                        bias=True,
                        variance_scale=2,
                        weight_scale=True
                    ),
                    learned_noise=LearnedNoise(
                        num_channels=num_channels(0)
                    ),
                    leaky_relu=nn.LeakyReLU(0.2),
                    adaptive_instance_norm=AdaptiveInstanceNorm(
                        in_channels=latent_size,
                        out_channels=num_channels(0),
                        bias=True,
                        variance_scale=1,
                        weight_scale=True
                    )
                ))
            )),
            conv_blocks=nn.ModuleList([
                nn.ModuleDict(OrderedDict(
                    first=nn.ModuleDict(OrderedDict(
                        conv_transpose2d=ConvTranspose2d(
                            in_channels=num_channels(n),
                            out_channels=num_channels(n + 1),
                            kernel_size=4,
                            padding=1,
                            stride=2,
                            bias=True,
                            variance_scale=2,
                            weight_scale=True
                        ),
                        learned_noise=LearnedNoise(
                            num_channels=num_channels(n + 1)
                        ),
                        leaky_relu=nn.LeakyReLU(0.2),
                        adaptive_instance_norm=AdaptiveInstanceNorm(
                            in_channels=latent_size,
                            out_channels=num_channels(n + 1),
                            bias=True,
                            variance_scale=1,
                            weight_scale=True
                        )
                    )),
                    second=nn.ModuleDict(OrderedDict(
                        conv2d=Conv2d(
                            in_channels=num_channels(n + 1),
                            out_channels=num_channels(n + 1),
                            kernel_size=3,
                            padding=1,
                            stride=1,
                            bias=True,
                            variance_scale=2,
                            weight_scale=True
                        ),
                        learned_noise=LearnedNoise(
                            num_channels=num_channels(n + 1)
                        ),
                        leaky_relu=nn.LeakyReLU(0.2),
                        adaptive_instance_norm=AdaptiveInstanceNorm(
                            in_channels=latent_size,
                            out_channels=num_channels(n + 1),
                            bias=True,
                            variance_scale=1,
                            weight_scale=True
                        )
                    ))
                )) for n in range(num_layers)
            ]),
            color_block=nn.ModuleDict(OrderedDict(
                conv2d=Conv2d(
                    in_channels=num_channels(num_layers),
                    out_channels=out_channels,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                    bias=True,
                    variance_scale=1,
                    weight_scale=True
                ),
                tanh=nn.Tanh()
            ))
        ))

    def forward(self, inputs):

        for conv_block in self.mapping_network.conv_blocks:
            inputs = conv_block.conv2d(inputs)
            inputs = conv_block.leaky_relu(inputs)

        outputs = self.synthesis_network.conv_block.first.leaned_constant(inputs)
        outputs = self.synthesis_network.conv_block.first.learned_noise(outputs)
        outputs = self.synthesis_network.conv_block.first.leaky_relu(outputs)
        outputs = self.synthesis_network.conv_block.first.adaptive_instance_norm(outputs, inputs)

        outputs = self.synthesis_network.conv_block.second.conv2d(outputs)
        outputs = self.synthesis_network.conv_block.second.learned_noise(outputs)
        outputs = self.synthesis_network.conv_block.second.leaky_relu(outputs)
        outputs = self.synthesis_network.conv_block.second.adaptive_instance_norm(outputs, inputs)

        for conv_block in self.synthesis_network.conv_blocks:

            outputs = conv_block.first.conv_transpose2d(outputs)
            outputs = conv_block.first.learned_noise(outputs)
            outputs = conv_block.first.leaky_relu(outputs)
            outputs = conv_block.first.adaptive_instance_norm(outputs, inputs)

            outputs = conv_block.second.conv2d(outputs)
            outputs = conv_block.second.learned_noise(outputs)
            outputs = conv_block.second.leaky_relu(outputs)
            outputs = conv_block.second.adaptive_instance_norm(outputs, inputs)

        outputs = self.synthesis_network.color_block.conv2d(outputs)
        outputs = self.synthesis_network.color_block.tanh(outputs)

        return outputs


class Discriminator(nn.Module):

    def __init__(self, in_channels, min_channels, max_channels, max_resolution, min_resolution, num_classes):

        super().__init__()

        num_layers = int(np.log2(max_resolution // min_resolution))
        def num_channels(n): return min(max_channels, min_channels << (num_layers - n))

        self.network = nn.ModuleDict(OrderedDict(
            color_block=nn.ModuleDict(OrderedDict(
                conv2d=Conv2d(
                    in_channels=in_channels,
                    out_channels=num_channels(num_layers),
                    kernel_size=1,
                    padding=0,
                    stride=1,
                    bias=True,
                    variance_scale=2,
                    weight_scale=True
                ),
                leaky_relu=nn.LeakyReLU(0.2)
            )),
            conv_blocks=nn.ModuleList([
                nn.ModuleDict(OrderedDict(
                    first=nn.ModuleDict(OrderedDict(
                        conv2d=Conv2d(
                            in_channels=num_channels(n + 1),
                            out_channels=num_channels(n + 1),
                            kernel_size=3,
                            padding=1,
                            stride=1,
                            bias=True,
                            variance_scale=2,
                            weight_scale=True
                        ),
                        leaky_relu=nn.LeakyReLU(0.2)
                    )),
                    second=nn.ModuleDict(OrderedDict(
                        conv2d=Conv2d(
                            in_channels=num_channels(n + 1),
                            out_channels=num_channels(n),
                            kernel_size=4,
                            padding=1,
                            stride=2,
                            bias=True,
                            variance_scale=2,
                            weight_scale=True
                        ),
                        leaky_relu=nn.LeakyReLU(0.2)
                    ))
                )) for n in range(num_layers)[::-1]
            ]),
            conv_block=nn.ModuleDict(OrderedDict(
                first=nn.ModuleDict(OrderedDict(
                    batch_std=BatchStd(groups=4),
                    conv2d=Conv2d(
                        in_channels=num_channels(0) + 1,
                        out_channels=num_channels(0),
                        kernel_size=3,
                        padding=1,
                        stride=1,
                        bias=True,
                        variance_scale=2,
                        weight_scale=True
                    ),
                    leaky_relu=nn.LeakyReLU(0.2)
                )),
                second=nn.ModuleDict(OrderedDict(
                    global_aberage_pooling=nn.AdaptiveAvgPool2d(1),
                    conv2d=Conv2d(
                        in_channels=num_channels(0),
                        out_channels=num_classes,
                        kernel_size=1,
                        padding=0,
                        stride=1,
                        bias=True,
                        variance_scale=1,
                        weight_scale=True
                    )
                ))
            ))
        ))

    def forward(self, inputs):

        outputs = self.network.color_block.conv2d(inputs)
        outputs = self.network.color_block.leaky_relu(outputs)

        for conv_block in self.network.conv_blocks:

            outputs = conv_block.first.conv2d(outputs)
            outputs = conv_block.first.leaky_relu(outputs)

            outputs = conv_block.second.conv2d(outputs)
            outputs = conv_block.second.leaky_relu(outputs)

        outputs = torch.cat((outputs, self.network.conv_block.first.batch_std(outputs)), dim=1)
        outputs = self.network.conv_block.first.conv2d(outputs)
        outputs = self.network.conv_block.first.leaky_relu(outputs)

        outputs = self.network.conv_block.second.global_aberage_pooling(outputs)
        outputs = self.network.conv_block.second.conv2d(outputs)

        return outputs
