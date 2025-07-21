import math
import torch
from functools import reduce
from operator import __add__
from .layers import Output


class conv3d_padding_same(torch.nn.Module):
    """
    Padding so that the next Conv3d layer outputs an array with the same dimension as the input.
    Depth, height and width are the kernel dimensions.

    Example:
    batch_size = 8
    in_channels = 3
    out_channel = 16
    kernel_size = (2, 3, 5)
    stride = 1  # could also be 2, or 3, etc.
    pad_value = 0
    conv = torch.nn.Conv3d(in_channels, out_channel, kernel_size, stride=stride)

    x = torch.empty(batch_size, in_channels, 100, 100, 100)
    conv_padding = reduce(__add__, [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_size[::-1]])
    out = F.pad(x, conv_padding, 'constant', pad_value)

    out = conv(out)
    print(out.shape): torch.Size([8, 16, 100, 100, 100])

    Source: https://stackoverflow.com/questions/58307036/is-there-really-no-padding-same-option-for-pytorchs-conv2d
    Source: https://pytorch.org/docs/master/generated/torch.nn.functional.pad.html#torch.nn.functional.pad
    """

    def __init__(self, depth, height, width, pad_value):
        super(conv3d_padding_same, self).__init__()
        self.kernel_size = (depth, height, width)
        self.pad_value = pad_value

    def forward(self, x):
        # Determine amount of padding
        # Internal parameters used to reproduce Tensorflow "Same" padding.
        # For some reasons, padding dimensions are reversed wrt kernel sizes.
        conv_padding = reduce(__add__, [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]])
        x_padded = torch.nn.functional.pad(x, conv_padding, 'constant', self.pad_value)

        return x_padded


class reshape_tensor(torch.nn.Module):
    """
    Reshape tensor.
    """
    def __init__(self, *args):
        super(reshape_tensor, self).__init__()
        self.output_dim = []
        for a in args:
            self.output_dim.append(a)

    def forward(self, x, batch_size):
        output_dim = [batch_size] + self.output_dim
        x = x.view(output_dim)

        return x


def conv3x3x3(in_planes, out_planes, stride=1):
    return torch.nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return torch.nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicResBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, lrelu_alpha, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(in_planes, in_planes, stride)
        self.norm1 = torch.nn.InstanceNorm3d(in_planes)
        self.act = torch.nn.LeakyReLU(negative_slope=lrelu_alpha)
        self.conv2 = conv3x3x3(in_planes, in_planes * self.expansion)
        self.norm2 = torch.nn.InstanceNorm3d(in_planes * self.expansion)
        self.downsample = downsample
        self.stride = stride
        self.conv3 = conv1x1x1(in_planes * self.expansion, planes)
        self.norm3 = torch.nn.InstanceNorm3d(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act(out)

        out = self.conv3(out)
        out = self.norm3(out)
        out = self.act(out)

        return out


class Bottleneck(torch.nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, lrelu_alpha, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1x1(in_planes, planes)
        self.norm1 = torch.nn.InstanceNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.norm2 = torch.nn.InstanceNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.norm3 = torch.nn.InstanceNorm3d(planes * self.expansion)
        self.act = torch.nn.LeakyReLU(negative_slope=lrelu_alpha)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act(out)

        return out


class pooling_conv(torch.nn.Module):
    def __init__(self, in_channels, filters, kernel_size, strides, lrelu_alpha, use_bias=False):
        super(pooling_conv, self).__init__()
        self.conv1 = torch.nn.Conv3d(in_channels=in_channels, out_channels=filters, kernel_size=kernel_size,
                                     stride=strides, bias=use_bias)

    def forward(self, x):
        x = self.conv1(x)
        return x


class ResNet_MP_LReLU(torch.nn.Module):
    """
    ResNet + MP
    """

    def __init__(self, n_input_channels, depth, height, width, n_features, num_classes, filters, kernel_sizes, strides,
                 pad_value, n_down_blocks, lrelu_alpha, dropout_p, pooling_conv_filters, perform_pooling,
                 linear_units, use_bias=False):
        super(ResNet_MP_LReLU, self).__init__()
        self.n_features = n_features
        self.pooling_conv_filters = pooling_conv_filters
        self.perform_pooling = perform_pooling

        # Initialize conv blocks
        in_channels = [n_input_channels] + list(filters[:-1])
        # Determine Linear input channel size
        end_depth = math.floor(depth / (2 ** len(in_channels)))
        end_height = math.floor(height / (2 ** len(in_channels)))
        end_width = math.floor(width / (2 ** len(in_channels)))

        self.blocks = torch.nn.ModuleList()
        for i in range(len(in_channels)):
            self.blocks.add_module('resblock%s' % i, BasicResBlock(in_planes=in_channels[i], planes=filters[i],
                                                                   lrelu_alpha=lrelu_alpha))
            self.blocks.add_module('max_pooling%s' % i,
                                   torch.nn.MaxPool3d(kernel_size=2, stride=2, padding=pad_value, dilation=1))

        # Initialize pooling conv
        if self.pooling_conv_filters is not None:
            pooling_conv_kernel_size = [end_depth, end_height, end_width]
            self.pool = pooling_conv(in_channels=filters[-1], filters=pooling_conv_filters,
                                     kernel_size=pooling_conv_kernel_size, strides=1,
                                     lrelu_alpha=lrelu_alpha, use_bias=use_bias)
            end_depth, end_height, end_width = 1, 1, 1
            filters[-1] = self.pooling_conv_filters
        elif self.perform_pooling:
            self.pool = torch.nn.Conv3d(filters[i], filters[i], kernel_size=(end_depth, end_height, end_width))
            end_depth, end_height, end_width = 1, 1, 1
        self.act = torch.nn.LeakyReLU(negative_slope=lrelu_alpha)

        # Initialize flatten layer
        self.flatten = torch.nn.Flatten()

        # Initialize linear blocks
        self.linear_layers = torch.nn.ModuleList()
        linear_units = [end_depth * end_height * end_width * filters[-1]] + linear_units
        for i in range(len(linear_units) - 1):
            self.linear_layers.add_module('dropout%s' % i, torch.nn.Dropout(dropout_p[i]))
            self.linear_layers.add_module('linear%s' % i,
                                          torch.nn.Linear(in_features=linear_units[i], out_features=linear_units[i+1],
                                                          bias=use_bias))
            # self.linear_layers.add_module('lrelu%s' % i, torch.nn.LeakyReLU(negative_slope=lrelu_alpha))

        # Initialize output layer
        self.out_layer = Output(in_features=linear_units[-1] + self.n_features, out_features=num_classes, bias=use_bias)
        # self.out_layer.__class__.__name__ = 'Output'

    def forward(self, x, features):
        # Blocks
        for block in self.blocks:
            x = block(x)

        # Pooling layers
        if (self.pooling_conv_filters is not None) or self.perform_pooling:
            x = self.pool(x)
            x = self.act(x)

        x = self.flatten(x)

        # Linear layers
        for layer in self.linear_layers:
            x = layer(x)

        # Add features
        if self.n_features > 0:
            x = torch.cat([x, features], dim=1)

        # Output
        x = self.out_layer(x)

        return x


