"""
Same as DCNN, but where LeakyReLU activation has been applied whenever possible.
"""
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


class conv_block(torch.nn.Module):
    def __init__(self, in_channels, filters, kernel_size, strides, pad_value,
                 lrelu_alpha, use_activation, use_bias=False):
        super(conv_block, self).__init__()

        if ((type(kernel_size) == list) or (type(kernel_size) == tuple)) and (len(kernel_size) == 3):
            kernel_depth = kernel_size[0]
            kernel_height = kernel_size[1]
            kernel_width = kernel_size[2]
        elif type(kernel_size) == int:
            kernel_depth = kernel_size
            kernel_height = kernel_size
            kernel_width = kernel_size
        else:
            raise ValueError("Kernel_size is invalid:", kernel_size)

        self.pad = conv3d_padding_same(depth=kernel_depth, height=kernel_height, width=kernel_width,
                                       pad_value=pad_value)
        self.conv1 = torch.nn.Conv3d(in_channels=in_channels, out_channels=filters, kernel_size=kernel_size,
                                     stride=strides, bias=use_bias)
        # DWS
        # self.conv1_1 = torch.nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
        #                                stride=strides, bias=use_bias, groups=in_channels)
        # self.conv1_2 = torch.nn.Conv3d(in_channels=in_channels, out_channels=filters, kernel_size=1,
        #                                stride=1, bias=use_bias)

        # self.conv1 = torch.nn.utils.parametrizations.spectral_norm(
        #     torch.nn.Conv3d(in_channels=in_channels, out_channels=filters, kernel_size=kernel_size, stride=strides,
        #                     bias=use_bias))
        # self.norm1 = torch.nn.BatchNorm3d(filters)
        self.norm1 = torch.nn.GroupNorm(1, filters)
        self.activation1 = torch.nn.LeakyReLU(negative_slope=lrelu_alpha)
        self.conv2 = torch.nn.Conv3d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, stride=1,
                                     bias=use_bias)
        # DWS
        # self.conv2_1 = torch.nn.Conv3d(in_channels=filters, out_channels=filters, kernel_size=kernel_size,
        #                                stride=1, bias=use_bias, groups=filters)
        # self.conv2_2 = torch.nn.Conv3d(in_channels=filters, out_channels=filters, kernel_size=1,
        #                                stride=1, bias=use_bias)

        # self.conv2 = torch.nn.utils.parametrizations.spectral_norm(
        #     torch.nn.Conv3d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, stride=1,
        #                     bias=use_bias))
        # self.norm2 = torch.nn.BatchNorm3d(filters)
        self.norm2 = torch.nn.GroupNorm(1, filters)
        self.use_activation = use_activation
        self.activation2 = torch.nn.LeakyReLU(negative_slope=lrelu_alpha)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv1(x)
        # DWS
        # x = self.conv1_1(x)
        # x = self.conv1_2(x)

        x = self.norm1(x)
        x = self.activation1(x)
        x = self.pad(x)

        x = self.conv2(x)
        # DWS
        # x = self.conv2_1(x)
        # x = self.conv2_2(x)

        x = self.norm2(x)

        if self.use_activation:
            x = self.activation2(x)
        return x


class pooling_conv(torch.nn.Module):
    def __init__(self, in_channels, filters, kernel_size, strides, lrelu_alpha, use_bias=False):
        super(pooling_conv, self).__init__()
        self.conv1 = torch.nn.Conv3d(in_channels=in_channels, out_channels=filters, kernel_size=kernel_size,
                                     stride=strides, bias=use_bias)
        # No InstanceNorm, because output size is (batch_size, filters, 1, 1, 1)
        # self.norm1 = torch.nn.InstanceNorm3d(filters)
        self.activation1 = torch.nn.LeakyReLU(negative_slope=lrelu_alpha)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.norm1(x)
        x = self.activation1(x)
        return x


class DCNN_LReLU_GN(torch.nn.Module):
    """
    Deep CNN
    """

    def __init__(self, n_input_channels, depth, height, width, n_features, num_classes, filters, kernel_sizes, strides,
                 pad_value, n_down_blocks, lrelu_alpha, dropout_p, pooling_conv_filters, perform_pooling,
                 linear_units, use_bias=False):
        super(DCNN_LReLU_GN, self).__init__()
        self.n_features = n_features
        self.pooling_conv_filters = pooling_conv_filters
        self.perform_pooling = perform_pooling

        # Determine Linear input channel size
        end_depth = math.ceil(depth / (2 ** n_down_blocks[0]))
        end_height = math.ceil(height / (2 ** n_down_blocks[1]))
        end_width = math.ceil(width / (2 ** n_down_blocks[2]))

        # Initialize conv blocks
        in_channels = [n_input_channels] + list(filters[:-1])
        self.conv_blocks = torch.nn.ModuleList()
        for i in range(len(in_channels)):
            # if i == len(in_channels) - 1:
            #     use_activation = False
            # else:
            #     use_activation = True
            use_activation = True
            self.conv_blocks.add_module('conv_block%s' % i,
                                        conv_block(in_channels=in_channels[i], filters=filters[i],
                                                   kernel_size=kernel_sizes[i], strides=strides[i],
                                                   pad_value=pad_value, lrelu_alpha=lrelu_alpha,
                                                   use_activation=use_activation, use_bias=use_bias))

        # Initialize pooling conv
        if self.pooling_conv_filters is not None:
            pooling_conv_kernel_size = [end_depth, end_height, end_width]
            self.pool = pooling_conv(in_channels=filters[-1], filters=pooling_conv_filters,
                                     kernel_size=pooling_conv_kernel_size, strides=1,
                                     lrelu_alpha=lrelu_alpha, use_bias=use_bias)
            end_depth, end_height, end_width = 1, 1, 1
            filters[-1] = self.pooling_conv_filters
        elif self.perform_pooling:
            # self.pool = torch.nn.AvgPool3d(kernel_size=(1, end_height, end_width))
            # end_depth, end_height, end_width = depth, 1, 1
            # self.pool = torch.nn.AvgPool3d(kernel_size=(end_depth, end_height, end_width))
            self.pool = torch.nn.MaxPool3d(kernel_size=(end_depth, end_height, end_width))
            end_depth, end_height, end_width = 1, 1, 1

        # Initialize flatten layer
        self.flatten = torch.nn.Flatten()

        # Initialize linear layers
        self.linear_layers = torch.nn.ModuleList()
        linear_units = [end_depth * end_height * end_width * filters[-1]] + linear_units
        for i in range(len(linear_units) - 1):
            self.linear_layers.add_module('dropout%s' % i, torch.nn.Dropout(dropout_p[i]))
            self.linear_layers.add_module('linear%s' % i,
                                          torch.nn.Linear(in_features=linear_units[i], out_features=linear_units[i+1],
                                                          bias=use_bias))
            self.linear_layers.add_module('lrelu%s' % i, torch.nn.LeakyReLU(negative_slope=lrelu_alpha))

        # Initialize output layer
        self.out_layer = Output(in_features=linear_units[-1] + self.n_features, out_features=num_classes, bias=use_bias)
        # self.out_layer.__class__.__name__ = 'Output'

    def forward(self, x, features):
        # Blocks
        for block in self.conv_blocks:
            x = block(x)

        # Pooling layers
        if (self.pooling_conv_filters is not None) or self.perform_pooling:
            x = self.pool(x)

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


