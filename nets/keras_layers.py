# !/usr/bin/env python
# -*- coding: UTF-8 -*-

########################################################################
# GNU General Public License v3.0
# GNU GPLv3
# Copyright (c) 2019, Noureldien Hussein
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
########################################################################

import math

from keras import backend as K
from keras.engine.topology import Layer

import tensorflow as tf
import tensorflow.contrib.layers as contrib_layers

class SliceLayer(Layer):
    def __init__(self, name, **kwargs):
        self.name = name
        self.index = -1
        super(SliceLayer, self).__init__(**kwargs)

    def set_index(self, index):
        self.index = index

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], input_shape[2])
        return output_shape

    def call(self, input, mask=None):
        value = input[:, self.index, :]
        return value

class ReshapeLayer(Layer):
    def __init__(self, new_shape, **kwargs):
        self.new_shape = new_shape
        super(ReshapeLayer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        output_shape = [input_shape[0]] + list(self.new_shape)
        output_shape = tuple(output_shape)
        return output_shape

    def call(self, input, mask=None):
        output_shape = [-1] + list(self.new_shape)
        output_shape = tuple(output_shape)
        value = tf.reshape(input, output_shape)
        return value

    def get_config(self):
        """
        For rebuilding models on load time.
        """
        config = {'new_shape': self.new_shape}
        base_config = super(ReshapeLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class TransposeLayer(Layer):
    def __init__(self, new_perm, **kwargs):
        self.new_perm = new_perm
        super(TransposeLayer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == len(self.new_perm)

        output_shape = [input_shape[self.new_perm[idx]] for idx in range(len(input_shape))]
        output_shape = tuple(output_shape)
        return output_shape

    def call(self, input, mask=None):
        value = tf.transpose(input, self.new_perm)
        return value

    def get_config(self):
        """
        For rebuilding models on load time.
        """
        config = {'new_perm': self.new_perm}
        base_config = super(TransposeLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ExpandDimsLayer(Layer):
    def __init__(self, axis, **kwargs):
        self.axis = axis
        super(ExpandDimsLayer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)

        axis_type = type(self.axis)
        axes = self.axis if (axis_type is list or axis_type is tuple) else [self.axis]

        for axis in axes:
            output_shape.insert(axis, 1)

        output_shape = tuple(output_shape)
        return output_shape

    def call(self, input, mask=None):

        axis_type = type(self.axis)
        axes = self.axis if (axis_type is list or axis_type is tuple) else [self.axis]
        value = input

        for axis in axes:
            value = tf.expand_dims(value, axis)

        return value

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(ExpandDimsLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SqueezeLayer(Layer):
    def __init__(self, axis, **kwargs):
        self.axis = axis
        super(SqueezeLayer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        n_dims = len(input_shape)
        axis_type = type(self.axis)
        axes = self.axis if (axis_type is list or axis_type is tuple) else [self.axis]
        output_shape = []
        for i in range(n_dims):
            if i not in axes:
                output_shape.append(input_shape[i])
        output_shape = tuple(output_shape)
        return output_shape

    def call(self, input, mask=None):
        value = tf.squeeze(input, self.axis)
        return value

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(SqueezeLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class MaxLayer(Layer):
    def __init__(self, axis, is_keep_dim=False, **kwargs):
        self.axis = axis
        self.is_keep_dim = is_keep_dim
        super(MaxLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MaxLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        n_dims = len(input_shape)
        axis_type = type(self.axis)
        axes = self.axis if (axis_type is list or axis_type is tuple) else [self.axis]
        output_shape = []
        for i in range(n_dims):
            if self.is_keep_dim:
                if i in axes:
                    output_shape.append(1)
                else:
                    output_shape.append(input_shape[i])
            else:
                if i not in axes:
                    output_shape.append(input_shape[i])

        output_shape = tuple(output_shape)
        return output_shape

    def call(self, input, mask=None):
        tensor = tf.reduce_max(input, axis=self.axis, keepdims=self.is_keep_dim)
        return tensor

    def get_config(self):
        config = {'axis': self.axis, 'is_keep_dim': self.is_keep_dim}
        base_config = super(MaxLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SumLayer(Layer):
    def __init__(self, axis, is_keep_dim=False, **kwargs):
        self.axis = axis
        self.is_keep_dim = is_keep_dim
        super(SumLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SumLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        n_dims = len(input_shape)
        axis_type = type(self.axis)
        axes = self.axis if (axis_type is list or axis_type is tuple) else [self.axis]
        output_shape = []
        for i in range(n_dims):
            if self.is_keep_dim:
                if i in axes:
                    output_shape.append(1)
                else:
                    output_shape.append(input_shape[i])
            else:
                if i not in axes:
                    output_shape.append(input_shape[i])

        output_shape = tuple(output_shape)
        return output_shape

    def call(self, input, mask=None):
        tensor = tf.reduce_sum(input, axis=self.axis, keepdims=self.is_keep_dim)
        return tensor

    def get_config(self):
        config = {'axis': self.axis, 'is_keep_dim': self.is_keep_dim}
        base_config = super(SumLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class MeanLayer(Layer):
    def __init__(self, axis, is_keep_dim=False, **kwargs):
        self.axis = axis
        self.is_keep_dim = is_keep_dim
        super(MeanLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MeanLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        n_dims = len(input_shape)
        axis_type = type(self.axis)
        axes = self.axis if (axis_type is list or axis_type is tuple) else [self.axis]
        output_shape = []
        for i in range(n_dims):
            if self.is_keep_dim:
                if i in axes:
                    output_shape.append(1)
                else:
                    output_shape.append(input_shape[i])
            else:
                if i not in axes:
                    output_shape.append(input_shape[i])

        output_shape = tuple(output_shape)
        return output_shape

    def call(self, input, mask=None):
        tensor = tf.reduce_mean(input, axis=self.axis, keepdims=self.is_keep_dim)
        return tensor

    def get_config(self):
        config = {'axis': self.axis, 'is_keep_dim': self.is_keep_dim}
        base_config = super(MeanLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class DepthwiseConv1DLayer(Layer):
    """
    Expects a tensor of 5D (Batch_Size, Temporal_Dimension, Width, Length, Channel_Dimension)
    Applies a local 1*1*k Conv1D on each separate channel of the input, and along the temporal dimension
    Returns a 5D tensor.
    """

    def __init__(self, kernel_size, padding, **kwargs):
        self.kernel_size = kernel_size
        self.padding = padding
        super(DepthwiseConv1DLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Input shape is (None, 20, 7, 7, 1024)
        :param input_shape:
        :return:
        """

        assert len(input_shape) == 5

        initializer = contrib_layers.xavier_initializer()

        _, n_timesteps, feat_map_side_dim1, feat_map_side_dim2, n_spatial_maps = input_shape
        self.n_timesteps = n_timesteps
        self.n_maps = n_spatial_maps
        self.side_dim1 = feat_map_side_dim1
        self.side_dim2 = feat_map_side_dim2

        weights_name = 'depthwise_conv_1d_weights'
        biases_name = 'depthwise_conv1d_biases'

        # 1x1 convolution kernel
        weights_shape = [self.kernel_size, 1, n_spatial_maps, 1]
        bias_shape = [n_spatial_maps, ]

        with tf.variable_scope(self.name) as scope:
            self.conv_weights = tf.get_variable(weights_name, shape=weights_shape, initializer=initializer)
            self.conv_biases = tf.get_variable(biases_name, shape=bias_shape, initializer=tf.constant_initializer(0.1))

        self.trainable_weights = [self.conv_weights, self.conv_biases]

        super(DepthwiseConv1DLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        output_dim = (None, self.n_timesteps, self.side_dim1, self.side_dim2, self.n_maps)
        return output_dim

    def call(self, input, mask=None):
        # inputs is of shape (None, 20, 7, 7, 1024)

        # transpose and reshape to hide the spatial dimension, only expose the temporal dimension for depthwise conv
        tensor = tf.transpose(input, (0, 2, 3, 1, 4))  # (None, 7, 7, 20, 1, 1024)
        tensor = tf.reshape(tensor, (-1, self.n_timesteps, 1, self.n_maps))  # (None*7*7, 20, 1, 1024)

        # depthwise conv on the temporal dimension, as if it was the spatial dimension
        tensor = tf.nn.depthwise_conv2d(tensor, self.conv_weights, strides=(1, 1, 1, 1), padding='SAME', data_format='NHWC')  # (None*7*7, 20, 1, 1024)
        tensor = tf.nn.bias_add(tensor, self.conv_biases)  # (None*7*7, 20, 1, 1024)

        # reshape to get the spatial dimensions
        tensor = tf.reshape(tensor, (-1, self.side_dim1, self.side_dim2, self.n_timesteps, self.n_maps))  # (None, 7, 7, 20, 1024)

        # finally, transpose to get the desired output shape
        tensor = tf.transpose(tensor, (0, 3, 1, 2, 4))  # (None, 20, 7, 7, 1024)

        return tensor

    def get_config(self):
        """
        For rebuilding models on load time.
        """
        config = {'kernel_size': self.kernel_size, 'padding': self.padding}
        base_config = super(DepthwiseConv1DLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
