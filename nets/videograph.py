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

import tensorflow as tf
import keras.backend as K

from keras.layers import Conv1D, Conv2D, MaxPooling3D, Conv3D, AveragePooling3D
from keras.layers import Dense, LeakyReLU, BatchNormalization, Dropout, Lambda, Activation

from nets.keras_layers import ExpandDimsLayer, SqueezeLayer, TransposeLayer, MeanLayer, DepthwiseConv1DLayer, ReshapeLayer

def graph_embedding(tensor, n_layers, c_avg_size, c_kernel_size, t_kernel_size, c_max_size, t_max_size):
    """
    Graph embedding.
    :param tensor:
    :param n_layers:
    :return:
    """

    input_shape = K.int_shape(tensor)
    _, n_centroids, n_timesteps, side_dim, side_dim, n_channels_in = input_shape

    # hide temporal dimension
    tensor = TransposeLayer((0, 2, 1, 3, 4, 5))(tensor)  # (None, 64, 100, 7, 7, 1024)
    tensor = ReshapeLayer((n_centroids, side_dim, side_dim, n_channels_in))(tensor)

    # pool over centroids
    tensor = AveragePooling3D(pool_size=(c_avg_size, 1, 1), name='pool_n')(tensor)
    _, n_centroids, side_dim, side_dim, n_channels_in = K.int_shape(tensor)

    # recover centroids dimension
    tensor = ReshapeLayer((n_timesteps, n_centroids, side_dim, side_dim, n_channels_in))(tensor)  # (None, 64, 100, 7, 7, 1024)
    tensor = TransposeLayer((0, 2, 1, 3, 4, 5))(tensor)  # (None, 100, 64, 7, 7, 1024)

    # hide the centroids dimension
    tensor = ReshapeLayer((n_timesteps, side_dim, side_dim, n_channels_in))(tensor)  # (None, 64, 7, 7, 1024)

    # 2 layers spatio-temporal conv
    for i in range(n_layers):
        layer_id = '%d' % (i + 1)

        # spatial conv
        tensor = Conv3D(n_channels_in, (1, 1, 1), padding='SAME', name='conv_s_%s' % (layer_id))(tensor)  # (None, 64, 7, 7, 1024)

        # temporal conv
        tensor = DepthwiseConv1DLayer(t_kernel_size, padding='SAME', name='conv_t_%s' % (layer_id))(tensor)  # (None, 64, 7, 7, 1024)

        # centroid conv
        tensor = __convolve_nodes(tensor, n_centroids, layer_id, c_kernel_size)  # (None, 100, 7, 7, 1024)

        # activation
        tensor = BatchNormalization()(tensor)
        tensor = LeakyReLU(alpha=0.2)(tensor)

        # max_pool over centroids
        tensor = MaxPooling3D(pool_size=(c_max_size, 1, 1), name='pool_n_%s' % (layer_id))(tensor)  # (None, 100, 7, 7, 1024)
        _, n_centroids, side_dim, side_dim, n_channels_in = K.int_shape(tensor)

        # get back temporal dimension and hide centroid dimension
        tensor = ReshapeLayer((n_timesteps, n_centroids, side_dim, side_dim, n_channels_in))(tensor)  # (None, 64, 100, 7, 7, 1024)
        tensor = TransposeLayer((0, 2, 1, 3, 4, 5))(tensor)  # (None, 100, 64, 7, 7, 1024)
        tensor = ReshapeLayer((n_timesteps, side_dim, side_dim, n_channels_in))(tensor)  # (None, 64, 7, 7, 1024)

        # max_pool over time
        tensor = MaxPooling3D(pool_size=(t_max_size, 1, 1), name='pool_t_%s' % (layer_id))(tensor)  # (None, 64, 7, 7, 1024)
        _, n_timesteps, side_dim, side_dim, n_channels_in = K.int_shape(tensor)  # (None, 64, 7, 7, 1024)

    # recover centroids dimension
    tensor = ReshapeLayer((n_centroids, n_timesteps, side_dim, side_dim, n_channels_in))(tensor)

    return tensor

def node_attention(x, c, n_channels_in, activation_type='softmax'):
    activation_types = ['relu', 'softmax', 'sigmoid']
    assert activation_type in activation_types, 'Sorry, unknown activation type: %s' % (activation_type)

    # expand for multiplication
    c = ExpandDimsLayer(axis=0)(c)

    # phi path (Q) or (x)
    x = BatchNormalization()(x)
    phi = x  # (None, 64, 1024)

    # theta path (K) or (c)
    theta = BatchNormalization()(c)  # (1, 100, 1024)
    theta = Conv1D(n_channels_in, 1, padding='same', name='node_embedding')(theta)  # (1, 100, 1024)

    # f path (theta and phi) or (Q and K)
    f = Lambda(__tensor_product)([phi, theta])  # (None, 7, 7, 100, 64)
    f = TransposeLayer((0, 1, 2, 4, 3))(f)  # (None, 7, 7, 64, 100)
    f = BatchNormalization()(f)
    if activation_type == 'relu':
        f = LeakyReLU(alpha=0.2, name='node_attention')(f)
        f = BatchNormalization()(f)
    elif activation_type == 'softmax':
        f = Activation('softmax', name='node_attention')(f)
    elif activation_type == 'sigmoid':
        f = Activation('sigmoid', name='node_attention')(f)
    else:
        raise Exception('sorry, unknown activation type')
    f = TransposeLayer((0, 1, 2, 4, 3))(f)  # (None, 7, 7, 100, 64)

    # g path (V)
    g = BatchNormalization()(c)

    y = Lambda(__tensor_multiplication, name='node_attenative')([f, g])  # (N, 100, 64, 7, 7, 1024)
    y = BatchNormalization()(y)
    y = LeakyReLU(alpha=0.2)(y)
    return y

def __convolve_nodes(tensor, n_centroids, layer_id, kernel_size):
    """
    Input size (None, 100, 7, 7, 1024)
    """

    # unreveal centroids dimension
    _, n_timesteps, side_dim, side_dim, n_channels_in = K.int_shape(tensor)  # (None, 64, 7, 7, 1024)
    tensor = ReshapeLayer((n_centroids, n_timesteps, side_dim, side_dim, n_channels_in))(tensor)  # (None, 100, 64, 7, 7, 1024)

    # hide temporal dimension
    tensor = TransposeLayer((0, 2, 1, 3, 4, 5))(tensor)  # (None, 64, 100, 7, 7, 1024)
    tensor = ReshapeLayer((n_centroids, side_dim, side_dim, n_channels_in))(tensor)  # (None, 100, 7, 7, 1024)

    # centroid conv
    tensor = DepthwiseConv1DLayer(kernel_size, padding='SAME', name='conv_n_%s' % (layer_id))(tensor)  # (None, 100, 7, 7, 1024)

    return tensor

def __tensor_product(inputs):
    # inputs # (x, c) or (phi, theta) or (q, v)
    x = inputs[0]  # (None, 64, 7, 7, 1024)
    c = inputs[1]  # (1, 100, 1024)

    result = tf.tensordot(x, c, axes=[4, 2])
    result = tf.squeeze(result, axis=4)  # (None, 64, 7, 7, 100)
    result = tf.transpose(result, (0, 2, 3, 4, 1))  # (None, 7, 7, 100, 64)
    return result

def __tensor_multiplication(inputs):
    f = inputs[0]  # (None, 7, 7, 100, 64)
    g = inputs[1]  # (1, 100, 1024)

    g = tf.expand_dims(g, axis=1)  # (1, 1, 100, 1024)
    g = tf.expand_dims(g, axis=1)  # (1, 1, 1, 100, 1024)
    g = tf.expand_dims(g, axis=4)  # (1, 1, 1, 100, 1, 1024)
    f = tf.expand_dims(f, axis=5)  # (N, 7, 7, 100, 64, 1)
    y = tf.multiply(f, g)  # (N, 7, 7, 100, 64, 1024)
    y = tf.transpose(y, (0, 3, 4, 1, 2, 5))
    return y
