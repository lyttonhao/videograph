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

"""
This experiment is for EPIC-Kitchens dataset
"""

import os
import time
import datetime

import numpy as np

import tensorflow as tf
from keras.layers import Input, BatchNormalization
from keras.layers import Dense, LeakyReLU, Dropout, Activation, Conv3D
from keras.optimizers import SGD, Adam
from keras.models import Model

from nets.keras_layers import MaxLayer, MeanLayer

from datasets import epic_kitchens
from nets import videograph
from core import utils, keras_utils
from core.utils import Path as Pth

def train_model_videograph():
    """
    Train model.
    """

    annotation_type = 'noun'
    annot_path = Pth('EPIC-Kitchens/annotations/annot_video_level_many_shots.pkl')
    (y_tr, y_te), n_classes = __load_annotation(annot_path, annotation_type)

    model_type = 'i3d_rgb'
    feature_type = 'mixed_5c'
    n_nodes = 128
    n_timesteps = 64
    n_frames_per_segment = 8
    n_frames_per_video = n_timesteps * n_frames_per_segment
    batch_size_tr = 20
    batch_size_te = 30
    n_epochs = 500
    epoch_offset = 0
    model_name = 'classifier_%s' % (utils.timestamp())
    model_root_path = Pth('EPIC-Kitchens/models')

    nodes_path = Pth('EPIC-Kitchens/features/nodes_random_%d.pkl', (n_nodes,))
    features_path = Pth('EPIC-Kitchens/features/features_i3d_mixed_5c_%d_frames.h5', (n_frames_per_video,))
    n_channels, side_dim = utils.get_model_feat_maps_info(model_type, feature_type)
    input_shape = (None, n_timesteps, side_dim, side_dim, n_channels)

    # either load nodes, or generate them on the fly, but remeber to save them, as you need them in test time
    # nodes = utils.pkl_load(nodes_path)
    nodes = utils.generate_centroids(n_nodes, n_channels)

    print ('--- start time')
    print (datetime.datetime.now())

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()
    model = __load_model_videograph(nodes, n_classes, input_shape)
    t2 = time.time()
    duration = t2 - t1
    print (model.summary(line_length=130, positions=None, print_fn=None))
    print ('... model built, duration (sec): %d' % (duration))

    # load data
    print ('... loading data: %s' % (features_path))
    t1 = time.time()
    (x_tr, x_te) = utils.h5_load_multi(features_path, ['x_tr', 'x_te'])
    t2 = time.time()

    duration = t2 - t1
    print ('... data loaded: %d' % (duration))

    n_tr = len(x_tr)
    n_te = len(x_te)
    n_batch_tr = utils.calc_num_batches(n_tr, batch_size_tr)
    n_batch_te = utils.calc_num_batches(n_te, batch_size_te)
    print ('... [tr]: n, n_batch, batch_size: %d, %d, %d' % (n_tr, n_batch_tr, batch_size_tr))
    print ('... [te]: n, n_batch, batch_size: %d, %d, %d' % (n_te, n_batch_te, batch_size_te))
    print (x_tr.shape)
    print (x_te.shape)
    print (y_tr.shape)
    print (y_te.shape)

    save_callback = keras_utils.ModelSaveCallback(model, model_name, epoch_offset, model_root_path)
    score_callback = keras_utils.MapScoreCallback(model, None, None, x_te, y_te, batch_size_te, n_classes)
    model_callbacks = [save_callback, score_callback]
    model.fit(x_tr, y_tr, epochs=n_epochs, batch_size=batch_size_tr, validation_split=0.0, validation_data=(x_te, y_te), shuffle=True, callbacks=model_callbacks, verbose=2)

    print ('--- finish time')
    print (datetime.datetime.now())

def __load_model_videograph(nodes, n_classes, input_shape_x):
    """
    Model
    """

    # optimizer and loss
    loss = keras_utils.LOSSES[3]
    output_activation = keras_utils.ACTIVATIONS[2]
    optimizer = SGD(lr=0.01)
    optimizer = Adam(lr=0.01, epsilon=1e-8)
    optimizer = Adam(lr=0.01, epsilon=1e-4)

    # per-layer kernel size and max pooling for nodes and timesteps
    n_graph_layers = 2

    # time kernel
    t_kernel_size = 7
    t_max_size = 3

    # node kernel
    n_kernel_size = 7
    n_max_size = 3
    n_avg_size = 4

    # space kernel
    s_kernel_size = 2
    s_kernel_size = 1

    n_nodes, _ = nodes.shape

    _, n_timesteps, side_dim, side_dim, n_channels_in = input_shape_x
    t_input_x = Input(shape=(n_timesteps, side_dim, side_dim, n_channels_in), name='input_x')  # (None, 64, 1024)
    t_input_n = Input(tensor=tf.constant(nodes, dtype=tf.float32), name='input_n')  # (1, 100, 1024)
    tensor = t_input_x

    # spatial convolution
    tensor = Conv3D(n_channels_in, (1, s_kernel_size, s_kernel_size), padding='VALID', name='conv_s')(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = LeakyReLU(alpha=0.2)(tensor)

    # pool over space
    tensor = MaxLayer(axis=(2, 3), is_keep_dim=True, name='global_pool_s')(tensor)  # (None, 64, 7, 7, 1024)

    # node attention
    tensor = videograph.node_attention(tensor, t_input_n, n_channels_in, activation_type='relu')  # (N, 100, 64, 7, 7, 1024)

    # graph embedding
    tensor = videograph.graph_embedding(tensor, n_graph_layers, n_avg_size, n_kernel_size, t_kernel_size, n_max_size, t_max_size)  # (N, 100, 64, 7, 7, 1024)

    # node pooling
    tensor = MeanLayer(axis=(1,), name='global_pool_n')(tensor)

    # temporal pooling
    tensor = MaxLayer(axis=(1, 2, 3), name='global_pool_t')(tensor)

    # mlp for classification
    tensor = Dropout(0.25)(tensor)
    tensor = Dense(512)(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = LeakyReLU(alpha=0.2)(tensor)
    tensor = Dropout(0.25)(tensor)
    tensor = Dense(n_classes)(tensor)
    t_output = Activation(output_activation)(tensor)

    model = Model(input=[t_input_x, t_input_n], output=t_output)
    model.compile(loss=loss, optimizer=optimizer)
    return model

def __load_annotation(annotation_path, annotation_type):
    annotation_types = ['noun', 'verb', 'noun_verb', 'action']
    assert annotation_type in annotation_types

    (y_noun_tr, y_verb_tr, y_actn_tr, y_noun_te, y_verb_te, y_actn_te) = utils.pkl_load(annotation_path)
    if annotation_type == 'noun':
        n_classes = epic_kitchens.N_NOUNS_MANY_SHOT
        (y_tr, y_te) = (y_noun_tr, y_noun_te)
    elif annotation_type == 'verb':
        n_classes = epic_kitchens.N_VERBS_MANY_SHOT
        (y_tr, y_te) = (y_verb_tr, y_verb_te)
    elif annotation_type == 'noun_verb':
        n_classes = epic_kitchens.N_NOUNS_MANY_SHOT + epic_kitchens.N_VERBS_MANY_SHOT
        (y_tr, y_te) = (np.hstack((y_noun_tr, y_verb_tr)), np.hstack((y_noun_te, y_verb_te)))
    elif annotation_type == 'action':
        n_classes = epic_kitchens.N_ACTNS_MANY_SHOT
        (y_tr, y_te) = (y_actn_tr, y_actn_te)
    else:
        raise Exception('Sorry, unknown annotation type: %s' % (annotation_type))

    return (y_tr, y_te), n_classes
