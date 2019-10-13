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

import keras.backend as K
from keras import callbacks
from keras.models import Sequential, model_from_json
from keras.utils import vis_utils
import json
import os
import pydot
import numpy as np
import matplotlib.pyplot as plt
import configs

from core import metrics
from core.utils import Path as Pth

__eps_value = 1e-9

LOSSES = ['categorical_crossentropy', 'mean_squared_error', 'mean_absolute_error', 'binary_crossentropy']
METRICS = ['accuracy', 'mean_squared_error', 'mean_absolute_error']
OPTIMIZERS = ['sgd', 'rmsprop', 'adam']
ACTIVATIONS = ['tanh', 'relu', 'sigmoid', 'softmax']

class LossPlotCallback(callbacks.Callback):
    def __init__(self, n_epochs, title, is_split=True, metric_name='acc'):

        self.__is_local_machine = configs.is_local_machine()

        if not self.__is_local_machine:
            return

        self.metric_name_tr = metric_name
        self.metric_name_te = 'val_%s' % (metric_name)
        self.n_epochs = n_epochs
        self.is_split = is_split
        self.colors = plot_utils.tableau_category20()
        self.title = title

        self.fig, self.ax1 = plt.subplots(num='Keras Loss Plot', figsize=(16, 5))
        self.ax2 = self.ax1.twinx()

        plt.ion()

        self.loss_tr = np.zeros((n_epochs,), dtype=np.float16)
        self.acc_tr = np.zeros((n_epochs,), dtype=np.float16)

        self.loss_te = np.zeros((n_epochs,), dtype=np.float16) if is_split else None
        self.acc_te = np.zeros((n_epochs,), dtype=np.float16) if is_split else None

    def on_epoch_end(self, epoch, logs={}):

        if not self.__is_local_machine:
            return

        epoch += 1

        is_split = self.is_split
        colors = self.colors
        ax1 = self.ax1
        ax2 = self.ax2
        fig = self.fig

        idx = epoch - 1
        self.loss_tr[idx] = logs.get('loss')
        self.acc_tr[idx] = logs.get(self.metric_name_tr)
        if is_split:
            self.loss_te[idx] = logs.get('val_loss')
            self.acc_te[idx] = logs.get(self.metric_name_te)

        loss_tr = self.loss_tr[:epoch]
        acc_tr = self.acc_tr[:epoch]

        if is_split:
            loss_te = self.loss_te[:epoch]
            acc_te = self.acc_te[:epoch]
        else:
            loss_te = None
            acc_te = None

        n_epochs = len(loss_tr)
        x = np.arange(1, n_epochs + 1)

        ax1.cla()
        ax2.cla()

        ax1.set_title('')
        ax2.set_title('')
        plt.title(self.title)

        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss')
        ax2.set_ylabel('accuracy')

        ax1.grid()
        ax2.grid()

        ax1.plot(x, loss_tr, color=colors[0], lw=2, label='loss_tr')
        ax2.plot(x, acc_tr, color=colors[2], lw=2, label='acc_tr')
        if is_split:
            ax1.plot(x, loss_te, color=colors[1], lw=2, label='loss_te')
            ax2.plot(x, acc_te, color=colors[3], lw=2, label='acc_te')

        ax1.legend(loc=2, framealpha=0.5)
        ax2.legend(loc=1, framealpha=0.5)

        plt.pause(0.01)

    def on_train_end(self, logs={}):

        if not self.__is_local_machine:
            return

        plt.tight_layout()
        plt.ioff()

class ModelSaveCallback(callbacks.Callback):
    def __init__(self, model, model_name, epoch_offset, model_root_path):
        self.model = model
        self.model_name = model_name
        self.epoch_offset = epoch_offset

        model_root_path = '%s/%s' % (model_root_path, self.model_name)
        if not os.path.exists(model_root_path):
            os.mkdir(model_root_path)
        self.model_root_path = model_root_path

        super(ModelSaveCallback, self).__init__()

    def on_epoch_end(self, idx_epoch, logs={}):
        # save the model, if required
        epoch_num = idx_epoch + 1 + self.epoch_offset
        model_root_path = self.model_root_path

        model_path = '%s/%03d.model' % (model_root_path, epoch_num)
        model_json_path = '%s/%03d.json' % (model_root_path, epoch_num)
        model_weight_path = '%s/%03d.pkl' % (model_root_path, epoch_num)

        # for very long model, this does not work
        # self.root_model.save(model_path)
        # only save model definition and weights
        self.__save_model(model_json_path, model_weight_path)

    def __save_model(self, json_path, weight_path):
        model = self.model
        model.save_weights(weight_path, overwrite=True)
        model_json = model.to_json()
        with open(json_path, 'w') as f:
            json.dump(model_json, f)

class MapScoreCallback(callbacks.Callback):
    def __init__(self, model, x_tr, y_tr, x_te, y_te, batch_size, n_classes):

        self.model = model
        self.x_tr = x_tr
        self.y_tr = y_tr
        self.x_te = x_te
        self.y_te = y_te
        self.batch_size = batch_size
        self.n_classes = n_classes

    def on_epoch_end(self, epoch, logs={}):

        y_pred_te = self.model.predict(self.x_te)
        if self.x_tr is not None and self.y_tr is not None:
            y_pred_tr = self.model.predict(self.x_tr, self.batch_size)
            a_tr = metrics.mean_avg_precision_sklearn(self.y_tr, y_pred_tr)
            a_te = metrics.mean_avg_precision_sklearn(self.y_te, y_pred_te)
            a_tr *= 100.0
            a_te *= 100.0
            msg = '        map_: %.02f%%, %.02f%%' % (a_tr, a_te)
            print (msg)
        else:
            a_te = metrics.mean_avg_precision_sklearn(self.y_te, y_pred_te)
            a_te *= 100.0
            msg = '        map_: %.02f%%' % (a_te)
            print (msg)

def plot_model(model, file_path='model.eps'):
    vis_utils.plot_model(model, file_path, show_shapes=True, show_layer_names=True)

def load_model(json_path, weight_path, metrics=None, loss=None, optimizer=None, custom_objects=None, is_compile=True):
    with open(json_path, 'r') as f:
        model_json_string = json.load(f)
    model_json_dict = json.loads(model_json_string)
    model = model_from_json(model_json_string, custom_objects=custom_objects)
    model.load_weights(weight_path)

    if is_compile:
        if optimizer is None:
            optimizer = model_json_dict['optimizer']['name']

        if loss is None:
            loss = model_json_dict['loss']

        if metrics is None:
            model.compile(loss=loss, optimizer=optimizer)
        else:
            model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model

def save_model(model, json_path, weight_path):
    model.save_weights(weight_path, overwrite=True)
    model_json = model.to_json()
    with open(json_path, 'w') as f:
        json.dump(model_json, f)

def get_output(input, func, batch_size=500):
    n_values = len(input)
    n_steps = int(n_values / float(batch_size))
    output = None
    if n_values % float(batch_size) != 0:
        n_steps += 1
    for i in range(n_steps):
        start_idx = i * batch_size
        stop_idx = (i + 1) * batch_size
        if stop_idx > n_values:
            stop_idx = n_values
        batch_output = func([input[start_idx:stop_idx], 0])[0]
        if output is None:
            output_shape = list(batch_output.shape)
            output_shape[0] = n_values
            output_shape = tuple(output_shape)
            output = np.zeros(shape=output_shape)
        output[start_idx:stop_idx] = batch_output
    return output

def get_output_for_bi_input(input_1, input_2, func, batch_size=500):
    n_values = len(input_1)
    n_steps = int(n_values / float(batch_size))
    output = None
    if n_values % float(batch_size) != 0:
        n_steps += 1
    for i in range(n_steps):
        start_idx = i * batch_size
        stop_idx = (i + 1) * batch_size
        if stop_idx > n_values:
            stop_idx = n_values
        batch_output = func([input_1[start_idx:stop_idx], input_2[start_idx:stop_idx], 0])[0]
        if output is None:
            output = np.zeros(shape=(n_values, batch_output.shape[1]))
        output[start_idx:stop_idx] = batch_output

    return output

def plot_model_loss_and_score(model_history, title, is_split, metric_name):
    is_local_machine = configs.is_local_machine()
    if not is_local_machine:
        return

    loss_tr = model_history['loss']
    acc_tr = model_history[metric_name]
    if is_split:
        metric_name_val = 'val_%s' % (metric_name)
        loss_te = model_history['val_loss']
        acc_te = model_history[metric_name_val]

    n_epochs = len(loss_tr)
    x = np.arange(1, n_epochs + 1)

    colors = plot_utils.tableau_category20()
    fig, ax1 = plt.subplots(figsize=(16, 5))
    ax2 = ax1.twinx()
    ax1.grid()
    ax2.grid()
    ax1.plot(x, loss_tr, color=colors[0], lw=2, label='loss_tr')
    ax2.plot(x, acc_tr, color=colors[2], lw=2, label='acc_tr')
    if is_split:
        ax1.plot(x, loss_te, color=colors[1], lw=2, label='loss_te')
        ax2.plot(x, acc_te, color=colors[3], lw=2, label='acc_te')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax2.set_ylabel('accuracy')
    ax1.legend(loc=2, framealpha=0.5)
    ax2.legend(loc=1, framealpha=0.5)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def layer_exist(model, layer_name):
    exist = False
    for layer in model.layers:
        if layer.name == layer_name:
            exist = True
            break

    return exist

def calc_num_batches(n_samples, batch_size):
    n_batch = int(n_samples / float(batch_size))
    n_batch = n_batch if n_samples % batch_size == 0 else n_batch + 1
    return n_batch
