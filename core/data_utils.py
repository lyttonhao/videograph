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

import numpy as np
import pickle as pkl
import h5py
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn import preprocessing, manifold

import os
import json
import natsort
import random
from multiprocessing.dummy import Pool
from core import utils

class AsyncVideoFeaturesLoader():
    """
    Load features for the video frames.
    """

    def __init__(self, feats_path, class_nums, n_classes, n_frames_per_video, batch_size, n_feat_maps, feat_map_side_dim, n_threads=10):
        random.seed(101)
        np.random.seed(101)

        self.__feats_pathes = feats_path
        self.__class_nums = class_nums
        self.__n_frames_per_video = n_frames_per_video
        self.__n_feat_maps = n_feat_maps
        self.__feat_map_side_dim = feat_map_side_dim

        self.__batch_size = batch_size

        # binarize the labels
        classes = range(1, n_classes + 1)
        self.__y = label_binarize(self.__class_nums, classes)

        self.__is_busy = False
        self.__batch_features = None
        self.__batch_y = None
        self.__n_threads_in_pool = n_threads
        self.__pool = Pool(self.__n_threads_in_pool)

    def load_feats_in_batch(self, batch_number):
        self.__is_busy = True

        idx_batch = batch_number - 1
        start_idx = idx_batch * self.__batch_size
        stop_idx = (idx_batch + 1) * self.__batch_size

        batch_feat_pathes = self.__feats_pathes[start_idx:stop_idx]
        batch_y = self.__y[start_idx:stop_idx]

        n_batch_feats = len(batch_feat_pathes)
        n_batch_y = len(batch_y)
        idxces = range(0, n_batch_feats)

        assert n_batch_feats == n_batch_y

        # parameters passed to the reading function
        params = [data_item for data_item in zip(idxces, batch_feat_pathes)]

        # set list of batch features before start reading
        batch_feats_shape = (n_batch_feats, self.__n_frames_per_video, self.__feat_map_side_dim, self.__feat_map_side_dim, self.__n_feat_maps)

        self.__batch_features = np.zeros(batch_feats_shape, dtype=np.float32)
        self.__batch_y = batch_y

        # start pool of threads
        self.__pool.map_async(self.__load_features, params, callback=self.__thread_pool_callback)

    def get_batch_data(self):
        if self.__is_busy:
            raise Exception('Sorry, you can\'t get features while threads are running!')
        else:
            return (self.__batch_features, self.__batch_y)

    def get_y(self):
        return self.__y

    def is_busy(self):
        return self.__is_busy

    def __thread_pool_callback(self, args):
        self.__is_busy = False

    def __load_features(self, params):

        idx_video = params[0]
        feats_path = params[1]

        try:
            # load feature from file
            feats = utils.pkl_load(feats_path)
            n_feats = len(feats)

            if self.__feat_map_side_dim == 1:
                feats = np.expand_dims(feats, 1)
                feats = np.expand_dims(feats, 1)

            # some videos have frames less than required, then take all features
            # and zero-pad (may be repeat the frames?) till the required is satisfied
            if n_feats < self.__n_frames_per_video:
                feats = self.__pad_video_feats(feats)
            else:
                # randomly sample only n frames
                idx_feats = np.arange(n_feats)
                np.random.shuffle(idx_feats)
                idx_feats = idx_feats[:self.__n_frames_per_video]
                feats = feats[idx_feats]

            assert len(feats) == self.__n_frames_per_video

            # as float
            feats = feats.astype(np.float32)
            self.__batch_features[idx_video] = feats
            _ = 10
        except Exception as exp:
            print ('\nSorry, error in loading feature %s' % (feats_path))
            print (exp)

    def __pad_video_feats(self, feats):

        n_padded = self.__n_frames_per_video
        n = len(feats)

        padded_feats_shape = list(feats.shape)
        padded_feats_shape[0] = n_padded
        padded_feats_shape = tuple(padded_feats_shape)
        padded_feats = np.zeros(padded_feats_shape, dtype=feats.dtype)

        n_chunks = int(n_padded / float(n)) + 1

        for idx_chunk in range(n_chunks):
            start_idx = idx_chunk * n
            stop_idx = (idx_chunk + 1) * n

            if idx_chunk + 1 == n_chunks:
                remainder = int(n_padded % float(n))
                stop_idx = start_idx + remainder
                padded_feats[start_idx: stop_idx] = feats[:remainder]
            else:
                padded_feats[start_idx: stop_idx] = feats

        return padded_feats

    def shuffle_data(self):
        """
        shuffle these data: self.__feats_pathes, self.__class_names, self.__y
        :return:
        """

        n_samples = len(self.__feats_pathes)

        idx = range(n_samples)
        np.random.shuffle(idx)
        self.__feats_pathes = self.__feats_pathes[idx]
        self.__class_nums = self.__class_nums[idx]
        self.__y = self.__y[idx]

    def close(self):
        self.__pool.close()
        self.__pool.terminate()

class AsyncVideoFeaturesLoaderCharades():
    """
    Load features for the video frames.
    """

    def __init__(self, feats_path, target, n_frames_per_video, batch_size, n_feat_maps, feat_map_side_dim, n_threads=10, annotation_dict=None):
        random.seed(101)
        np.random.seed(101)

        self.__feats_pathes = feats_path
        self.__n_frames_per_video = n_frames_per_video
        self.__n_feat_maps = n_feat_maps
        self.__feat_map_side_dim = feat_map_side_dim
        self.__annotation_dict = annotation_dict

        self.__batch_size = batch_size
        self.__y = target

        self.__is_busy = False
        self.__batch_features = None
        self.__batch_y = None
        self.__n_threads_in_pool = n_threads
        self.__pool = Pool(self.__n_threads_in_pool)

    def load_feats_in_batch(self, batch_number):
        self.__is_busy = True

        idx_batch = batch_number - 1
        start_idx = idx_batch * self.__batch_size
        stop_idx = (idx_batch + 1) * self.__batch_size

        batch_feat_pathes = self.__feats_pathes[start_idx:stop_idx]
        batch_y = self.__y[start_idx:stop_idx]

        n_batch_feats = len(batch_feat_pathes)
        n_batch_y = len(batch_y)
        idxces = range(0, n_batch_feats)

        assert n_batch_feats == n_batch_y

        # parameters passed to the reading function
        params = [data_item for data_item in zip(idxces, batch_feat_pathes)]

        # set list of batch features before start reading
        batch_feats_shape = (n_batch_feats, self.__n_frames_per_video, self.__feat_map_side_dim, self.__feat_map_side_dim, self.__n_feat_maps)

        self.__batch_features = np.zeros(batch_feats_shape, dtype=np.float32)
        self.__batch_y = batch_y

        # start pool of threads
        self.__pool.map_async(self.__load_features, params, callback=self.__thread_pool_callback)

    def get_batch_data(self):
        if self.__is_busy:
            raise Exception('Sorry, you can\'t get features while threads are running!')
        else:
            return (self.__batch_features, self.__batch_y)

    def get_y(self):
        return self.__y

    def is_busy(self):
        return self.__is_busy

    def __thread_pool_callback(self, args):
        self.__is_busy = False

    def __load_features(self, params):

        idx_video = params[0]
        feats_path = params[1]
        video_name = feats_path.split('/')[-1]

        try:
            # load feature from file
            feats = utils.pkl_load(feats_path)

            n_feats = len(feats)
            assert n_feats == self.__n_frames_per_video, 'Sorry, wrong number of frames, expected: %d, got: %d' % (self.__n_frames_per_video, n_feats)
            self.__batch_features[idx_video] = feats

        except Exception as exp:
            print ('\nSorry, error in loading feature %s' % (feats_path))
            print (exp)

    def __pad_video_feats(self, feats):

        n_padded = self.__n_frames_per_video
        n = len(feats)

        padded_feats_shape = list(feats.shape)
        padded_feats_shape[0] = n_padded
        padded_feats_shape = tuple(padded_feats_shape)
        padded_feats = np.zeros(padded_feats_shape, dtype=feats.dtype)

        n_chunks = int(n_padded / float(n)) + 1

        for idx_chunk in range(n_chunks):
            start_idx = idx_chunk * n
            stop_idx = (idx_chunk + 1) * n
            if idx_chunk + 1 == n_chunks:
                remainder = int(n_padded % float(n))
                stop_idx = start_idx + remainder
                padded_feats[start_idx: stop_idx] = feats[:remainder]
            else:
                padded_feats[start_idx: stop_idx] = feats

        return padded_feats

    def __order_feats(self, feats_path, feats):

        video_name = feats_path.split('/')[-1]
        video_name = utils.remove_extension(video_name)

        frames_names = self.__annotation_dict[video_name]
        idx = natsort.index_natsorted(frames_names)
        feats = feats[idx]

        return feats

    def shuffle_data(self):
        """
        shuffle these data: self.__feats_pathes, self.__class_names, self.__y
        :return:
        """

        n_samples = len(self.__feats_pathes)

        idx = range(n_samples)
        np.random.shuffle(idx)
        self.__feats_pathes = self.__feats_pathes[idx]
        self.__y = self.__y[idx]

    def close(self):
        self.__pool.close()
        self.__pool.terminate()

class AsyncVideoFeaturesLoaderMultiTHUMOS():
    """
    Load features for the video frames.
    """

    def __init__(self, feats_path, target, n_frames_per_video, batch_size, n_feat_maps, feat_map_side_dim, n_threads=10, annotation_dict=None):
        random.seed(101)
        np.random.seed(101)

        self.__feats_pathes = feats_path
        self.__n_frames_per_video = n_frames_per_video
        self.__n_feat_maps = n_feat_maps
        self.__feat_map_side_dim = feat_map_side_dim
        self.__annotation_dict = annotation_dict

        self.__batch_size = batch_size
        self.__y = target

        self.__is_busy = False
        self.__batch_features = None
        self.__batch_y = None
        self.__n_threads_in_pool = n_threads
        self.__pool = Pool(self.__n_threads_in_pool)

    def load_feats_in_batch(self, batch_number):
        self.__is_busy = True

        idx_batch = batch_number - 1
        start_idx = idx_batch * self.__batch_size
        stop_idx = (idx_batch + 1) * self.__batch_size

        batch_feat_pathes = self.__feats_pathes[start_idx:stop_idx]
        batch_y = self.__y[start_idx:stop_idx]

        n_batch_feats = len(batch_feat_pathes)
        n_batch_y = len(batch_y)
        idxces = range(0, n_batch_feats)

        assert n_batch_feats == n_batch_y

        # parameters passed to the reading function
        params = [data_item for data_item in zip(idxces, batch_feat_pathes)]

        # set list of batch features before start reading
        batch_feats_shape = (n_batch_feats, self.__n_frames_per_video, self.__feat_map_side_dim, self.__feat_map_side_dim, self.__n_feat_maps)

        self.__batch_features = np.zeros(batch_feats_shape, dtype=np.float32)
        self.__batch_y = batch_y

        # start pool of threads
        self.__pool.map_async(self.__load_features, params, callback=self.__thread_pool_callback)

    def get_batch_data(self):
        if self.__is_busy:
            raise Exception('Sorry, you can\'t get features while threads are running!')
        else:
            return (self.__batch_features, self.__batch_y)

    def get_y(self):
        return self.__y

    def is_busy(self):
        return self.__is_busy

    def __thread_pool_callback(self, args):
        self.__is_busy = False

    def __load_features(self, params):

        idx_video = params[0]
        feats_path = params[1]

        try:
            # load feature from file
            feats = utils.pkl_load(feats_path)

            n_feats = len(feats)
            assert n_feats == self.__n_frames_per_video, 'Sorry, wrong number of frames, expected: %d, got: %d' % (self.__n_frames_per_video, n_feats)
            self.__batch_features[idx_video] = feats

        except Exception as exp:
            print ('\nSorry, error in loading feature %s' % (feats_path))
            print (exp)

    def shuffle_data(self):
        """
        shuffle these data: self.__feats_pathes, self.__class_names, self.__y
        :return:
        """

        n_samples = len(self.__feats_pathes)

        idx = range(n_samples)
        np.random.shuffle(idx)
        self.__feats_pathes = self.__feats_pathes[idx]
        self.__y = self.__y[idx]

    def close(self):
        self.__pool.close()
        self.__pool.terminate()

class AsyncVideoFeaturesLoaderBreakfast():
    """
    Load features for the video frames.
    """

    def __init__(self, feats_path, target, n_frames_per_video, batch_size, n_feat_maps, feat_map_side_dim, n_threads=10):
        random.seed(101)
        np.random.seed(101)

        self.__feats_pathes = feats_path
        self.__n_frames_per_video = n_frames_per_video
        self.__n_feat_maps = n_feat_maps
        self.__feat_map_side_dim = feat_map_side_dim

        self.__batch_size = batch_size
        self.__y = target

        self.__is_busy = False
        self.__batch_features = None
        self.__batch_y = None
        self.__n_threads_in_pool = n_threads
        self.__pool = Pool(self.__n_threads_in_pool)

    def load_feats_in_batch(self, batch_number):
        self.__is_busy = True

        idx_batch = batch_number - 1
        start_idx = idx_batch * self.__batch_size
        stop_idx = (idx_batch + 1) * self.__batch_size

        batch_feat_pathes = self.__feats_pathes[start_idx:stop_idx]
        batch_y = self.__y[start_idx:stop_idx]

        n_batch_feats = len(batch_feat_pathes)
        n_batch_y = len(batch_y)
        idxces = range(0, n_batch_feats)

        assert n_batch_feats == n_batch_y

        # parameters passed to the reading function
        params = [data_item for data_item in zip(idxces, batch_feat_pathes)]

        # set list of batch features before start reading
        batch_feats_shape = (n_batch_feats, self.__n_frames_per_video, self.__feat_map_side_dim, self.__feat_map_side_dim, self.__n_feat_maps)

        self.__batch_features = np.zeros(batch_feats_shape, dtype=np.float32)
        self.__batch_y = batch_y

        # start pool of threads
        self.__pool.map_async(self.__load_features, params, callback=self.__thread_pool_callback)

    def get_batch_data(self):
        if self.__is_busy:
            raise Exception('Sorry, you can\'t get features while threads are running!')
        else:
            return (self.__batch_features, self.__batch_y)

    def get_y(self):
        return self.__y

    def is_busy(self):
        return self.__is_busy

    def __thread_pool_callback(self, args):
        self.__is_busy = False

    def __load_features(self, params):

        idx_video = params[0]
        feats_path = params[1]
        video_name = feats_path.split('/')[-1]

        try:
            # load feature from file
            feats = utils.pkl_load(feats_path)

            n_feats = len(feats)
            assert n_feats == self.__n_frames_per_video, 'Sorry, wrong number of frames, expected: %d, got: %d' % (self.__n_frames_per_video, n_feats)
            self.__batch_features[idx_video] = feats

        except Exception as exp:
            print ('\nSorry, error in loading feature %s' % (feats_path))
            print (exp)

    def shuffle_data(self):
        """
        shuffle these data: self.__feats_pathes, self.__class_names, self.__y
        :return:
        """

        n_samples = len(self.__feats_pathes)

        idx = range(n_samples)
        np.random.shuffle(idx)
        self.__feats_pathes = self.__feats_pathes[idx]
        self.__y = self.__y[idx]

    def close(self):
        self.__pool.close()
        self.__pool.terminate()


