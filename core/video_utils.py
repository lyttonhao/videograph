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

import os
import cv2
import pickle as pkl
import natsort
from scipy.io import loadmat
import numpy as np
import imageio
import optparse
from moviepy.video.io.ffmpeg_reader import FFMPEG_VideoReader
import moviepy.editor as moviepyeditor
import paramiko
import image_utils

def read_frame_pathes(video_name, max_n_frames, video_path):
    folder_path = video_path + video_name
    frame_names = natsort.natsorted(os.walk(folder_path).next()[2])
    n_frames = len(frame_names)

    if n_frames > max_n_frames:
        frame_names = np.array(np.linspace(0, n_frames - 1, num=max_n_frames), dtype=int)
        frame_pathes = [folder_path + '/' + str(name) + '.jpeg' for name in frame_names]
    else:
        frame_pathes = [folder_path + '/' + name for name in frame_names]

    return frame_pathes

def read_frames_from_desk(video_name, max_n_frames):
    frame_pathes = read_frame_pathes(video_name, max_n_frames)
    frames = [cv2.imread(frame_path) for frame_path in frame_pathes]
    return frames

def read_frames_opencv(video_fullpath, frames_save_dir):
    '''
    opencv ugly and sluggish way to get frames
    :param video:
    :return:
    '''
    cap = cv2.VideoCapture(video_fullpath)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()

        frame_file_name = frames_save_dir + str(i) + '.jpeg'
        cv2.imwrite(frame_file_name, frame)
        i += 1

        if not ret:
            break

def read_frames_imageio(video_fullpath, frames_save_dir):
    '''
     much better/faster way to read video frames
    :param video_fullpath:
    :return:
    '''
    vid = imageio.get_reader(video_fullpath, 'ffmpeg')
    for i, frame in enumerate(vid):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_file_name = frames_save_dir + str(i) + '.jpeg'
        cv2.imwrite(frame_file_name, frame)

def video_uniform_sampling(spf, video_path, resize_type, is_local, verbose=False):
    assert resize_type in ['resize', 'resize_crop', 'resize_crop_scaled']

    resize_function = None
    if resize_type == 'resize':
        resize_function = image_utils.resize_frame
    elif resize_type == 'resize_crop':
        resize_function = image_utils.resize_crop
    elif resize_type == 'resize_crop_scaled':
        resize_function = image_utils.resize_crop_scaled

    cap = FFMPEG_VideoReader(video_path, False)
    cap.initialize()
    fps = cap.fps
    n_frames = cap.nframes
    duration = cap.duration
    n_samples = int(duration / float(spf))

    # check if no samples because the video duration is less than spf
    # then at least, get 1 frame of the video
    if n_samples == 0:
        n_samples = 1

    frame_size = 224
    frames = np.zeros(shape=(n_samples, frame_size, frame_size, 3), dtype='float32')
    for i in range(n_samples):
        num = i + 1
        if num % 100 == 0 and verbose:
            print (' ... reading frame %d/%d' % (num, n_samples))
        time_sec = i * spf
        frame = cap.get_frame(time_sec)
        # resize frame to fit in the array, it's going to be used by caffe anyway
        frame = resize_function(frame)
        # frame encoded as uint and values are from 0-255
        # but caffe needs float32 and values from 0-1
        frame = frame.astype('float32') / float(255)
        frames[i] = frame

    # very important, or we'd have memory leakage
    cap.__del__()

    return frames, fps, n_frames, duration

def video_uniform_sample_and_save_old(spf, video_path, frames_path, image_name_format, resize_type, verbose=False):
    if resize_type is not None:
        assert resize_type in ['resize', 'resize_crop', 'resize_crop_scaled']

    resize_function = None
    if resize_type == 'resize':
        resize_function = image_utils.resize_frame
    elif resize_type == 'resize_crop':
        resize_function = image_utils.resize_crop
    elif resize_type == 'resize_crop_scaled':
        resize_function = image_utils.resize_crop_scaled

    cap = FFMPEG_VideoReader(video_path, False)
    cap.initialize()
    fps = cap.fps
    n_frames = cap.nframes
    duration = cap.duration
    n_samples = int(duration / float(spf))

    # check if no samples because the video duration is less than spf
    # then at least, get 1 frame of the video
    if n_samples == 0:
        n_samples = 1

    for i in range(n_samples):
        num = i + 1
        if verbose:
            print (' ... reading frame %d/%d' % (num, n_samples))
        time_sec = i * spf
        frame = cap.get_frame(time_sec)

        if resize_type is not None:
            # resize frame to fit in the array, it's going to be used by caffe anyway
            frame = resize_function(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_path = image_name_format % (frames_path, num)
        cv2.imwrite(frame_path, frame)

    # very important, or we'd have memory leakage
    cap.close()

    return fps, n_frames, duration

def video_uniform_sample_and_save(spf, video_path, frames_path, image_name_format, resize_type, verbose=False):
    if resize_type is not None:
        assert resize_type in ['resize', 'resize_crop', 'resize_crop_scaled']

    resize_function = None
    if resize_type == 'resize':
        resize_function = image_utils.resize_frame
    elif resize_type == 'resize_crop':
        resize_function = image_utils.resize_crop
    elif resize_type == 'resize_crop_scaled':
        resize_function = image_utils.resize_crop_scaled

    cap = moviepyeditor.VideoFileClip(video_path)
    fps = cap.fps
    duration = cap.duration
    n_frames = int(fps * duration)
    n_samples = int(duration / float(spf))

    # check if no samples because the video duration is less than spf
    # then at least, get 1 frame of the video
    if n_samples == 0:
        n_samples = 1

    for i in range(n_samples):
        num = i + 1
        if verbose:
            print (' ... reading frame %d/%d' % (num, n_samples))
        time_sec = i * spf
        frame = cap.get_frame(time_sec)

        if resize_type is not None:
            # resize frame to fit in the array, it's going to be used by caffe anyway
            frame = resize_function(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_path = image_name_format % (frames_path, num)
        cv2.imwrite(frame_path, frame)

    # very important, or we'd have memory leakage
    cap.close()

    return fps, n_frames, duration

def video_uniform_sample_and_save_i3d(spf, video_path, frames_path, resize_type, verbose=False):
    if resize_type is not None:
        assert resize_type in ['resize', 'resize_crop', 'resize_crop_scaled']

    resize_function = None
    if resize_type == 'resize':
        resize_function = image_utils.resize_frame
    elif resize_type == 'resize_crop':
        resize_function = image_utils.resize_crop
    elif resize_type == 'resize_crop_scaled':
        resize_function = image_utils.resize_crop_scaled

    cap = moviepyeditor.VideoFileClip(video_path)
    fps = cap.fps
    duration = cap.duration
    n_frames = int(fps * duration)
    n_samples = int(duration / float(spf))
    n_frames_per_segment = 8
    fps_float = float(fps)

    # check if no samples because the video duration is less than spf
    # then at least, get 1 frame of the video
    if n_samples == 0:
        n_samples = 1

    for i in range(n_samples):
        num = i + 1
        if verbose:
            print (' ... reading frame %d/%d' % (num, n_samples))
        time_sec = i * spf

        t_stop = time_sec + n_frames_per_segment - 1
        if t_stop > duration:
            return

        # get 8 successive frames for i3d
        for j in range(n_frames_per_segment):
            t = time_sec + (j / fps_float)
            frame = cap.get_frame(t)

            # resize frame to fit in the array, it's going to be used by caffe anyway
            if resize_type is not None:
                frame = resize_function(frame)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_path = '%s/%06d_%d.jpg' % (frames_path, num, j + 1)
            cv2.imwrite(frame_path, frame)

    # very important, or we'd have memory leakage
    cap.close()

    return fps, n_frames, duration

def video_uniform_sample_n_frames_old(video_path, n_samples, max_dim):
    """
    Sample only n frames from the video.
    """

    raise Exception('Needs to add argument about resizing type')

    cap = FFMPEG_VideoReader(video_path, False)
    cap.initialize()
    fps = cap.fps
    n_frames = cap.nframes
    duration = cap.duration
    step = duration / (n_samples)

    frames = []
    for i in range(n_samples):
        time_sec = i * step
        frame = cap.get_frame(time_sec)
        # resize frame to fit in the array, it's going to be used by caffe anyway
        frame = image_utils.resize_keep_aspect_ratio_max_dim(frame, max_dim)
        # frame encoded as uint and values are from 0-255
        # but caffe needs float32 and values from 0-1
        frame = frame.astype('float32') / float(255)
        frames.append(frame)

    # very important, or we'd have memory leakage
    cap.__del__()

    return frames

def video_uniform_sample_n_frames_and_save(n_samples, video_path, frames_path, image_name_format, resize_type, verbose=False):
    if resize_type is not None:
        assert resize_type in ['resize', 'resize_crop', 'resize_crop_scaled']

    resize_function = None
    if resize_type == 'resize':
        resize_function = image_utils.resize_frame
    elif resize_type == 'resize_crop':
        resize_function = image_utils.resize_crop
    elif resize_type == 'resize_crop_scaled':
        resize_function = image_utils.resize_crop_scaled

    cap = moviepyeditor.VideoFileClip(video_path)
    fps = cap.fps
    duration = cap.duration
    step = duration / (n_samples)

    for i in range(n_samples):
        num = i + 1
        if verbose:
            print (' ... reading frame %d/%d' % (num, n_samples))

        time_sec = i * step
        frame = cap.get_frame(time_sec)

        if resize_type is not None:
            # resize frame to fit in the array, it's going to be used by caffe anyway
            frame = resize_function(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_path = image_name_format % (frames_path, num)
        cv2.imwrite(frame_path, frame)

    # very important, or we'd have memory leakage
    cap.reader.close()
    cap.close()
    del cap.reader
    del cap

    return fps, n_samples, duration

def video_save_frames(video_path, frames_path, image_name_format, resize_type=None, verbose=False):
    if resize_type is not None:
        assert resize_type in ['resize', 'resize_crop', 'resize_crop_scaled']

    resize_function = None
    if resize_type == 'resize':
        resize_function = image_utils.resize_frame
    elif resize_type == 'resize_crop':
        resize_function = image_utils.resize_crop
    elif resize_type == 'resize_crop_scaled':
        resize_function = image_utils.resize_crop_scaled

    cap = moviepyeditor.VideoFileClip(video_path)
    fps = float(cap.fps)
    duration_sec = cap.duration
    n_frames = int(fps * duration_sec)

    index = 0
    while True:
        time_sec = index / fps
        frame = cap.get_frame(time_sec)

        # resize frame to fit in the array, it's going to be used by caffe anyway
        if resize_type is not None:
            frame = resize_function(frame)

        index += 1
        if index > n_frames:
            break

        if verbose and index % 100 == 0:
            print (' ... reading frame %d/%d' % (index, n_frames))

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frame_path = image_name_format % (frames_path, index)
        cv2.imwrite(frame_path, frame)

    # very important, or we'd have memory leakage
    cap.reader.close()
    cap.close()
    del cap.reader
    del cap

    return fps, n_frames, duration_sec

def video_save_frames_specific_duration(action_num, video_num, video_path, frames_root_pathes, start_stop_sec, image_name_format, verbose=False):
    assert len(frames_root_pathes) == len(start_stop_sec)

    cap = FFMPEG_VideoReader(video_path, False)
    cap.initialize()
    fps = float(cap.fps)
    duration_sec = cap.duration
    img_dim = 224

    start_stop_sec = np.array(start_stop_sec)

    for i, s_s_sec in enumerate(start_stop_sec):
        start_sec, stop_sec = s_s_sec
        frame_root_path = frames_root_pathes[i]

        # offset of starting/stopping the action
        sec_offset = 0.25

        start_idx = int((start_sec + sec_offset) * fps)
        stop_idx = int((stop_sec + sec_offset) * fps) + 1

        if verbose:
            print ('action, video: %d, %d' % (action_num, video_num))
            print ('%d/%d' % (start_sec, stop_sec))
            print ('%d/%d' % (start_idx, stop_idx))

        for idx_frame in range(start_idx, stop_idx):
            time_sec = idx_frame / fps
            if verbose and idx_frame % 100 == 0:
                print ('... time_sec, frame: %d/%d' % (time_sec, idx_frame))

            frame = cap.get_frame(time_sec)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = resize_crop(frame, target_width=img_dim, target_height=img_dim)
            frame_path = image_name_format % (frame_root_path, idx_frame)
            cv2.imwrite(frame_path, frame)

    # very important, or we'd have memory leakage
    cap.__del__()

def get_video_info(video_path):
    # video_fps, video_n_frames, video_duration = video_utils.

    cap = FFMPEG_VideoReader(video_path, False)
    cap.initialize()
    fps = cap.fps
    n_frames = cap.nframes
    duration = cap.duration
    cap.close()
    del cap

    return fps, n_frames, duration

def get_regions(video_path, annot, resize_type, verbose=False):
    """
    Get the frames whose numbers are given in the "annot" dictionary.. Then, for each frame get the regions as specificed in the "annot" dictionary.
    Finally, return these regions.
    """

    assert resize_type in ['resize', 'resize_crop', 'resize_crop_scaled']

    resize_function = None
    if resize_type == 'resize':
        resize_function = image_utils.resize_frame
    elif resize_type == 'resize_crop':
        resize_function = image_utils.resize_crop
    elif resize_type == 'resize_crop_scaled':
        resize_function = image_utils.resize_crop_scaled

    cap = FFMPEG_VideoReader(video_path, False)
    cap.initialize()
    fps = float(cap.fps)
    n_frames = cap.nframes
    duration = cap.duration
    n_regions = sum([len(v) for k, v in annot.iteritems()])

    frame_size = 224
    bbox_resize_factor = 2
    regions = np.zeros(shape=(n_regions, frame_size, frame_size, 3), dtype='float32')
    region_idx = -1

    frame_nums = annot.keys()
    for frame_num in frame_nums:

        if (region_idx + 1) % 100 == 0 and verbose:
            print (' ... reading region %d/%d' % (region_idx + 1, n_regions))

        # get the frame
        i = frame_num - 1
        time_sec = i / fps
        frame = cap.get_frame(time_sec)

        # get the regions (resized) from the frame
        regions_info = annot[frame_num]
        for region_info in regions_info:
            region_idx += 1
            bbox = region_info[1:5]
            bbox = np.multiply(bbox, bbox_resize_factor).astype(np.int)
            x1, y1, x2, y2 = bbox
            region = frame[y1:y2, x1:x2]
            # resize frame to fit in the array, it's going to be used by caffe anyway
            region = resize_function(region)
            # frame encoded as uint and values are from 0-255, but caffe needs float32 and values from 0-1
            region = region.astype('float32') / float(255)
            regions[region_idx] = region

    # very important, or we'd have memory leakage
    cap.__del__()

    return regions
