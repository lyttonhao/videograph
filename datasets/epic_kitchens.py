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
For training videos
total no. of videos downloaded: 274
total no. of videos with action narration: 272
total no. of noun classes: 352
total no. of verb classes: 125

For training videos, we have 28 actors, and 274 videos in total,
On average, 9.78 videos per actor

For rgb_frames, it's either one of these sizes:
(456, 256) => original: (1920, 1080) or (1280, 720) => ratio 1.78
(342, 256) => original: (1920, 1440)                => ratio 1.33

There are 28472 annotations for training,
the duration (in sec.) per each is min, max, avg: 0.5, 207, 3.74

The annotation for actions in videos
00:uid
01:participant_id
02:video_id
03:narration
04:start_timestamp
05:stop_timestamp
06:start_frame
07:stop_frame
08:verb
09:verb_class
10:noun
11:noun_class
12:all_nouns
13:all_noun_classes
"""

import random
import time
import threading
import numpy as np
from natsort import natsort
from datetime import datetime
from keras.utils import multi_gpu_utils

from core import utils, image_utils, configs, sobol
from core.utils import Path as Pth
from core import const as c

# region Constants

N_NOUNS_MANY_SHOT = 71
N_VERBS_MANY_SHOT = 26
N_ACTNS_MANY_SHOT = 819

# endregion

# region Prepare Annotation: Nouns, Verbs, Actions

def _702_prepare_annot_id_of_many_shots():
    """
    Prepare list of annot ids whose both noun and verb is many_shot.
    :return:
    """

    many_shot_noun_csv_path = Pth('EPIC-Kitchens/annotations/EPIC_many_shot_nouns.csv')
    many_shot_verb_csv_path = Pth('EPIC-Kitchens/annotations/EPIC_many_shot_verbs.csv')
    annot_lines_path = Pth('EPIC-Kitchens/annotations/EPIC_train_action_labels.csv')
    annot_idxes_many_shots_noun_verb_path = Pth('EPIC-Kitchens/annotations/annot_idxes_many_shots_noun_verb.pkl')
    annot_invalid_path = Pth('EPIC-Kitchens/annotations/EPIC_train_invalid_labels.csv')
    video_info_dict_path = Pth('EPIC-Kitchens/annotations/EPIC_video_info_dict.pkl')

    many_shot_noun = utils.csv_load(many_shot_noun_csv_path)
    many_shot_verb = utils.csv_load(many_shot_verb_csv_path)
    annot_invalid = utils.csv_load(annot_invalid_path)
    video_info_dict = utils.pkl_load(video_info_dict_path)

    # I3D takes 8 frames, we only sample frames with rate 29.97
    # so, one I3D-segment: 0.267 sec
    #     two I3D-segment: 0.534 sec
    min_fps = 29.97
    min_duration = 0.5

    annot_invalid_ids = annot_invalid[:, 0]
    many_shot_noun_ids = many_shot_noun[:, 0]
    many_shot_verb_ids = many_shot_verb[:, 0]

    # uid, participant_id, video_id, narration, start_timestamp, stop_timestamp, start_frame, stop_frame, verb,verb_class, noun, noun_class, all_nouns, all_noun_classes
    annot_lines = utils.csv_load(annot_lines_path)
    n_annot = len(annot_lines)

    many_shots_annot_count_n_v = 0
    annot_idxes_many_shots_noun_verb = []

    # loop on action_annotations
    for idx in range(n_annot):

        # uid,participant_id,video_id,narration,start_timestamp,stop_timestamp,start_frame,stop_frame,verb,verb_class,noun,noun_class,all_nouns,all_noun_classes
        annot_line = annot_lines[idx]

        # get the current action
        annot_idx = annot_line[0]
        verb_idx = annot_line[9]
        noun_idx = annot_line[11]
        frame_start = annot_line[5]
        frame_stop = annot_line[6]
        video_id = annot_line[1]
        video_fps = video_info_dict[video_id][2]

        duration = __get_duration_in_sec_using_timestamp_strings(annot_line[4], annot_line[5])
        duration_ = __get_duration_in_sec_using_frame_nums(frame_start, frame_stop, video_fps)

        if duration < min_duration:
            continue

        # to consider annotation instance, it has to be:
        # verb of many_instance
        # noun of many_instance
        # not in the invalid_list
        # has enough duration
        if noun_idx in many_shot_noun_ids and verb_idx in many_shot_verb_ids and annot_idx not in annot_invalid_ids:
            many_shots_annot_count_n_v += 1
            annot_idxes_many_shots_noun_verb.append(annot_idx)

    print n_annot
    print many_shots_annot_count_n_v

    annot_idxes_many_shots_noun_verb = np.array(annot_idxes_many_shots_noun_verb)
    utils.pkl_dump(annot_idxes_many_shots_noun_verb, annot_idxes_many_shots_noun_verb_path)

def _703_prepare_data_splits():
    """
    Sample fram pathes for the i3d model.
    :return:
    """

    annot_dict_path = Pth('EPIC-Kitchens/annotations/EPIC_train_action_labels_dict.pkl')
    annot_idxes_many_shots_path = Pth('EPIC-Kitchens/annotations/annot_idxes_many_shots_noun_verb.pkl')
    video_names_splits_path = Pth('EPIC-Kitchens/annotations/video_names_splits.pkl')

    annot_idxes_many_shots = utils.pkl_load(annot_idxes_many_shots_path)
    annot_dict = utils.pkl_load(annot_dict_path)

    # split_ratio
    split_ratio = 0.8
    person_videos_dict = {}

    # first loop to collect all unique video ids
    for annot_id in annot_idxes_many_shots:
        annot_line = annot_dict[annot_id]
        person_id = annot_line[0]
        video_id = annot_line[1]
        if person_id not in person_videos_dict:
            person_videos_dict[person_id] = []

        person_videos_dict[person_id].append(video_id)

    for person_id in person_videos_dict:
        video_names = natsort.natsorted(np.unique(person_videos_dict[person_id]))
        person_videos_dict[person_id] = video_names

    # now that we have collected the persons, and their videos, see how much videos if we split
    video_names_tr = []
    video_names_te = []

    for person_id in person_videos_dict:
        v_names = person_videos_dict[person_id]
        idx = int(len(v_names) * split_ratio)
        v_names_tr = v_names[:idx]
        v_names_te = v_names[idx:]
        video_names_tr += v_names_tr
        video_names_te += v_names_te

    video_names_tr = np.array(video_names_tr)
    video_names_te = np.array(video_names_te)

    print len(video_names_tr) + len(video_names_te)
    print len(video_names_tr)
    print len(video_names_te)

    # save video names
    utils.pkl_dump((video_names_tr, video_names_te), video_names_splits_path)

def _704_prepare_many_shots_noun_verb_action_ids():
    """
    Prepeare two dicts of nouns and verbs to convert from id to many_shot id. All ids are zero-indexed.
    71 noun classes
    26 verb classes
    xx actions
    :return:
    """

    annot_dict_path = Pth('EPIC-Kitchens/annotations/EPIC_train_action_labels_dict.pkl')
    annot_idxes_many_shots_path = Pth('EPIC-Kitchens/annotations/annot_idxes_many_shots_noun_verb.pkl')
    noun_ids_many_shots_dict_path = Pth('EPIC-Kitchens/annotations/noun_ids_many_shots_dict.pkl')
    verb_ids_many_shots_dict_path = Pth('EPIC-Kitchens/annotations/verb_ids_many_shots_dict.pkl')
    actn_ids_many_shots_dict_path = Pth('EPIC-Kitchens/annotations/actn_ids_many_shots_dict.pkl')
    actn_ids_many_shots_list_path = Pth('EPIC-Kitchens/annotations//EPIC_many_shot_actions.csv')

    annot_idxes_many_shots = utils.pkl_load(annot_idxes_many_shots_path)
    annot_dict = utils.pkl_load(annot_dict_path)

    # get all verb_ids, noun_ids
    noun_ids = [annot_dict[annot_id][10] for annot_id in annot_idxes_many_shots]
    verb_ids = [annot_dict[annot_id][8] for annot_id in annot_idxes_many_shots]
    actn_ids = __get_action_ids_from_annotation(actn_ids_many_shots_list_path)

    noun_ids = np.sort(np.unique(noun_ids))
    verb_ids = np.sort(np.unique(verb_ids))

    n_nouns = len(noun_ids)
    n_verbs = len(verb_ids)
    n_actns = len(actn_ids)

    # these dictionaries get the id of many_shot (noun or verb) given the original (noun or verb)
    many_shot_noun_ids_dict = dict(zip(noun_ids, np.arange(n_nouns)))
    many_shot_verb_ids_dict = dict(zip(verb_ids, np.arange(n_verbs)))
    many_shot_actn_ids_dict = dict(zip(actn_ids, np.arange(n_actns)))

    utils.pkl_dump(many_shot_noun_ids_dict, noun_ids_many_shots_dict_path)
    utils.pkl_dump(many_shot_verb_ids_dict, verb_ids_many_shots_dict_path)
    utils.pkl_dump(many_shot_actn_ids_dict, actn_ids_many_shots_dict_path)

def _705_sample_frame_pathes_i3d():
    """
    Sample fram pathes for the i3d model.
    According to values here, we will have number of samples:
    total, train, test: 27908, 21427, 6481
    """

    noun_ids_dict_path = Pth('EPIC-Kitchens/annotations/noun_ids_many_shots_dict.pkl')
    verb_ids_dict_path = Pth('EPIC-Kitchens/annotations/verb_ids_many_shots_dict.pkl')

    video_info_dict_path = Pth('EPIC-Kitchens/annotations/EPIC_video_info_dict.pkl')
    annot_dict_path = Pth('EPIC-Kitchens/annotations/EPIC_train_action_labels_dict.pkl')
    annot_idxes_many_shots_path = Pth('EPIC-Kitchens/annotations/annot_idxes_many_shots_noun_verb.pkl')
    video_names_splits_path = Pth('EPIC-Kitchens/annotations/video_names_splits.pkl')

    root_path_id = 0 if configs.is_local_machine() else 5
    frames_root_path = Pth('EPIC-Kitchens/frames_rgb_resized/train', root_type=c.ROOT_PATH_TYPES[root_path_id])

    noun_ids_dict = utils.pkl_load(noun_ids_dict_path)
    verb_ids_dict = utils.pkl_load(verb_ids_dict_path)

    annot_idxes_many_shots = utils.pkl_load(annot_idxes_many_shots_path)
    video_info_dict = utils.pkl_load(video_info_dict_path)

    annot_dict = utils.pkl_load(annot_dict_path)
    (video_names_tr, video_names_te) = utils.pkl_load(video_names_splits_path)

    # stride between two sampled segment in the same annotation instance
    sampling_stride_tr = 17
    sampling_stride_te = 50
    # fps to use when sampling frames
    ideal_fps = 29
    # how many successive frames in each sampled segment
    n_frames_per_segment = 8

    frame_pathes_tr = []
    frame_pathes_te = []
    gt_noun_tr = []
    gt_verb_tr = []
    gt_noun_te = []
    gt_verb_te = []

    # shuffle ids
    random.shuffle(annot_idxes_many_shots)

    for annot_id in annot_idxes_many_shots:
        # participant_id,video_id,narration,start_timestamp,stop_timestamp,start_frame,stop_frame,verb,verb_class,noun,noun_class,all_nouns,all_noun_classes
        annot_line = annot_dict[annot_id]
        person_id = annot_line[0]
        video_id = annot_line[1]

        frame_start = annot_line[5]
        frame_stop = annot_line[6]

        noun_id = noun_ids_dict[annot_line[10]]
        verb_id = verb_ids_dict[annot_line[8]]

        is_training = video_id not in video_names_te
        sampling_stride = sampling_stride_tr if is_training else sampling_stride_te

        video_fps = video_info_dict[video_id][2]
        video_fps = int(video_fps)
        fps_factor = video_fps / ideal_fps

        # sample frames, an example of frame name: frame_0000000001
        video_frame_nums = __random_sample_frames_for_i3d(frame_start, frame_stop, fps_factor, sampling_stride, n_frames_per_segment)
        video_frame_pathes = ['%s/%s/%s/frame_%010d.jpg' % (frames_root_path, person_id, video_id, n) for n in video_frame_nums]

        # reshape
        video_frame_pathes = np.reshape(video_frame_pathes, (-1, n_frames_per_segment)).tolist()
        n_video_segments = len(video_frame_pathes)
        video_noun_ids = [noun_id] * n_video_segments
        video_verb_ids = [verb_id] * n_video_segments

        if is_training:
            frame_pathes_tr += video_frame_pathes
            gt_noun_tr += video_noun_ids
            gt_verb_tr += video_verb_ids
        else:
            frame_pathes_te += video_frame_pathes
            gt_noun_te += video_noun_ids
            gt_verb_te += video_verb_ids

    frame_pathes_tr = np.array(frame_pathes_tr)
    frame_pathes_te = np.array(frame_pathes_te)

    gt_noun_tr = np.array(gt_noun_tr)
    gt_noun_te = np.array(gt_noun_te)
    gt_verb_tr = np.array(gt_verb_tr)
    gt_verb_te = np.array(gt_verb_te)

    # binarize labels
    classes_noun = np.arange(N_NOUNS_MANY_SHOT)
    classes_verb = np.arange(N_VERBS_MANY_SHOT)
    gt_noun_tr = utils.label_binarize(gt_noun_tr, classes_noun)
    gt_noun_te = utils.label_binarize(gt_noun_te, classes_noun)
    gt_verb_tr = utils.label_binarize(gt_verb_tr, classes_verb)
    gt_verb_te = utils.label_binarize(gt_verb_te, classes_verb)

    data = (frame_pathes_tr, gt_noun_tr, gt_verb_tr, frame_pathes_te, gt_noun_te, gt_verb_te)

    return data

def __random_sample_frames_for_i3d(frame_start, frame_stop, fps_factor, stride, n_frames_per_segment):
    frame_nums = np.arange(frame_start, frame_stop + 1, step=fps_factor)
    n_frames = len(frame_nums)

    # sample frames according to the strides
    step = n_frames_per_segment * stride
    n_segments = len(np.arange(0, n_frames, step=step, dtype=np.int))
    idces_start = np.random.randint(low=0, high=n_frames - n_frames_per_segment, size=(n_segments,))
    idces_start = np.sort(idces_start)
    idx = []
    for idx_start in idces_start:
        idx += np.arange(idx_start, idx_start + n_frames_per_segment, dtype=np.int).tolist()

    sampled_frame_nums = frame_nums[idx]
    return sampled_frame_nums

def __get_action_ids_from_annotation(annotation_path):
    annotation_lines = utils.csv_load(annotation_path)
    ids = []
    for l in annotation_lines:
        id = l[0].replace('(', '').replace(')', '')
        splits = id.split(', ')
        id = '_'.join(splits)
        ids.append(id)

    ids = np.array(ids)
    return ids

# endregion

# region Prepare Annotation: Video-level

def _800_prepare_video_frames_path_dict():
    frame_relative_pathes_dict_path = Pth('EPIC-Kitchens/annotations/frame_relative_pathes_dict.pkl')
    video_names_splits_path = Pth('EPIC-Kitchens/annotations/video_names_splits.pkl')
    imgs_root_path = Pth('EPIC-Kitchens/frames_rgb_resized/train')

    (video_names_tr, video_names_te) = utils.pkl_load(video_names_splits_path)
    video_names = np.hstack((video_names_tr, video_names_te))

    frame_relative_pathes_dict = {}
    n_videos = len(video_names)
    for idx, video_id in enumerate(video_names):
        utils.print_counter(idx, n_videos)

        person_id = video_id.split('_')[0]
        video_frames_root_path = '%s/%s/%s' % (imgs_root_path, person_id, video_id)
        video_frames_names = utils.file_names(video_frames_root_path, is_nat_sort=True)
        video_frames_names = np.array(video_frames_names)
        video_frames_relative_pathes = np.array(['%s/%s/%s' % (person_id, video_id, n) for n in video_frames_names])
        frame_relative_pathes_dict[video_id] = video_frames_relative_pathes

    utils.pkl_dump(frame_relative_pathes_dict, frame_relative_pathes_dict_path)

def _800_spit_video_frames_relative_pathes():
    video_names_splits_path = Pth('EPIC-Kitchens/annotations/video_names_splits.pkl')
    frame_relative_pathes_dict_path = Pth('EPIC-Kitchens/annotations/frame_relative_pathes_dict.pkl')
    frame_relative_pathes_dict_tr_path = Pth('EPIC-Kitchens/annotations/frame_relative_pathes_dict_tr.pkl')
    frame_relative_pathes_dict_te_path = Pth('EPIC-Kitchens/annotations/frame_relative_pathes_dict_te.pkl')

    (video_names_tr, video_names_te) = utils.pkl_load(video_names_splits_path)
    frames_dict = utils.pkl_load(frame_relative_pathes_dict_path)

    dict_tr = dict()
    dict_te = dict()

    for v_name in video_names_tr:
        dict_tr[v_name] = frames_dict[v_name]

    for v_name in video_names_te:
        dict_te[v_name] = frames_dict[v_name]

    utils.pkl_dump(dict_tr, frame_relative_pathes_dict_tr_path)
    utils.pkl_dump(dict_te, frame_relative_pathes_dict_te_path)

def _801_prepare_video_level_annotation():
    video_names_splits_path = Pth('EPIC-Kitchens/annotations/video_names_splits.pkl')
    noun_ids_dict_path = Pth('EPIC-Kitchens/annotations/noun_ids_many_shots_dict.pkl')
    verb_ids_dict_path = Pth('EPIC-Kitchens/annotations/verb_ids_many_shots_dict.pkl')
    actn_ids_dict_path = Pth('EPIC-Kitchens/annotations/actn_ids_many_shots_dict.pkl')
    annot_dict_path = Pth('EPIC-Kitchens/annotations/EPIC_train_action_labels_dict.pkl')
    annot_idxes_many_shots_path = Pth('EPIC-Kitchens/annotations/annot_idxes_many_shots_noun_verb.pkl')
    annot_video_level_many_shots_path = Pth('EPIC-Kitchens/annotations/annot_video_level_many_shots.pkl')

    noun_ids_dict = utils.pkl_load(noun_ids_dict_path)
    verb_ids_dict = utils.pkl_load(verb_ids_dict_path)
    actn_ids_dict = utils.pkl_load(actn_ids_dict_path)
    annot_dict = utils.pkl_load(annot_dict_path)
    annot_idxes_many_shots = utils.pkl_load(annot_idxes_many_shots_path)
    (video_names_tr, video_names_te) = utils.pkl_load(video_names_splits_path)

    y_noun_dict_tr = dict()
    y_verb_dict_tr = dict()
    y_actn_dict_tr = dict()

    y_noun_dict_te = dict()
    y_verb_dict_te = dict()
    y_actn_dict_te = dict()

    # loop on the annotations of videos and get accumulate the ground_truth of videos
    for annot_id in annot_idxes_many_shots:

        annot_line = annot_dict[annot_id]
        video_id = annot_line[1]
        noun_id = annot_line[10]
        verb_id = annot_line[8]
        actn_id = '%d_%d' % (verb_id,noun_id)

        noun_id = noun_ids_dict[noun_id]
        verb_id = verb_ids_dict[verb_id]
        actn_id = actn_ids_dict[actn_id]

        is_training = video_id not in video_names_te

        if is_training:
            if video_id not in y_noun_dict_tr:
                y_noun_dict_tr[video_id] = np.zeros((N_NOUNS_MANY_SHOT,), dtype=np.int32)

            if video_id not in y_verb_dict_tr:
                y_verb_dict_tr[video_id] = np.zeros((N_VERBS_MANY_SHOT,), dtype=np.int32)

            if video_id not in y_actn_dict_tr:
                y_actn_dict_tr[video_id] = np.zeros((N_ACTNS_MANY_SHOT,), dtype=np.int32)

            y_noun_dict_tr[video_id][noun_id] = 1
            y_verb_dict_tr[video_id][verb_id] = 1
            y_actn_dict_tr[video_id][actn_id] = 1

        else:
            if video_id not in y_noun_dict_te:
                y_noun_dict_te[video_id] = np.zeros((N_NOUNS_MANY_SHOT,), dtype=np.int32)

            if video_id not in y_verb_dict_te:
                y_verb_dict_te[video_id] = np.zeros((N_VERBS_MANY_SHOT,), dtype=np.int32)

            if video_id not in y_actn_dict_te:
                y_actn_dict_te[video_id] = np.zeros((N_ACTNS_MANY_SHOT,), dtype=np.int32)

            y_noun_dict_te[video_id][noun_id] = 1
            y_verb_dict_te[video_id][verb_id] = 1
            y_actn_dict_te[video_id][actn_id] = 1

    counts1 = np.array([np.where(x == 1)[0][0] if len(np.where(x == 1)[0]) > 0 else 0 for x in np.transpose(y_noun_dict_tr.values())])
    counts2 = np.array([np.where(x == 1)[0][0] if len(np.where(x == 1)[0]) > 0 else 0 for x in np.transpose(y_verb_dict_tr.values())])
    counts3 = np.array([np.where(x == 1)[0][0] if len(np.where(x == 1)[0]) > 0 else 0 for x in np.transpose(y_actn_dict_tr.values())])
    counts4 = np.array([np.where(x == 1)[0][0] if len(np.where(x == 1)[0]) > 0 else 0 for x in np.transpose(y_noun_dict_te.values())])
    counts5 = np.array([np.where(x == 1)[0][0] if len(np.where(x == 1)[0]) > 0 else 0 for x in np.transpose(y_verb_dict_te.values())])
    counts6 = np.array([np.where(x == 1)[0][0] if len(np.where(x == 1)[0]) > 0 else 0 for x in np.transpose(y_actn_dict_te.values())])

    print ('Number of zeros')
    print len(np.where(counts1 == 0)[0])
    print len(np.where(counts2 == 0)[0])
    print len(np.where(counts3 == 0)[0])
    print len(np.where(counts4 == 0)[0])
    print len(np.where(counts5 == 0)[0])
    print len(np.where(counts6 == 0)[0])

    # Number of zeros
    # 4
    # 2
    # 60
    # 21
    # 12
    # 352

    y_noun_list_tr = np.array([y_noun_dict_tr[n] for n in video_names_tr])
    y_verb_list_tr = np.array([y_verb_dict_tr[n] for n in video_names_tr])
    y_actn_list_tr = np.array([y_actn_dict_tr[n] for n in video_names_tr])

    y_noun_list_te = np.array([y_noun_dict_te[n] for n in video_names_te])
    y_verb_list_te = np.array([y_verb_dict_te[n] for n in video_names_te])
    y_actn_list_te = np.array([y_actn_dict_te[n] for n in video_names_te])

    print ('Annotation shapes')
    print (y_noun_list_tr.shape)
    print (y_noun_list_te.shape)

    print (y_verb_list_tr.shape)
    print (y_verb_list_te.shape)

    print (y_actn_list_tr.shape)
    print (y_actn_list_te.shape)

    # Annotation shapes
    # (205, 71)
    # (67, 71)
    # (205, 26)
    # (67, 26)
    # (205, 819)
    # (67, 819)

    data = (y_noun_list_tr, y_verb_list_tr, y_actn_list_tr, y_noun_list_te, y_verb_list_te, y_actn_list_te)
    utils.pkl_dump(data, annot_video_level_many_shots_path)

def _802_uniform_sample_frames_for_i3d_test_video_level():
    video_names_splits_path = Pth('EPIC-Kitchens/annotations/video_names_splits.pkl')
    frame_relative_pathes_dict_tr_path = Pth('EPIC-Kitchens/annotations/frame_relative_pathes_dict_tr.pkl')
    frame_relative_pathes_dict_te_path = Pth('EPIC-Kitchens/annotations/frame_relative_pathes_dict_te.pkl')
    sampled_frames_relative_pathes = Pth('EPIC-Kitchens/annotations/frame_relative_pathes_uniform_sample.pkl')

    (video_names_tr, video_names_te) = utils.pkl_load(video_names_splits_path)
    frame_relative_pathes_dict_tr = utils.pkl_load(frame_relative_pathes_dict_tr_path)
    frame_relative_pathes_dict_te = utils.pkl_load(frame_relative_pathes_dict_te_path)

    n_timesteps = 64
    n_frames_per_segment = 8
    n_frames_per_video = n_timesteps * n_frames_per_segment

    sampled_frames_tr = __uniform_sample_frames_per_video_for_i3d(video_names_tr, frame_relative_pathes_dict_tr, n_frames_per_segment, n_frames_per_video)
    sampled_frames_te = __uniform_sample_frames_per_video_for_i3d(video_names_te, frame_relative_pathes_dict_te, n_frames_per_segment, n_frames_per_video)

    data = (sampled_frames_tr, sampled_frames_te)
    utils.pkl_dump(data, sampled_frames_relative_pathes)

def _803_random_sample_frames_for_i3d_test_video_level():
    video_names_splits_path = Pth('EPIC-Kitchens/annotations/video_names_splits.pkl')
    frame_relative_pathes_dict_tr_path = Pth('EPIC-Kitchens/annotations/frame_relative_pathes_dict_tr.pkl')
    frame_relative_pathes_dict_te_path = Pth('EPIC-Kitchens/annotations/frame_relative_pathes_dict_te.pkl')

    root_path_id = 0 if configs.is_local_machine() else 5
    frames_root_path = Pth('EPIC-Kitchens/frames_rgb_resized/train', root_type=c.ROOT_PATH_TYPES[root_path_id])

    (video_names_tr, video_names_te) = utils.pkl_load(video_names_splits_path)
    frame_relative_pathes_dict_tr = utils.pkl_load(frame_relative_pathes_dict_tr_path)
    frame_relative_pathes_dict_te = utils.pkl_load(frame_relative_pathes_dict_te_path)

    # loop on the videos and sample frames for i3d

    n_timesteps = 64
    n_frames_per_segment = 8
    n_frames_per_video = n_timesteps * n_frames_per_segment

    sampled_frames_tr = __random_sample_frames_per_video_for_i3d(video_names_tr, frames_root_path, frame_relative_pathes_dict_tr, n_frames_per_segment, n_frames_per_video)
    sampled_frames_te = __random_sample_frames_per_video_for_i3d(video_names_te, frames_root_path, frame_relative_pathes_dict_te, n_frames_per_segment, n_frames_per_video)

    return (sampled_frames_tr, sampled_frames_te)

def _804_random_sample_frames_for_i3d_test_video_level_by_split(split_type='train'):

    assert split_type in ['train', 'test'], 'Sorry, unknown split type: %s' % (split_type)
    is_train = split_type == 'train'

    file_name_suffix = 'tr' if is_train else 'te'
    root_path_id = 0 if configs.is_local_machine() else 5

    frames_root_path = Pth('EPIC-Kitchens/frames_rgb_resized/train', root_type=c.ROOT_PATH_TYPES[root_path_id])
    frame_relative_pathes_dict_path = Pth('EPIC-Kitchens/annotations/frame_relative_pathes_dict_%s.pkl', (file_name_suffix,))
    video_names_splits_path = Pth('EPIC-Kitchens/annotations/video_names_splits.pkl')

    (video_names_tr, video_names_te) = utils.pkl_load(video_names_splits_path)
    video_names = video_names_tr if is_train else video_names_te
    del video_names_tr
    del video_names_te

    frame_relative_pathes_dict = utils.pkl_load(frame_relative_pathes_dict_path)

    # loop on the videos and sample frames for i3d
    n_timesteps = 64
    n_frames_per_segment = 8
    n_frames_per_video = n_timesteps * n_frames_per_segment
    sampled_frames = __random_sample_frames_per_video_for_i3d(video_names, frames_root_path, frame_relative_pathes_dict, n_frames_per_segment, n_frames_per_video)

    return sampled_frames

def __uniform_sample_frames_per_video_for_i3d(video_names, frame_relative_pathes_dict, n_frames_per_segment, n_frames_per_video):
    sampled_frame_pathes = []

    # loop on videos and sample
    for v_name in video_names:

        # i3d model accepts sequence of 8 frames
        v_frames_relative_pathes = frame_relative_pathes_dict[v_name]
        n_frames = len(v_frames_relative_pathes)
        n_segments = int(n_frames_per_video / n_frames_per_segment)

        if n_frames < n_frames_per_video:
            step = (n_frames - n_frames_per_segment) / float(n_segments)
            idces_start = np.arange(0, n_frames - n_frames_per_segment, step=step, dtype=np.int)
            idx = []
            for idx_start in idces_start:
                idx += np.arange(idx_start, idx_start + n_frames_per_segment, dtype=np.int).tolist()
        elif n_frames == n_frames_per_video:
            idx = np.arange(n_frames_per_video)
        else:
            step = n_frames / float(n_segments)
            idces_start = np.arange(0, n_frames, step=step, dtype=np.int)
            idx = []
            for idx_start in idces_start:
                idx += np.arange(idx_start, idx_start + n_frames_per_segment, dtype=np.int).tolist()

        v_sampled = v_frames_relative_pathes[idx]
        sampled_frame_pathes.append(v_sampled)

    sampled_frame_pathes = np.array(sampled_frame_pathes)
    return sampled_frame_pathes

def __random_sample_frames_per_video_for_i3d(video_names, frames_root_path, frame_relative_pathes_dict, n_frames_per_segment, n_frames_per_video):
    sampled_frame_pathes = dict()

    # loop on videos and sample
    for v_name in video_names:

        # i3d model accepts sequence of 8 frames
        v_frames_relative_pathes = frame_relative_pathes_dict[v_name]
        n_frames = len(v_frames_relative_pathes)
        n_segments = int(n_frames_per_video / n_frames_per_segment)

        if n_frames < n_frames_per_video:
            idces_start = np.random.randint(0, n_frames - n_frames_per_segment, (n_segments,))
            idces_start = np.sort(idces_start)
            idx = []
            for idx_start in idces_start:
                idx += np.arange(idx_start, idx_start + n_frames_per_segment, dtype=np.int).tolist()
        elif n_frames == n_frames_per_video:
            idx = np.arange(n_frames_per_video)
        else:
            idces_start = np.random.randint(0, n_frames - n_frames_per_segment, (n_segments,))
            idces_start = np.sort(idces_start)
            idx = []
            for idx_start in idces_start:
                idx += np.arange(idx_start, idx_start + n_frames_per_segment, dtype=np.int).tolist()

        v_sampled = v_frames_relative_pathes[idx]
        assert len(v_sampled) == n_frames_per_video
        v_sampled = np.array(['%s/%s' % (frames_root_path, p) for p in v_sampled])
        sampled_frame_pathes[v_name] = v_sampled

    return sampled_frame_pathes

# endregion

# region Extract Features

def _901_extract_features_i3d():
    """
    Extract i3d features.
    :return:
    """

    n_threads = 32
    n_frames_per_segment = 8
    n_segments_per_video = 64
    n_frames_per_video = n_segments_per_video * n_frames_per_segment

    video_names_splits_path = Pth('EPIC-Kitchens/annotations/video_names_splits.pkl')
    sampled_frames_relative_pathes = Pth('EPIC-Kitchens/annotations/frame_relative_pathes_uniform_sample.pkl')

    root_path_id = 0 if configs.is_local_machine() else 5
    frames_root_path = Pth('EPIC-Kitchens/frames_rgb_resized/train', root_type=c.ROOT_PATH_TYPES[root_path_id])
    features_path = Pth('EPIC-Kitchens/features/features_i3d_mixed_5c_%d_frames.h5', (n_frames_per_video,), root_type=c.ROOT_PATH_TYPES[root_path_id])

    (video_names_tr, video_names_te) = utils.pkl_load(video_names_splits_path)
    (sampled_frames_tr, sampled_frames_te) = utils.pkl_load(sampled_frames_relative_pathes)

    video_names = np.hstack((video_names_tr, video_names_te))
    sampled_frames = np.vstack((sampled_frames_tr, sampled_frames_te))
    n_videos = len(video_names)
    n_tr = len(video_names_tr)

    img_reader = image_utils.AsyncImageReaderEpicKitchensForI3dKerasModel(n_threads=n_threads)
    v_frames_pathes = np.array(['%s/%s' % (frames_root_path, p) for p in sampled_frames[0]])
    img_reader.load_imgs_in_batch(v_frames_pathes)

    model = i3d_keras_epic_kitchens.Inception_Inflated3d_Backbone()
    print (model.summary())
    model = multi_gpu_utils.multi_gpu_model(model, 4)

    dataset_features = np.zeros((n_videos, n_segments_per_video, 7, 7, 1024), dtype=np.float32)

    # loop on videos, extract features and save them
    for idx_video, v_name in enumerate(video_names):

        video_num = idx_video + 1

        # wait untill the image_batch is loaded
        t1 = time.time()
        while img_reader.is_busy():
            threading._sleep(0.1)
        t2 = time.time()
        duration_waited = t2 - t1
        print ('...... video %d/%d:, waited: %d' % (video_num, n_videos, duration_waited))

        # get the video frames
        video_frames = img_reader.get_images()

        # reshape to get the segments in one dimension
        frames_shape = video_frames.shape
        frames_shape = [n_segments_per_video, n_frames_per_segment] + list(frames_shape[1:])
        video_frames = np.reshape(video_frames, frames_shape)

        # pre-load for the next video
        if video_num < n_videos:
            v_frames_pathes = np.array(['%s/%s' % (frames_root_path, p) for p in sampled_frames[idx_video + 1]])
            img_reader.load_imgs_in_batch(v_frames_pathes)

        # extract features
        features = model.predict(video_frames, verbose=0)

        # remove the temporal dimension
        features = np.squeeze(features, axis=1)

        # append feature to list
        dataset_features[idx_video] = features

    # split features
    features_tr = dataset_features[:n_tr]
    features_te = dataset_features[n_tr:]

    print features_tr.shape
    print features_te.shape

    # save features
    utils.h5_dump_multi((features_tr, features_te), ['x_tr', 'x_te'], features_path)

# endregion

# region Centroids

def _602_generate_centroids(n_centroids, n_dims):
    pass

    c1_path = Pth('EPIC-Kitchens/features/centroids_random_%d_centroids.pkl', (n_centroids,))
    c2_path = Pth('EPIC-Kitchens/features/centroids_sobol_%d_centroids.pkl', (n_centroids,))

    # centroids as random vectors
    c1 = np.random.rand(n_centroids, n_dims)

    # centroids as sobol sequence
    c2 = sobol.sobol_generate(n_dims, n_centroids)
    c2 = np.array(c2)

    print c1.shape
    print c2.shape

    # save centroids
    utils.pkl_dump(c1, c1_path)
    utils.pkl_dump(c2, c2_path)

# endregion

# region Misc

def __convert_string_to_timestamp(timestamp_string):
    time_format = '%H:%M:%S.%f'
    timestamp = datetime.strptime(timestamp_string, time_format)

    return timestamp

def __convert_timestamp_to_seconds(timestamp_string):
    t2 = __convert_string_to_timestamp(timestamp_string)
    t1 = __convert_string_to_timestamp('00:00:00.00')
    time_in_sec = (t2 - t1).total_seconds()
    return time_in_sec

def __get_duration_in_sec_using_timestamp_strings(begin_time, end_time):
    begin_time = __convert_string_to_timestamp(begin_time)
    end_time = __convert_string_to_timestamp(end_time)
    duration = end_time - begin_time
    duration = duration.total_seconds()
    return duration

def __get_duration_in_sec_using_frame_nums(frame_start, frame_stop, fps):
    duration = frame_stop - frame_start
    duration = duration / fps
    return duration

def __get_duration_in_sec_using_frame_count(start_frame, stop_frame, video_id, video_annot):
    pass

def __get_action_names_from_action_keys(action_keys):
    root_path = c.DATA_ROOT_PATH
    verb_dict_path = '%s/EPIC-Kitchens/annotations/EPIC_verb_classes_dict.pkl' % (root_path)
    noun_dict_path = '%s/EPIC-Kitchens/annotations/EPIC_noun_classes_dict.pkl' % (root_path)

    verb_dict = utils.pkl_load(verb_dict_path)
    noun_dict = utils.pkl_load(noun_dict_path)

    action_names = []
    for key in action_keys:
        verb_idx, noun_idx = key.split('_')
        verb_idx = int(verb_idx)
        noun_idx = int(noun_idx)
        verb_name = verb_dict[verb_idx][0]
        noun_name = noun_dict[noun_idx][0]
        action_name = '%s_%s' % (verb_name, noun_name)
        action_names.append(action_name)

    action_names = np.array(action_names)
    return action_names

# endregion
