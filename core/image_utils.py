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

import cv2
import numpy as np
import random
import math
import utils
from multiprocessing.dummy import Pool

# region frame resizing

def resize_frame(image, target_height=224, target_width=224):
    return __resize_frame(image, target_height, target_width)

def resize_keep_aspect_ratio_max_dim(image, max_dim=None):
    return __resize_keep_aspect_ratio_max_dim(image, max_dim)

def resize_keep_aspect_ratio_min_dim(image, min_dim=None):
    return __resize_keep_aspect_ratio_min_dim(image, min_dim)

def resize_crop(image, target_height=224, target_width=224):
    return __resize_crop(image, target_height, target_width)

def resize_crop_scaled(image, target_height=224, target_width=224):
    return __resize_crop_scaled(image, target_height, target_width)

def resize_keep_aspect_ratio_padded(image, target_height=224, target_width=224):
    return __resize_keep_aspect_ratio_padded(image, target_height, target_width)

def __resize_frame(image, target_height=224, target_width=224):
    """
    Resize to the given dimensions. Don't care about maintaining the aspect ratio of the given image.
    """
    if len(image.shape) == 2:
        image = np.tile(image[:, :, None], 3)
    elif len(image.shape) == 4:
        image = image[:, :, :, 0]

    resized_image = cv2.resize(image, dsize=(target_height, target_width))
    return resized_image

def __resize_keep_aspect_ratio_max_dim(image, max_dim=224):
    """
    Resize the given image while maintaining the aspect ratio.
    """
    if len(image.shape) == 2:
        image = np.tile(image[:, :, None], 3)
    elif len(image.shape) == 4:
        image = image[:, :, :, 0]

    height = image.shape[0]
    width = image.shape[1]

    if height > width:
        target_height = max_dim
        target_width = int(target_height * width / float(height))
    else:
        target_width = max_dim
        target_height = int(target_width * height / float(width))

    resized_image = cv2.resize(image, dsize=(target_width, target_height))
    return resized_image

def __resize_keep_aspect_ratio_min_dim(image, min_dim=224):
    """
    Resize the given image while maintaining the aspect ratio.
    """
    if len(image.shape) == 2:
        image = np.tile(image[:, :, None], 3)
    elif len(image.shape) == 4:
        image = image[:, :, :, 0]

    height = image.shape[0]
    width = image.shape[1]

    if height > width:
        target_width = min_dim
        target_height = int(target_width * height / float(width))
    else:
        target_height = min_dim
        target_width = int(target_height * width / float(height))

    resized_image = cv2.resize(image, dsize=(target_width, target_height))
    return resized_image

def __resize_crop(image, target_height=224, target_width=224):
    if len(image.shape) == 2:
        image = np.tile(image[:, :, None], 3)
    elif len(image.shape) == 4:
        image = image[:, :, :, 0]

    height, width, rgb = image.shape
    if width == height:
        resized_image = cv2.resize(image, (target_height, target_width))

    elif height < width:
        resized_image = cv2.resize(image, (int(width * float(target_height) / height), target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:, cropping_length:resized_image.shape[1] - cropping_length]

    else:
        resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length, :]

    resized_image = cv2.resize(resized_image, (target_height, target_width))
    return resized_image

def __resize_crop_scaled(image, target_height=224, target_width=224):
    # re-scale the image by ratio 3/4 so a landscape or portrait image becomes square
    # then resize_crop it

    # for example, if input image is (height*width) is 400*1000 it will be (400 * 1000 * 3/4) = 400 * 750

    if len(image.shape) == 2:
        image = np.tile(image[:, :, None], 3)
    elif len(image.shape) == 4:
        image = image[:, :, :, 0]

    height, width, _ = image.shape
    if width == height:
        resized_image = cv2.resize(image, (target_height, target_width))
    else:

        # first, rescale it, only if the rescale won't bring the scaled dimention to lower than target_dim (= 224)
        scale_factor = 3 / 4.0
        if height < width:
            new_width = int(width * scale_factor)
            if new_width >= target_width:
                image = cv2.resize(image, (new_width, height))
        else:
            new_height = int(height * scale_factor)
            if new_height >= target_height:
                image = cv2.resize(image, (width, new_height))

        # now, resize and crop
        height, width, _ = image.shape
        if height < width:
            resized_image = cv2.resize(image, (int(width * float(target_height) / height), target_width))
            cropping_length = int((resized_image.shape[1] - target_height) / 2)
            resized_image = resized_image[:, cropping_length:resized_image.shape[1] - cropping_length]

        else:
            resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)))
            cropping_length = int((resized_image.shape[0] - target_width) / 2)
            resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length, :]

        # this line is important, because sometimes the cropping there is a 1 pixel more
        height, width, _ = resized_image.shape
        if height > target_height or width > target_width:
            resized_image = cv2.resize(resized_image, (target_height, target_width))

    return resized_image

def __resize_keep_aspect_ratio_padded(image, target_height=224, target_width=224):
    """
    Resize the frame while keeping aspect ratio. Also, to result in an image with the given dimensions, the resized image is zero-padded.
    """

    if len(image.shape) == 2:
        image = np.tile(image[:, :, None], 3)
    elif len(image.shape) == 4:
        image = image[:, :, :, 0]

    original_height, original_width, _ = image.shape
    original_aspect_ratio = original_height / float(original_width)
    target_aspect_ratio = target_height / float(target_width)

    if target_aspect_ratio >= original_aspect_ratio:
        if original_width >= original_height:
            max_dim = target_width
        else:
            max_dim = int(original_height * target_width / float(original_width))
    else:
        if original_height >= original_width:
            max_dim = target_height
        else:
            max_dim = int(original_width * target_height / float(original_height))

    image = __resize_keep_aspect_ratio_max_dim(image, max_dim=max_dim)

    new_height, new_width, _ = image.shape
    new_aspect_ratio = new_height / float(new_width)

    # do zero-padding for the image (vertical or horizontal)
    img_padded = np.zeros((target_height, target_width, 3), dtype=image.dtype)

    if target_aspect_ratio < new_aspect_ratio:
        # horizontal padding
        y1 = 0
        y2 = new_height
        x1 = int((target_width - new_width) / 2.0)
        x2 = x1 + new_width
    else:
        # vertical padding
        x1 = 0
        x2 = new_width
        y1 = int((target_height - new_height) / 2.0)
        y2 = y1 + new_height

    img_padded[y1:y2, x1:x2, :] = image
    return img_padded

# endregion

# region Image Reader ResNet-152 Keras

class AsyncImageReaderResNet152Keras():
    def __init__(self, bgr_mean, n_threads=20):
        random.seed(101)
        np.random.seed(101)

        self.__is_busy = False
        self.__images = None
        self.__n_channels = 3
        self.__img_dim = 224
        self.__bgr_mean = bgr_mean

        self.__n_threads_in_pool = n_threads
        self.__pool = Pool(self.__n_threads_in_pool)

    def load_imgs_in_batch(self, image_pathes):
        self.__is_busy = True

        n_pathes = len(image_pathes)
        idxces = np.arange(0, n_pathes)

        # parameters passed to the reading function
        params = [data_item for data_item in zip(idxces, image_pathes)]

        # set list of images before start reading
        imgs_shape = (n_pathes, self.__img_dim, self.__img_dim, self.__n_channels)
        self.__images = np.zeros(imgs_shape, dtype=np.float32)

        # start pool of threads
        self.__pool.map_async(self.__preprocess_img_wrapper, params, callback=self.__thread_pool_callback)

    def get_images(self):
        if self.__is_busy:
            raise Exception('Sorry, you can\'t get images while threads are running!')
        else:
            return self.__images

    def is_busy(self):
        return self.__is_busy

    def __thread_pool_callback(self, args):
        self.__is_busy = False

    def __preprocess_img_wrapper(self, params):
        try:
            self.__preprocess_img(params)
        except Exception as exp:
            print ('Error in __preprocess_img')
            print (exp)

    def __preprocess_img(self, params):

        idx = params[0]
        path = params[1]

        img = cv2.imread(path)
        img = img.astype(np.float32)

        # subtract mean pixel from image
        img[:, :, 0] -= self.__bgr_mean[0]
        img[:, :, 1] -= self.__bgr_mean[1]
        img[:, :, 2] -= self.__bgr_mean[2]

        # convert from bgr to rgb
        img = img[:, :, (2, 1, 0)]

        self.__images[idx] = img

    def close(self):
        self.__pool.close()
        self.__pool.terminate()

# endregion

# region Image/Video Readers Breakfast

class AsyncImageReaderBreakfastForI3DKerasModel():
    def __init__(self, n_threads=20):
        random.seed(101)
        np.random.seed(101)

        self.__is_busy = False
        self.__images = None
        self.__n_channels = 3
        self.__img_dim = 224

        self.__n_threads_in_pool = n_threads
        self.__pool = Pool(self.__n_threads_in_pool)

    def load_imgs_in_batch(self, image_pathes):
        self.__is_busy = True

        n_pathes = len(image_pathes)
        idxces = np.arange(0, n_pathes)

        # parameters passed to the reading function
        params = [data_item for data_item in zip(idxces, image_pathes)]

        # set list of images before start reading
        imgs_shape = (n_pathes, self.__img_dim, self.__img_dim, self.__n_channels)
        self.__images = np.zeros(imgs_shape, dtype=np.float32)

        # start pool of threads
        self.__pool.map_async(self.__preprocess_img_wrapper, params, callback=self.__thread_pool_callback)

    def get_images(self):
        if self.__is_busy:
            raise Exception('Sorry, you can\'t get images while threads are running!')
        else:
            return self.__images

    def is_busy(self):
        return self.__is_busy

    def __thread_pool_callback(self, args):
        self.__is_busy = False

    def __preprocess_img_wrapper(self, params):
        try:
            self.__preprocess_img(params)
        except Exception as exp:
            print ('Error in __preprocess_img')
            print (exp)

    def __preprocess_img(self, params):

        idx = params[0]
        path = params[1]

        img = cv2.imread(path)
        img = img.astype(np.float32)
        # normalize such that values range from -1 to 1
        img /= float(127.5)
        img -= 1.0
        # convert from bgr to rgb
        img = img[:, :, (2, 1, 0)]

        self.__images[idx] = img

    def close(self):
        self.__pool.close()
        self.__pool.terminate()

# endregion

# region Image/Video Readers Epic-Kitchens

class AsyncImageProcessorEpicKitchens():
    def __init__(self, n_threads=20):
        random.seed(101)
        np.random.seed(101)

        self.__is_busy = False
        self.__images = None
        self.__n_channels = 3

        self.__n_threads_in_pool = n_threads
        self.__pool = Pool(self.__n_threads_in_pool)

    def process_images(self, src_root_path, dst_root_path, image_names):
        self.__is_busy = True
        self.__src_root_path = src_root_path
        self.__dst_root_path = dst_root_path

        # start pool of threads
        self.__pool.map_async(self.__preprocess_img_wrapper, image_names, callback=self.__thread_pool_callback)


    def is_busy(self):
        return self.__is_busy

    def __thread_pool_callback(self, args):
        self.__is_busy = False

    def __preprocess_img_wrapper(self, image_name):
        try:
            self.__preprocess_img(image_name)
        except Exception as exp:
            print ('Error in __preprocess_img')
            print (exp)

    def __preprocess_img(self, image_name):

        src_path = '%s/%s' % (self.__src_root_path, image_name)
        dst_path = '%s/%s' % (self.__dst_root_path, image_name)

        img = cv2.imread(src_path)
        img = resize_crop(img)
        cv2.imwrite(dst_path, img)

    def close(self):
        self.__pool.close()
        self.__pool.terminate()

class AsyncImageReaderEpicKitchensForI3dKerasModel():
    def __init__(self, n_threads=16):
        random.seed(101)
        np.random.seed(101)

        self.__is_busy = False
        self.__images = None
        self.__n_channels = 3
        self.__img_dim = 224

        self.__n_threads_in_pool = n_threads
        self.__pool = Pool(self.__n_threads_in_pool)

    def load_imgs_in_batch(self, image_pathes):
        self.__is_busy = True

        n_pathes = len(image_pathes)
        idxces = np.arange(0, n_pathes)

        # parameters passed to the reading function
        params = [data_item for data_item in zip(idxces, image_pathes)]

        # set list of images before start reading
        imgs_shape = (n_pathes, self.__img_dim, self.__img_dim, self.__n_channels)
        self.__images = np.zeros(imgs_shape, dtype=np.float32)

        # start pool of threads
        self.__pool.map_async(self.__preprocess_img_wrapper, params, callback=self.__thread_pool_callback)

    def get_images(self):
        if self.__is_busy:
            raise Exception('Sorry, you can\'t get images while threads are running!')
        else:
            return self.__images

    def is_busy(self):
        return self.__is_busy

    def __thread_pool_callback(self, args):
        self.__is_busy = False

    def __preprocess_img_wrapper(self, params):
        try:
            self.__preprocess_img(params)
        except Exception as exp:
            print ('Error in __preprocess_img')
            print (exp)

    def __preprocess_img(self, params):

        idx = params[0]
        path = params[1]

        img = cv2.imread(path)
        img = img.astype(np.float32)
        # normalize such that values range from -1 to 1
        img /= float(127.5)
        img -= 1.0
        # convert from bgr to rgb
        img = img[:, :, (2, 1, 0)]

        self.__images[idx] = img

    def close(self):
        self.__pool.close()
        self.__pool.terminate()

# endregion

# region Image Readers Others

class AsyncImageReaderPlacesOld():
    def __init__(self, batch_size, image_pathes, vgg_mean, is_training):
        random.seed(101)
        np.random.seed(101)

        self.__batch_size = batch_size
        self.__image_pathes = image_pathes
        self.__is_busy = False
        self.__images = None

        # if training, take random 224*224 crop from the image
        # if testing, take 5 crops
        self.__is_training = is_training
        self.__n_crops_per_test_img = 10

        self.__img_dim = 224
        self.__img_dim_resized = 256
        self.__n_channels = 3

        self.__n_threads_in_pool = 8
        self.__pool = Pool(self.__n_threads_in_pool)

        self.__mean = vgg_mean

    def set_image_pathes(self, image_pathes):
        self.__image_pathes = image_pathes

    def load_imgs_in_batch(self, batch_num):
        self.__is_busy = True
        n_pathes = len(self.__image_pathes)

        # this is to take into consideration if the total number of pathes is not divisable by batch_size
        idx_start = (batch_num - 1) * self.__batch_size
        idx_stop = (batch_num) * self.__batch_size
        idx_stop = n_pathes if idx_stop > n_pathes else idx_stop

        pathes = self.__image_pathes[idx_start:idx_stop]
        n_pathes_batch = len(pathes)
        idxces = np.arange(0, n_pathes_batch)
        random_flips = self.__get_random_flips(n_pathes_batch)

        # parameters passed to the reading function
        params = [data_item for data_item in zip(idxces, pathes, random_flips)]

        # set list of images before start reading
        n_imgs_batch = n_pathes_batch if self.__is_training else n_pathes_batch * self.__n_crops_per_test_img
        imgs_shape = (n_imgs_batch, self.__img_dim, self.__img_dim, self.__n_channels)
        self.__images = np.zeros(imgs_shape, dtype=np.float32)

        # start pool of threads
        self.__pool.map_async(self.__preprocess_img, params, callback=self.__thread_pool_callback)

    def get_images(self):
        if self.__is_busy:
            raise Exception('Sorry, you can\'t get images while threads are running!')
        else:
            return self.__images

    def is_busy(self):
        return self.__is_busy

    def __thread_pool_callback(self, args):
        self.__is_busy = False

    def __preprocess_img(self, params):
        # if training, random crop, if testing, the 5 crops
        try:
            if self.__is_training:
                self.__preprocess_img_training(params)
            else:
                self.__preprocess_img_testing(params)
        except Exception as exp:
            print ('')

    def __preprocess_img_training(self, params):

        idx = params[0]
        path = params[1]
        flip = params[2]

        # load image
        img = cv2.imread(path)

        # resize
        img = self.__random_sized_crop_as_in_pytorch(img)

        # flip
        img = cv2.flip(img, 1) if flip else img

        # as float
        img = img.astype(np.float32)

        # subtract mean
        img[:, :, 0] -= self.__mean[0]
        img[:, :, 1] -= self.__mean[1]
        img[:, :, 2] -= self.__mean[2]

        self.__images[idx] = img

    def __preprocess_img_testing(self, params):

        img_dim = self.__img_dim

        idx = params[0]
        path = params[1]

        img = cv2.imread(path)
        h, w, _ = img.shape

        # resize before crop
        if h < img_dim or w < img_dim:
            img = resize_keep_aspect_ratio_min_dim(img, img_dim)
        else:
            img = resize_keep_aspect_ratio_min_dim(img, self.__img_dim_resized)

        h, w, _ = img.shape
        idx1 = idx * self.__n_crops_per_test_img
        idx2 = idx1 + 1
        idx3 = idx2 + 1
        idx4 = idx3 + 1
        idx5 = idx4 + 1

        img = img.astype(np.float32)
        img[:, :, 0] -= self.__mean[0]
        img[:, :, 1] -= self.__mean[1]
        img[:, :, 2] -= self.__mean[2]

        # crops are top-left, top-right, bottom-left, bottom-right, center
        crop_x1 = w - img_dim
        crop_y1 = h - img_dim
        crop5_x1 = int((w - img_dim) / 2.0)
        crop5_y1 = int((h - img_dim) / 2.0)

        # crop 1
        x1, y1, x2, y2 = 0, 0, img_dim, img_dim
        img_cropped = img[y1:y2, x1:x2]
        self.__images[idx1] = img_cropped
        self.__images[idx1 + 5] = cv2.flip(img_cropped, 1)

        # crop 2
        x1, y1, x2, y2 = crop_x1, 0, w, img_dim
        img_cropped = img[y1:y2, x1:x2]
        self.__images[idx2] = img_cropped
        self.__images[idx2 + 5] = cv2.flip(img_cropped, 1)

        # crop 3
        x1, y1, x2, y2 = 0, crop_y1, img_dim, h
        img_cropped = img[y1:y2, x1:x2]
        self.__images[idx3] = img_cropped
        self.__images[idx3 + 5] = cv2.flip(img_cropped, 1)

        # crop 4
        x1, y1, x2, y2 = crop_x1, crop_y1, w, h
        img_cropped = img[y1:y2, x1:x2]
        self.__images[idx4] = img_cropped
        self.__images[idx4 + 5] = cv2.flip(img_cropped, 1)

        # crop 5
        x1, y1, x2, y2 = crop5_x1, crop5_y1, crop5_x1 + img_dim, crop5_y1 + img_dim
        img_cropped = img[y1:y2, x1:x2]
        self.__images[idx5] = img_cropped
        self.__images[idx5 + 5] = cv2.flip(img_cropped, 1)

    def __get_random_flips(self, n_imgs):
        idxes = np.random.randint(low=0, high=2, size=(n_imgs,))
        flips = [i == 0 for i in idxes]
        return flips

    def __zero_pad(self, img):
        h, w, _ = img.shape
        img_dim = self.__img_dim

        padded_w = img_dim if w < img_dim else w
        padded_h = img_dim if h < img_dim else h
        padded = np.zeros((padded_h, padded_w, 3), dtype=img.dtype)

        offset_x = int((img_dim - w) / 2.0) if w < img_dim else 0
        offset_y = int((img_dim - h) / 2.0) if h < img_dim else 0

        x1 = offset_x
        x2 = offset_x + w

        y1 = offset_y
        y2 = offset_y + h

        padded[y1:y2, x1:x2] = img

        return padded

    def __random_sized_crop_as_in_pytorch(self, img):

        img_h, img_w, _ = img.shape

        for attempt in range(10):
            area = img_w * img_h
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img_w and h <= img_h:
                x1 = random.randint(0, img_w - w)
                y1 = random.randint(0, img_h - h)
                x2 = x1 + w
                y2 = y1 + h

                img = img[y1:y2, x1:x2]
                img_h_new, img_w_new, _ = img.shape

                assert ((img_w_new, img_h_new) == (w, h))

                img = cv2.resize(img, (self.__img_dim, self.__img_dim), interpolation=cv2.INTER_LINEAR)
                return img

        # Fallback
        img = resize_crop(img)
        return img

    def close(self):
        self.__pool.close()
        self.__pool.terminate()

class AsyncImageReaderPlaces():
    def __init__(self, vgg_mean, is_training, is_multi_test_crops):
        random.seed(101)
        np.random.seed(101)

        self.__is_busy = False
        self.__images = None
        self.__is_training = is_training
        self.__is_multi_test_crops = is_multi_test_crops
        self.__n_crops_per_test_img = 10 if is_multi_test_crops else 1

        self.__n_channels = 3
        self.__img_dim = 224
        self.__img_dim_resized = 256

        self.__mean = vgg_mean
        self.image_pathes = None

        self.__n_threads_in_pool = 8
        self.__pool = Pool(self.__n_threads_in_pool)

    def load_imgs_in_batch(self, image_pathes):
        self.__is_busy = True
        self.image_pathes = image_pathes
        n_pathes = len(image_pathes)

        idxces = np.arange(0, n_pathes)

        random_flips = self.__get_random_flips(n_pathes)

        # parameters passed to the reading function
        params = [data_item for data_item in zip(idxces, image_pathes, random_flips)]

        # set list of images before start reading
        n_imgs = n_pathes if self.__is_training else n_pathes * self.__n_crops_per_test_img
        imgs_shape = (n_imgs, self.__img_dim, self.__img_dim, self.__n_channels)
        self.__images = np.zeros(imgs_shape, dtype=np.float32)

        # start pool of threads
        self.__pool.map_async(self.__preprocess_img, params, callback=self.__thread_pool_callback)

    def get_images(self):
        if self.__is_busy:
            raise Exception('Sorry, you can\'t get images while threads are running!')
        else:
            return self.__images

    def is_busy(self):
        return self.__is_busy

    def __thread_pool_callback(self, args):
        self.__is_busy = False

    def __preprocess_img(self, params):
        try:
            if self.__is_training:
                self.__preprocess_img_training(params)
            else:
                if self.__is_multi_test_crops:
                    self.__preprocess_img_testing_ten_crops(params)
                else:
                    self.__preprocess_img_testing_one_crop(params)
        except Exception as exp:
            _ = 10

    def __preprocess_img_training(self, params):

        idx = params[0]
        path = params[1]
        flip = params[2]

        # load image
        img = cv2.imread(path)

        # resize
        img = self.__random_sized_crop_as_in_pytorch(img)

        # flip
        img = cv2.flip(img, 1) if flip else img

        # as float
        img = img.astype(np.float32)

        # subtract mean
        img[:, :, 0] -= self.__mean[0]
        img[:, :, 1] -= self.__mean[1]
        img[:, :, 2] -= self.__mean[2]

        self.__images[idx] = img

    def __preprocess_img_testing_ten_crops(self, params):
        img_dim = self.__img_dim

        idx = params[0]
        path = params[1]

        img = cv2.imread(path)
        h, w, _ = img.shape

        # resize before crop
        if h < img_dim or w < img_dim:
            img = resize_keep_aspect_ratio_min_dim(img, img_dim)
        else:
            img = resize_keep_aspect_ratio_min_dim(img, self.__img_dim_resized)

        h, w, _ = img.shape
        idx1 = idx * self.__n_crops_per_test_img
        idx2 = idx1 + 1
        idx3 = idx2 + 1
        idx4 = idx3 + 1
        idx5 = idx4 + 1

        img = img.astype(np.float32)
        img[:, :, 0] -= self.__mean[0]
        img[:, :, 1] -= self.__mean[1]
        img[:, :, 2] -= self.__mean[2]

        # crops are top-left, top-right, bottom-left, bottom-right, center
        crop_x1 = w - img_dim
        crop_y1 = h - img_dim
        crop5_x1 = int((w - img_dim) / 2.0)
        crop5_y1 = int((h - img_dim) / 2.0)

        # crop 1
        x1, y1, x2, y2 = 0, 0, img_dim, img_dim
        img_cropped = img[y1:y2, x1:x2]
        self.__images[idx1] = img_cropped
        self.__images[idx1 + 5] = cv2.flip(img_cropped, 1)

        # crop 2
        x1, y1, x2, y2 = crop_x1, 0, w, img_dim
        img_cropped = img[y1:y2, x1:x2]
        self.__images[idx2] = img_cropped
        self.__images[idx2 + 5] = cv2.flip(img_cropped, 1)

        # crop 3
        x1, y1, x2, y2 = 0, crop_y1, img_dim, h
        img_cropped = img[y1:y2, x1:x2]
        self.__images[idx3] = img_cropped
        self.__images[idx3 + 5] = cv2.flip(img_cropped, 1)

        # crop 4
        x1, y1, x2, y2 = crop_x1, crop_y1, w, h
        img_cropped = img[y1:y2, x1:x2]
        self.__images[idx4] = img_cropped
        self.__images[idx4 + 5] = cv2.flip(img_cropped, 1)

        # crop 5
        x1, y1, x2, y2 = crop5_x1, crop5_y1, crop5_x1 + img_dim, crop5_y1 + img_dim
        img_cropped = img[y1:y2, x1:x2]
        self.__images[idx5] = img_cropped
        self.__images[idx5 + 5] = cv2.flip(img_cropped, 1)

    def __preprocess_img_testing_one_crop(self, params):
        img_dim = self.__img_dim

        idx = params[0]
        path = params[1]

        img = cv2.imread(path)
        h, w, _ = img.shape

        # resize before crop
        if h < img_dim or w < img_dim:
            img = resize_keep_aspect_ratio_min_dim(img, img_dim)
        else:
            img = resize_keep_aspect_ratio_min_dim(img, self.__img_dim_resized)

        # enter crop
        img = resize_crop(img)

        # as float
        img = img.astype(np.float32)

        # subtract mean
        img[:, :, 0] -= self.__mean[0]
        img[:, :, 1] -= self.__mean[1]
        img[:, :, 2] -= self.__mean[2]

        self.__images[idx] = img

    def __random_sized_crop_as_in_pytorch(self, img):

        img_h, img_w, _ = img.shape

        for attempt in range(10):
            area = img_w * img_h
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img_w and h <= img_h:
                x1 = random.randint(0, img_w - w)
                y1 = random.randint(0, img_h - h)
                x2 = x1 + w
                y2 = y1 + h

                img = img[y1:y2, x1:x2]
                img_h_new, img_w_new, _ = img.shape

                assert ((img_w_new, img_h_new) == (w, h))

                img = cv2.resize(img, (self.__img_dim, self.__img_dim), interpolation=cv2.INTER_LINEAR)
                return img

        # Fallback
        img = resize_crop(img)
        return img

    def __get_random_flips(self, n_imgs):
        idxes = np.random.randint(low=0, high=2, size=(n_imgs,))
        flips = [i == 0 for i in idxes]
        return flips

    def close(self):
        self.__pool.close()
        self.__pool.terminate()

class AsyncImageReaderSimple():
    def __init__(self, vgg_mean, is_training):
        random.seed(101)
        np.random.seed(101)

        self.__is_busy = False
        self.__images = None
        self.__is_training = is_training

        self.__n_channels = 3
        self.__img_dim = 224
        self.__img_dim_resized = 256

        self.__mean = vgg_mean
        self.image_pathes = None

        self.__n_threads_in_pool = 8
        self.__pool = Pool(self.__n_threads_in_pool)

    def load_imgs_in_batch(self, image_pathes):
        self.__is_busy = True
        self.image_pathes = image_pathes
        n_pathes = len(image_pathes)

        idxces = np.arange(0, n_pathes)

        random_flips = self.__get_random_flips(n_pathes)

        # parameters passed to the reading function
        params = [data_item for data_item in zip(idxces, image_pathes, random_flips)]

        # set list of images before start reading
        imgs_shape = (n_pathes, self.__img_dim, self.__img_dim, self.__n_channels)
        self.__images = np.zeros(imgs_shape, dtype=np.float32)

        # start pool of threads
        self.__pool.map_async(self.__preprocess_img, params, callback=self.__thread_pool_callback)

    def get_images(self):
        if self.__is_busy:
            raise Exception('Sorry, you can\'t get images while threads are running!')
        else:
            return self.__images

    def is_busy(self):
        return self.__is_busy

    def __thread_pool_callback(self, args):
        self.__is_busy = False

    def __preprocess_img(self, params):
        # if training, random crop, if testing, the 5 crops
        try:
            if self.__is_training:
                self.__preprocess_img_training(params)
            else:
                self.__preprocess_img_testing(params)
        except Exception as exp:
            print ('')

    def __preprocess_img_training(self, params):

        idx = params[0]
        path = params[1]
        flip = params[2]

        # load image
        img = cv2.imread(path)

        # resize
        img = self.__random_sized_crop_as_in_pytorch(img)

        # flip
        img = cv2.flip(img, 1) if flip else img

        # as float
        img = img.astype(np.float32)

        # subtract mean
        img[:, :, 0] -= self.__mean[0]
        img[:, :, 1] -= self.__mean[1]
        img[:, :, 2] -= self.__mean[2]

        self.__images[idx] = img

    def __preprocess_img_testing(self, params):
        img_dim = self.__img_dim

        idx = params[0]
        path = params[1]

        img = cv2.imread(path)
        h, w, _ = img.shape

        # resize before crop
        if h < img_dim or w < img_dim:
            img = resize_keep_aspect_ratio_min_dim(img, img_dim)
        else:
            img = resize_keep_aspect_ratio_min_dim(img, self.__img_dim_resized)

        # enter crop
        img = resize_crop(img)

        # as float
        img = img.astype(np.float32)

        # subtract mean
        img[:, :, 0] -= self.__mean[0]
        img[:, :, 1] -= self.__mean[1]
        img[:, :, 2] -= self.__mean[2]

        self.__images[idx] = img

    def __preprocess_img_simple(self, params):

        idx = params[0]
        path = params[1]

        # load image
        img = cv2.imread(path)

        # resize
        img = resize_crop(img)

        # as float
        img = img.astype(np.float32)

        # subtract mean
        img[:, :, 0] -= self.__mean[0]
        img[:, :, 1] -= self.__mean[1]
        img[:, :, 2] -= self.__mean[2]

        self.__images[idx] = img

    def __random_sized_crop_as_in_pytorch(self, img):

        img_h, img_w, _ = img.shape

        for attempt in range(10):
            area = img_w * img_h
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img_w and h <= img_h:
                x1 = random.randint(0, img_w - w)
                y1 = random.randint(0, img_h - h)
                x2 = x1 + w
                y2 = y1 + h

                img = img[y1:y2, x1:x2]
                img_h_new, img_w_new, _ = img.shape

                assert ((img_w_new, img_h_new) == (w, h))

                img = cv2.resize(img, (self.__img_dim, self.__img_dim), interpolation=cv2.INTER_LINEAR)
                return img

        # Fallback
        img = resize_crop(img)
        return img

    def __get_random_flips(self, n_imgs):
        idxes = np.random.randint(low=0, high=2, size=(n_imgs,))
        flips = [i == 0 for i in idxes]
        return flips

    def close(self):
        self.__pool.close()
        self.__pool.terminate()

# endregion
