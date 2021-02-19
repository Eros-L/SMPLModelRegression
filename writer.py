# -*- coding: utf-8 -*-

"""
writer.py

Created by Lau, KwanYuen on 2019/03/04.
Copyright© 2019. Lau, KwanYuen. All rights reserved.

Abstract:
    A writer class used for saving TFRecords file.
"""

import os
import tensorflow as tf
import re
import cv2 as cv
import numpy as np
import pickle


class Writer:
    def __init__(self,
                 filenames,
                 path,
                 image_size=256,
                 capacity=None,
                 tag=None):
        self.filenames = ('%s.tfrecords' % filenames)
        self.writer = None
        self.path = path
        self.image_size = image_size
        self.capacity = capacity
        self.tag = tag

    def __generate_heatmap(self, keypoint):
        gaussian_map = np.zeros(shape=(self.image_size, self.image_size, 14), dtype=np.float32)
        for x in range(self.image_size):
            for y in range(self.image_size):
                for c in range(14):
                    dist_sq = (x - keypoint[0, c]) * (x - keypoint[0, c]) + \
                              (y - keypoint[1, c]) * (y - keypoint[1, c])
                    exponent = dist_sq / 200.0 / keypoint[2, c] / keypoint[2, c]
                    gaussian_map[y, x, c] = np.exp(-exponent)
        return gaussian_map

    def __serialize(self, info_path, crop_info_path, image_path, silhouette_path, keypoint_path, mesh_path):
        # verify the tag of the current sample
        if self.tag is not None:
            with open(info_path, 'r') as info:
                if info.readline().split(' ')[0] != self.tag:
                    return
        # pre-process
        with open(crop_info_path, 'r') as crop_info:
            boundary = crop_info.readline().split(' ')
            top, bottom, left, right = [int(boundary[i]) for i in range(2, 6)]
            height, width = [bottom - top, right - left]
        # resize the original image (the input of Human2D)
        image = cv.imread(image_path)[top:bottom+1, left:right+1]
        image = cv.cvtColor(cv.resize(image, dsize=(self.image_size, self.image_size)), cv.COLOR_BGR2RGB)
        # convert the body segmentation image into a binary mask, in which,
        # pixels with the value of 255 represents foreground
        silhouette = cv.imread(silhouette_path)[top:bottom+1, left:right+1]
        _, silhouette = cv.threshold(cv.cvtColor(silhouette, cv.COLOR_BGR2GRAY), 254, 255, cv.THRESH_BINARY_INV)
        silhouette = cv.resize(silhouette, dsize=(self.image_size, self.image_size))
        # convert the keypoint ndarray into a heatmap of 14 channels
        keypoint = np.load(keypoint_path)
        for i in range(keypoint.shape[1]):
            keypoint[0, i] = (keypoint[0, i] - left) / width * (self.image_size - 1)
            keypoint[1, i] = (keypoint[1, i] - top) / height * (self.image_size - 1)
        keypoint = self.__generate_heatmap(keypoint)
        # extract pose parameters (theta) and shape parameters (beta) from a pkl file
        mesh = pickle.load(open(mesh_path, 'rb'), encoding='iso-8859-1')
        theta = mesh['pose']
        beta = mesh['betas']
        # write
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),
            'silhouette': tf.train.Feature(bytes_list=tf.train.BytesList(value=[silhouette.tobytes()])),
            'keypoint': tf.train.Feature(bytes_list=tf.train.BytesList(value=[keypoint.tobytes()])),
            'theta': tf.train.Feature(bytes_list=tf.train.BytesList(value=[theta.tobytes()])),
            'beta': tf.train.Feature(bytes_list=tf.train.BytesList(value=[beta.tobytes()]))
        }))
        self.writer.write(example.SerializeToString())

    def dump(self):
        # initialize tfrecord writer
        if os.path.exists(self.filenames):
            flag = input('file %s already exists, do you want to remove it? [y/n]\n' % self.filenames)
            if flag == 'y':
                os.remove(self.filenames)
            elif flag == 'n':
                raise FileExistsError
            else:
                raise ValueError
        self.writer = tf.python_io.TFRecordWriter(self.filenames)
        # create tfrecord
        if self.capacity is not None:
            for index in range(0, self.capacity):
                if index % 10 == 0:
                    print('processing sample %05d' % index)
                # define sub-path
                info_path = self.path + ('%05d' % index) + '_dataset_info.txt'
                crop_info_path = self.path + ('%05d' % index) + '_fit_crop_info.txt'
                image_path = self.path + ('%05d' % index) + '_image.png'
                silhouette_path = self.path + ('%05d' % index) + '_render_light.png'
                keypoint_path = self.path + ('%05d' % index) + '_joints.npy'
                mesh_path = self.path + ('%05d' % index) + '_body.pkl'
                # serialize
                self.__serialize(info_path, crop_info_path, image_path, silhouette_path, keypoint_path, mesh_path)
            self.writer.close()
        else:
            # obtain available data automatically
            files = os.listdir(self.path)
            regulation = [r'\d+_dataset_info.txt', r'\d+_fit_crop_info.txt'
                          r'\d+_image.png', r'\d+_render_light.png', r'\d+_joints.npy', r'\d+_body.pkl']
            category = []
            for r in regulation:
                p = re.compile(r)
                target = []
                for name in files:
                    if re.match(p, name):
                        target.append(name)
                category.append(target)
            # judge the validity of data
            category[0].sort(key=lambda x: x[0:5])
            for i in range(1, len(category)):
                if len(category[i]) != len(category[i-1]):
                    raise ValueError('Incomplete sample or redundant sample exists')
                category[i].sort(key=lambda x: x[0:5])
            # create tfrecord
            for index in range(len(category[0])):
                for i in range(1, len(category)):
                    if category[i][index][0:5] != category[i-1][index][0:5]:
                        raise ValueError('Mismatched sample exists at index %d' % index)
                info_path = self.path + category[0][index]
                crop_info_path = self.path + category[1][index]
                image_path = self.path + category[2][index]
                silhouette_path = self.path + category[3][index]
                keypoint_path = self.path + category[4][index]
                mesh_path = self.path + category[5][index]
                # serialize
                self.__serialize(info_path, crop_info_path, image_path, silhouette_path, keypoint_path, mesh_path)
            self.writer.close()


def __test():
    tags = ['lsp', 'lspext', 'mpii', 'fashionpose']
    for tag in tags:
        print('Processing %s dataset: ' % tag)
        writer = Writer(filenames=tag, path='/home/laukyuen3/up-3d/', capacity=8515, tag=tag)
        writer.dump()


if __name__ == '__main__':
    # __test()
    # writer = Writer(filenames='mini', path='/home/laukyuen3/up-3d/', capacity=3016, tag='mpii')
    # writer.dump()
    path = '/home/laukyuen3/up-3d/'
    for index in range(2615, 3016):
        # define sub-path
        mesh_path = path + ('%05d' % index) + '_body.pkl'
        mesh = pickle.load(open(mesh_path, 'rb'), encoding='iso-8859-1')
        theta = mesh['pose']
        print(theta)
        pass
