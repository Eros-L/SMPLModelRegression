# -*- coding: utf-8 -*-

"""
reader.py

Created by Lau, KwanYuen on 2019/03/02.
Copyright© 2019. Lau, KwanYuen. All rights reserved.

Abstract:
    A reader class used for loading TFRecords file.
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class Reader:
    def __init__(self,
                 filenames,
                 image_size=256,
                 batch_size=4,
                 num_epochs=1):
        self.dataset = tf.data.TFRecordDataset(filenames=filenames)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def __parser(self, serialized_example):
        # define feature parser
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'silhouette': tf.FixedLenFeature([], tf.string),
                'keypoint': tf.FixedLenFeature([], tf.string),
                'theta': tf.FixedLenFeature([], tf.string),
                'beta': tf.FixedLenFeature([], tf.string)
            }
        )
        # post-process for the input of Human2D
        image = tf.decode_raw(features['image'], tf.uint8)
        image = tf.reshape(image, [self.image_size, self.image_size, 3])
        image = tf.cast(image, tf.float32)
        # resize silhouette and keypoint (to make their resolutions same as the resolution of Human2D's output)
        silhouette = tf.decode_raw(features['silhouette'], tf.uint8)
        silhouette = tf.reshape(silhouette, [self.image_size, self.image_size, 1])
        silhouette = tf.image.resize_images(silhouette, size=(64, 64))
        keypoint = tf.decode_raw(features['keypoint'], tf.float32)
        keypoint = tf.reshape(keypoint, [self.image_size, self.image_size, 14])
        keypoint = tf.image.resize_images(keypoint, size=(64, 64))
        # post-process for the groundtruth of ShapePrior and PosePrior
        theta = tf.decode_raw(features['theta'], tf.float64)
        beta = tf.decode_raw(features['beta'], tf.float64)

        return image, silhouette, keypoint, theta, beta

    def feed(self, shuffle=True):
        dataset = self.dataset.map(self.__parser)
        if shuffle:
            dataset = dataset.repeat(self.num_epochs).shuffle(buffer_size=500).batch(self.batch_size)
        else:
            dataset = dataset.repeat(self.num_epochs).batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()


def __test():
    init = tf.global_variables_initializer()

    reader = Reader('fashionpose.tfrecords')
    data = reader.feed()

    with tf.Session() as sess:
        sess.run(init)
        img, sil, kpt, t, b = sess.run(data)

        print(img.shape)
        print(sil.shape)
        print(kpt.shape)
        print(t.shape)
        print(b.shape)

        Image.fromarray(np.uint8(img[0]), mode='RGB').show()
        Image.fromarray(np.uint8(sil[0, :, :, 0]), mode='L').show()
        plt.imshow(kpt[0, :, :, 0])
        plt.show()
        print(t)
        print(b)


if __name__ == '__main__':
    __test()
