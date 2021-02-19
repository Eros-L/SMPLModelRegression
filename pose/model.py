# -*- coding: utf-8 -*-

"""
pose.model.py

Created by Lau, KwanYuen on 2019/03/02.
Copyright© 2019. Lau, KwanYuen. All rights reserved.

Abstract:
    A tensorflow implementation of PosePrior network proposed by Georgios Pavlakos et al.
    PosePrior network estimates 3D pose from 2D keypoints.

    @inproceedings {
        pavlakos2018humanshape,
        Author = {Pavlakos, Georgios and Zhu, Luyang and Zhou, Xiaowei and Daniilidis, Kostas},
        Title = {Learning to Estimate 3{D} Human Pose and Shape from a Single Color Image},
        Booktitle = {CVPR},
        Year = {2018}
    }
"""

import tensorflow as tf
from reader import Reader
from utils import fc_layer, bilinear_layer
from smpl.smpl_webuser.serialization import load_model
import time
import numpy as np
import cv2 as cv
import os


class PosePrior:
    """ Base class for building the PosePrior Model. """
    model_template = load_model('../smpl/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')

    def __init__(self,
                 filenames='',
                 image_size=256,
                 batch_size=256,
                 num_epochs=None,
                 iterations=1e5,
                 learning_rate=3e-4):
        """ Creates a model for estimating an image.

            Args:
                filenames: The filename of the dataset
                image_size: The size of the input images
                batch_size: The number of examples in a single batch
                num_epochs: The maximum number of times the model can iterate over the entire dataset
                iterations: The maximum number of iterations
                learning_rate: The learning rate of the optimizer
        """
        with tf.variable_scope(name_or_scope='pose_prior_init'):
            self.filenames = filenames
            self.image_size = image_size
            self.batch_size = batch_size
            self.num_epochs = num_epochs
            self.iterations = int(iterations)
            self.learning_rate = learning_rate
            self.training = tf.placeholder(tf.bool, name='training')
            self._input = tf.placeholder(tf.float32, [None, 3, 14], name='input')
            self._theta = tf.placeholder(tf.float32, [None, 72], name='theta')

    def model(self):
        """ Generates a model

            Returns:
                theta_output: The estimation of theta parameters of SMPL model
        """
        with tf.variable_scope(name_or_scope='pose_prior_network'):
            fc_0 = fc_layer(tf.layers.flatten(self._input[:, 0:2]), 1024, training=self.training, name='fc_0')

            bi_0 = bilinear_layer(fc_0, 1024, training=self.training, name='bi_0')
            bi_1 = bilinear_layer(bi_0, 1024, training=self.training, name='bi_1')

            theta_output = tf.layers.dense(bi_1, units=72, activation=None, use_bias=False, name='theta_output')

            return theta_output

    def mini_model(self):
        with tf.variable_scope(name_or_scope='pose_prior_network'):
            conv_0 = tf.layers.conv1d(tf.transpose(self._input[:, 0:2], perm=[0, 2, 1]), filters=4, kernel_size=[2],
                                      strides=1, padding='SAME')
            conv_1 = tf.layers.conv1d(conv_0, filters=1, kernel_size=[1], strides=1, padding='SAME')
            fc_0 = fc_layer(tf.layers.flatten(conv_1), 1024, training=self.training, dropout=False, name='fc_0')
            fc_1 = fc_layer(fc_0, 512, training=self.training, dropout=False, name='fc_1')
            fc_2 = fc_layer(fc_1, 128, training=self.training, dropout=False, name='fc_2')
            theta_output = tf.layers.dense(fc_2, 72, activation=None, use_bias=False, name='theta_output')

            return theta_output

    def train(self):
        """ Train a PosePrior model

        """
        # generate model
        theta = self.model()
        theta_summary = tf.summary.histogram('theta_summary', theta)

        with tf.variable_scope(name_or_scope='pose_prior_optimizer'):
            _vertex = tf.placeholder(tf.float32, [None, 6890, 3], name='vertex')
            _vertex_gt = tf.placeholder(tf.float32, [None, 6890, 3], name='vertex_gt')

            # loss function
            theta_loss = tf.reduce_mean(tf.reduce_sum(tf.square(theta[:, 3:] - self._theta[:, 3:]), axis=1),
                                        name='theta_loss')
            vertex_loss = tf.reduce_mean(tf.reduce_sum(tf.square(_vertex - _vertex_gt), axis=[1, 2]),
                                         name='vertex_loss')
            total_loss = theta_loss + 100 * vertex_loss
            loss_summary = tf.summary.scalar('loss_summary', total_loss)
            # optimizer
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_ops = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(total_loss)

        # configuration
        if not os.path.isdir('model'):
            os.mkdir('model')
        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.5
        # config.allow_soft_placement = True
        # config.log_device_placement = False

        print('Start training PosePrior')
        with tf.Session(config=config) as sess:
            # initialization
            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

            # saver
            saver = tf.train.Saver(max_to_keep=5)

            # store the network graph for tensorboard visualization
            writer = tf.summary.FileWriter('model/network_graph', sess.graph)
            merge_op = tf.summary.merge([theta_summary, loss_summary])

            # dataset
            data = Reader(filenames=self.filenames, image_size=self.image_size,
                          batch_size=self.batch_size, num_epochs=self.num_epochs).feed(shuffle=True)

            try:
                for i in range(self.iterations):
                    _, _, _keypoint, _theta, _ = sess.run(data)
                    _keypoint = PosePrior.generate_keypoint(_keypoint)
                    _theta = _theta * 500

                    if i < 0.4 * self.iterations:
                        zero = np.zeros(shape=[self.batch_size, 6890, 3])
                        _, loss, sm = sess.run([train_ops, total_loss, merge_op],
                                               feed_dict={self.training: True,
                                                          self._input: _keypoint, self._theta: _theta,
                                                          _vertex: zero, _vertex_gt: zero})
                    else:
                        vertex, _ = PosePrior.get_mesh(
                            sess.run(theta, feed_dict={self.training: False, self._input: _keypoint}) / 500)
                        vertex_gt, _ = PosePrior.get_mesh(_theta / 500)

                        _, loss, sm = sess.run([train_ops, total_loss, merge_op],
                                               feed_dict={self.training: True,
                                                          self._input: _keypoint, self._theta: _theta,
                                                          _vertex: vertex, _vertex_gt: vertex_gt})
                    if i % 10 == 0:
                        print('iteration %d: loss = %f' % (i, loss))
                        writer.add_summary(sm, i)
                        writer.flush()
                    if i % 200 == 0:
                        predict = sess.run(theta, feed_dict={self.training: False, self._input: _keypoint})
                        PosePrior.save_mesh(predict / 500, 'visual/out.obj')
                        PosePrior.save_mesh(_theta / 500, 'visual/gt.obj')
                        print(predict[0])
                        print()
                        print(_theta[0])
                        print()
                        print(np.maximum(1, np.abs(predict[0] - _theta[0])))
                    if i % 500 == 0 and i != 0:
                        print('save at iteration %d' % i)
                        saver.save(sess, 'model/%s/pose' %
                                   (time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))))

                    # time.sleep(1)
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
                writer.close()

    @staticmethod
    def test(filenames='', to_quantify=False, to_visualize=False):
        """ Test a trained Human2D model

            Args:
                filenames: The filename of the model to use
                to_quantify: Whether to show the loss of each batch or not
                to_visualize: Whether to print the result of each batch or not
        """
        # get graph
        graph = tf.get_default_graph()
        data = Reader(filenames=filenames, batch_size=1738, num_epochs=1).feed(shuffle=False)
        # session
        with tf.Session(graph=graph) as sess:
            # restore the latest model
            file_list = os.listdir('model/')
            file_list.sort(key=lambda x: x)
            loader = tf.train.import_meta_graph('model/%s/pose.meta' % file_list[-2])
            # get input tensor
            training_tensor = graph.get_tensor_by_name('pose_prior_init/training:0')
            input_tensor = graph.get_tensor_by_name('pose_prior_init/input:0')
            vertex_tensor = graph.get_tensor_by_name('pose_prior_optimizer/vertex:0')
            vertex_gt_tensor = graph.get_tensor_by_name('pose_prior_optimizer/vertex_gt:0')
            # get output tensor
            theta_output_tensor = graph.get_tensor_by_name('pose_prior_network/theta_output/MatMul:0')
            # get loss tensor
            vertex_loss_tensor = graph.get_tensor_by_name('pose_prior_optimizer/vertex_loss:0')

            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
            loader.restore(sess, tf.train.latest_checkpoint('model/%s' % file_list[-2]))

            # output
            index = 0
            try:
                if not os.path.isdir('result'):
                    os.mkdir('result')
                while True:
                    _input, _, _keypoint, _theta, _ = sess.run(data)
                    _keypoint = PosePrior.generate_keypoint(_keypoint)

                    theta_output = sess.run(theta_output_tensor,
                                            feed_dict={training_tensor: False, input_tensor: _keypoint}) / 500

                    _vertex, _ = PosePrior.get_mesh(theta_output)
                    _vertex_gt, _ = PosePrior.get_mesh(_theta)
                    vertex_loss = sess.run(vertex_loss_tensor,
                                           feed_dict={vertex_tensor: _vertex, vertex_gt_tensor: _vertex_gt})
                    theta_loss = np.average(np.sum(np.square(_theta[:, 3:] - theta_output[:, 3:]), axis=1))

                    # optional operation
                    if to_quantify:
                        print('theta loss = %f, vertex loss = %f' % (theta_loss, vertex_loss))
                    if to_visualize:
                        cv.imwrite('result/input_%05d.jpg' % index, _input[0, :, :, ::-1])
                        PosePrior.save_mesh(theta_output, 'result/out_%05d.obj' % index)
                    index = index + 1
            except tf.errors.OutOfRangeError:
                print('Done testing -- epoch limit reached')

    @staticmethod
    def fine_tune(filenames='',
                  image_size=256,
                  batch_size=256,
                  num_epochs=None,
                  iterations=int(1e5),
                  learning_rate=3e-4):
        # get graph
        graph = tf.get_default_graph()
        data = Reader(filenames=filenames, image_size=image_size, batch_size=batch_size, num_epochs=num_epochs).feed()

        # configuration
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.allow_soft_placement = True
        config.log_device_placement = False

        # session
        with tf.Session(graph=graph, config=config) as sess:
            # restore the latest model
            file_list = os.listdir('model/')
            file_list.sort(key=lambda x: x)
            loader = tf.train.import_meta_graph('model/%s/pose.meta' % file_list[-2])

            # get input tensor
            training_tensor = graph.get_tensor_by_name('pose_prior_init/training:0')
            input_tensor = graph.get_tensor_by_name('pose_prior_init/input:0')
            theta_tensor = graph.get_tensor_by_name('pose_prior_init/theta:0')
            # get output tensor
            # theta_output_tensor = graph.get_tensor_by_name('pose_prior_network/theta_output/output/cond/Merge:0')
            theta_output_tensor = graph.get_tensor_by_name('pose_prior_network/theta_output/LeakyRelu/Maximum:0')
            # get loss tensor
            _vertex = tf.placeholder(tf.float32, [None, 6890, 3])
            _vertex_gt = tf.placeholder(tf.float32, [None, 6890, 3])
            # loss function
            theta_loss = tf.reduce_mean(tf.reduce_sum(tf.square(theta_tensor - theta_output_tensor), axis=1))
            vertex_loss = tf.reduce_mean(tf.reduce_sum(tf.square(_vertex - _vertex_gt), axis=[1, 2]))
            total_loss = theta_loss + vertex_loss

            # optimizer
            with tf.variable_scope(name_or_scope='fine_tune'):
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_ops = optimizer.minimize(total_loss)

            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
            loader.restore(sess, tf.train.latest_checkpoint('model/%s' % file_list[-2]))

            # saver
            saver = tf.train.Saver(max_to_keep=5)

            print('Start fine tuning PosePrior')
            try:
                for i in range(iterations):
                    # input
                    _, _, _keypoint, _theta, _ = sess.run(data)
                    _keypoint = PosePrior.generate_keypoint(_keypoint)

                    vertex, _ = PosePrior.get_mesh(
                        sess.run(theta_output_tensor, feed_dict={training_tensor: False, input_tensor: _keypoint}))
                    vertex_gt, _ = PosePrior.get_mesh(_theta)

                    _, t_out, t_loss, v_loss = sess.run([train_ops,
                                                         theta_output_tensor, theta_loss, vertex_loss],
                                                        feed_dict={training_tensor: True, input_tensor: _keypoint,
                                                                   theta_tensor: _theta,
                                                                   _vertex: vertex, _vertex_gt: vertex_gt})
                    if i % 10 == 0:
                        print(t_out[0], '\n')
                        print(_theta[0], '\n')
                        print(np.square(t_out[0] - _theta[0]), '\n')
                        print(np.sum(np.square(t_out[0] - _theta[0]), axis=-1))
                        print('iteration %d: theta loss = %f, joint loss = %f' % (i, t_loss, v_loss))
                    if i % 200 == 0:
                        PosePrior.save_mesh(sess.run(theta_output_tensor,
                                                     feed_dict={training_tensor: False, input_tensor: _keypoint})
                                            , 'visual/out.obj')
                        PosePrior.save_mesh(_theta, 'visual/gt.obj')
                    if i % 500 == 0 and i != 0:
                        print('save at iteration %d' % i)
                        saver.save(sess, 'model/%s/pose' %
                                   (time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))))
                    time.sleep(1)
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')

    @classmethod
    def generate_keypoint(cls, keypoint):
        """ Generates keypoints from a prediction (a 14-channel heatmap)

            Args:
                keypoint: The prediction of keypoints (with shape of [1, 64, 64, 14])

            Returns:
                keypoint: A keypoint ndarray generated from a given prediction
        """
        batch_size = keypoint.shape[0]

        joints = np.zeros([batch_size, 3, 14])
        for i in range(batch_size):
            for j in range(14):
                coords = cv.minMaxLoc(keypoint[i, :, :, j])
                joints[i, 0, j] = coords[3][0]
                joints[i, 1, j] = coords[3][1]
                joints[i, 2, j] = coords[1]
        return joints

    @classmethod
    def get_mesh(cls, theta):
        """ Generates a 3D mesh from a set of theta

            Args:
                theta: A set of theta parameters (with shape of [None, 72])

            Returns:
                vertex: The vertices of SMPL meshes
                joint: The joints of SMPL meshes
        """
        batch_size = theta.shape[0]

        vertex = np.zeros(shape=[batch_size, 6890, 3])
        joint = np.zeros(shape=[batch_size, 24, 3])

        for i in range(batch_size):
            PosePrior.model_template.pose[3:] = theta[i, 3:]
            vertex[i] = PosePrior.model_template.r
            joint[i] = PosePrior.model_template.J

        return vertex, joint

    @classmethod
    def save_mesh(cls, theta, filename=''):
        PosePrior.model_template.pose[3:] = theta[0, 3:]

        with open(filename, 'w') as fp:
            for v in PosePrior.model_template.r:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

            for f in PosePrior.model_template.f + 1:  # Faces are 1-based, not 0-based in obj files
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


if __name__ == '__main__':
    old_v = tf.logging.get_verbosity()
    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.logging.set_verbosity(old_v)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    np.set_printoptions(threshold=np.inf, suppress=True)

    # p = PosePrior(filenames='/home/laukyuen3/thesis/mpii.tfrecords', batch_size=32, learning_rate=3e-5)
    # p.train()
    # PosePrior.fine_tune(filenames='/home/laukyuen3/thesis/mpii.tfrecords', batch_size=4, learning_rate=3e-5)

    PosePrior.test(filenames='/home/laukyuen3/thesis/fashionpose.tfrecords', to_quantify=True, to_visualize=False)
