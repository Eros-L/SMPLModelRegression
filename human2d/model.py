# -*- coding: utf-8 -*-

"""
human2d.model.py

Created by Lau, KwanYuen on 2019/03/02.
Copyright© 2019. Lau, KwanYuen. All rights reserved.

Abstract:
    A tensorflow implementation of Human2D network proposed by Georgios Pavlakos et al.
    Human2D network is used for 2D keypoint and silhouette estimation.
    It is a ConvNet follows the Stacked Hourglass design.

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
from utils import res_layer, max_pool_layer, hourglass_layer
from utils import cross_entropy, kl_divergence, squared_l2_norm
import time
import os
import numpy as np
import cv2 as cv


class Human2D:
    """ Base class for building the Human2D Model. """
    def __init__(self,
                 filenames='',
                 image_size=256,
                 batch_size=8,
                 num_epochs=None,
                 iterations=5e5,
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
        with tf.variable_scope(name_or_scope='human2d_init'):
            self.filenames = filenames
            self.image_size = image_size
            self.batch_size = batch_size
            self.num_epochs = num_epochs
            self.iterations = int(iterations)
            self.learning_rate = learning_rate
            self.training = tf.placeholder(tf.bool, name='training')
            self._input = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='input')
            self._silhouette = tf.placeholder(tf.float32, [None, 64, 64, 1], name='silhouette')
            self._keypoint = tf.placeholder(tf.float32, [None, 64, 64, 14], name='keypoint')

    def model(self):
        """ Generates a model

            Returns:
                keypoint_inter: The intermediate estimation of keypoints
                silhouette_inter: The intermediate estimation of silhouette
                keypoint_output: The estimation of keypoints
                silhouette_output: The estimation of silhouette
        """
        with tf.variable_scope(name_or_scope='human2d_network'):
            # down-sampling
            resi_0 = res_layer(self._input, filters=16, strides=2, kernel_size=7, training=self.training, name='resi_0')
            resi_1 = res_layer(resi_0, filters=32, strides=1, kernel_size=3, training=self.training, name='resi_1')
            pool_0 = max_pool_layer(resi_1, name='pool_0')
            resi_2 = res_layer(pool_0, filters=32, strides=1, kernel_size=3, training=self.training, name='resi_2')
            # first hourglass module
            resi_3 = res_layer(resi_2, filters=64, strides=1, kernel_size=3, training=self.training, name='resi_3')
            hrgs_0 = hourglass_layer(resi_3, training=self.training, name='hrgs_0')
            resi_4 = res_layer(hrgs_0, filters=64, strides=1, kernel_size=3, training=self.training, name='resi_4')
            # keypoint intermediate output
            keypoint_pre_0 = res_layer(resi_4, filters=64, strides=1, kernel_size=3, training=self.training,
                                       name='keypoint_pre_0')
            keypoint_pre_1 = res_layer(keypoint_pre_0, filters=64, strides=1, kernel_size=3, training=self.training,
                                       name='keypoint_pre_1')
            keypoint_inter_raw = res_layer(keypoint_pre_1, filters=14, strides=1, kernel_size=1,
                                           training=self.training, bottleneck=False, name='keypoint_inter_raw')
            keypoint_inter = tf.nn.sigmoid(x=keypoint_inter_raw, name='keypoint_inter')
            keypoint_post = res_layer(keypoint_inter, filters=64, strides=1, kernel_size=1,
                                      training=self.training, name='keypoint_post')
            # silhouette intermediate output
            silhouette_pre_0 = res_layer(resi_4, filters=64, strides=1, kernel_size=3, training=self.training,
                                         name='silhouette_pre_0')
            silhouette_pre_1 = res_layer(silhouette_pre_0, filters=64, strides=1, kernel_size=3, training=self.training,
                                         name='silhouette_pre_1')
            silhouette_inter_raw = res_layer(silhouette_pre_1, filters=2, strides=1, kernel_size=1,
                                             training=self.training, bottleneck=False, name='silhouette_inter_raw')
            silhouette_inter = tf.nn.softmax(logits=silhouette_inter_raw, name='silhouette_inter')
            silhouette_post = res_layer(silhouette_inter, filters=64, strides=1, kernel_size=1,
                                        training=self.training, name='silhouette_post')
            # second hourglass module
            concat = tf.concat([resi_3, resi_4, keypoint_post, silhouette_post], axis=-1, name='concat')
            hrgs_1 = hourglass_layer(concat, training=True, name='hrgs_1')
            resi_5 = res_layer(hrgs_1, filters=64, strides=1, kernel_size=3, training=self.training, name='resi_5')
            # keypoint output
            keypoint_pre_2 = res_layer(resi_5, filters=64, strides=1, kernel_size=3, training=self.training,
                                       name='keypoint_pre_2')
            keypoint_pre_3 = res_layer(keypoint_pre_2, filters=64, strides=1, kernel_size=3, training=self.training,
                                       name='keypoint_pre_3')
            keypoint_output_raw = res_layer(keypoint_pre_3, filters=14, strides=1, kernel_size=1,
                                            training=self.training, bottleneck=False, name='keypoint_output_raw')
            keypoint_output = tf.nn.sigmoid(x=keypoint_output_raw, name='keypoint_output')
            # silhouette output
            silhouette_pre_2 = res_layer(resi_5, filters=64, strides=1, kernel_size=3, training=self.training,
                                         name='silhouette_pre_2')
            silhouette_pre_3 = res_layer(silhouette_pre_2, filters=64, strides=1, kernel_size=3, training=self.training,
                                         name='silhouette_pre_3')
            silhouette_output_raw = res_layer(silhouette_pre_3, filters=2, strides=1, kernel_size=1,
                                              training=self.training, bottleneck=False, name='silhouette_output_raw')
            silhouette_output = tf.nn.softmax(logits=silhouette_output_raw, name='silhouette_output')
            # return
            return keypoint_inter, silhouette_inter, keypoint_output, silhouette_output

    def mini_model(self):
        """ Generates a mini model (cut off the stacking design of hourglass module)

            Returns:
                keypoint_output: The estimation of keypoints
                silhouette_output: The estimation of silhouette
        """
        with tf.variable_scope(name_or_scope='human2d_network'):
            # down-sampling
            resi_0 = res_layer(self._input, filters=16, strides=2, kernel_size=7, training=self.training, name='resi_0')
            resi_1 = res_layer(resi_0, filters=32, strides=1, kernel_size=3, training=self.training, name='resi_1')
            pool_0 = max_pool_layer(resi_1, name='pool_0')
            resi_2 = res_layer(pool_0, filters=32, strides=1, kernel_size=3, training=self.training, name='resi_2')
            # hourglass module
            resi_3 = res_layer(resi_2, filters=64, strides=1, kernel_size=3, training=self.training, name='resi_3')
            hrgs_0 = hourglass_layer(resi_3, training=True, name='hrgs_0')
            # keypoint output
            keypoint_pre_0 = res_layer(hrgs_0, filters=64, strides=1, kernel_size=3, training=self.training,
                                       name='keypoint_pre_0')
            keypoint_pre_1 = res_layer(keypoint_pre_0, filters=64, strides=1, kernel_size=3, training=self.training,
                                       name='keypoint_pre_1')
            keypoint_pre_2 = res_layer(keypoint_pre_1, filters=64, strides=1, kernel_size=3, training=self.training,
                                       name='keypoint_pre_2')
            keypoint_output_raw = res_layer(keypoint_pre_2, filters=14, strides=1, kernel_size=1,
                                            training=self.training, bottleneck=False, name='keypoint_output_raw')
            keypoint_output = tf.nn.sigmoid(x=keypoint_output_raw, name='keypoint_output')
            # silhouette output
            silhouette_pre_0 = res_layer(hrgs_0, filters=64, strides=1, kernel_size=3, training=self.training,
                                         name='silhouette_pre_0')
            silhouette_pre_1 = res_layer(silhouette_pre_0, filters=64, strides=1, kernel_size=3, training=self.training,
                                         name='silhouette_pre_1')
            silhouette_pre_2 = res_layer(silhouette_pre_1, filters=64, strides=1, kernel_size=3, training=self.training,
                                         name='silhouette_pre_2')
            silhouette_output_raw = res_layer(silhouette_pre_2, filters=2, strides=1, kernel_size=1,
                                              training=self.training, bottleneck=False, name='silhouette_output_raw')
            silhouette_output = tf.nn.softmax(logits=silhouette_output_raw, name='silhouette_output')
            # return
            return None, None, keypoint_output, silhouette_output

    def train(self, use_mini=False, use_mse=False):
        """ Train a Human2D model

            Args:
                use_mini: Whether to use the mini Human2D model or not
                use_mse: Whether to use MSE as the keypoint loss function or not
        """
        # decide whether to use MSE as the keypoint loss function
        if use_mse:  # to use MSE loss
            loss_fn = squared_l2_norm
        else:  # to use KL divergence loss
            loss_fn = kl_divergence

        # generate model
        if not use_mini:
            keypoint_inter, silhouette_inter, keypoint_output, silhouette_output = self.model()
        else:
            keypoint_inter, silhouette_inter, keypoint_output, silhouette_output = self.mini_model()
        keypoint_summary = tf.summary.histogram('keypoint_summary', keypoint_output)
        silhouette_summary = tf.summary.histogram('silhouette_summary', silhouette_output)

        with tf.variable_scope(name_or_scope='human2d_optimizer'):
            # loss functions
            if not use_mini:
                keypoint_loss = tf.reduce_mean(tf.concat(
                    [tf.reduce_sum(loss_fn(self._keypoint, keypoint_inter), axis=[1, 2]),
                     tf.reduce_sum(loss_fn(self._keypoint, keypoint_output), axis=[1, 2])], axis=-1),
                    name='keypoint_loss')
                silhouette_loss = tf.reduce_mean(tf.concat(
                    [tf.reduce_sum(cross_entropy(self._silhouette, silhouette_inter), axis=[1, 2]),
                     tf.reduce_sum(cross_entropy(self._silhouette, silhouette_output), axis=[1, 2])], axis=-1),
                    name='silhouette_loss')
            else:
                keypoint_loss = tf.reduce_mean(tf.reduce_sum(loss_fn(self._keypoint, keypoint_output),
                                                             axis=[1, 2]), name='keypoint_loss')
                silhouette_loss = tf.reduce_mean(tf.reduce_sum(cross_entropy(self._silhouette, silhouette_output),
                                                               axis=[1, 2]), name='silhouette_loss')
            total_loss = 100 * keypoint_loss + silhouette_loss
            loss_summary = tf.summary.scalar('loss_summary', total_loss)

            # optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_ops = optimizer.minimize(total_loss)
                # unfrozen = [var for var in tf.trainable_variables() if not var.name.startswith('')]
                # grads_and_vars = optimizer.compute_gradients(total_loss, var_list=unfrozen)
                # capped_grads_and_vars = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads_and_vars]
                # train_ops = optimizer.apply_gradients(capped_grads_and_vars)

        # configuration
        if not os.path.isdir('model'):
            os.mkdir('model')
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.allow_soft_placement = True
        config.log_device_placement = False

        print('Start training Human2D')
        with tf.Session(config=config) as sess:
            # initialization
            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

            # saver
            saver = tf.train.Saver(max_to_keep=5)

            # store the network graph for tensorboard visualization
            writer = tf.summary.FileWriter('model/network_graph', sess.graph)
            merge_op = tf.summary.merge([keypoint_summary, silhouette_summary, loss_summary])

            # dataset
            data = Reader(filenames=self.filenames, image_size=self.image_size,
                          batch_size=self.batch_size, num_epochs=self.num_epochs).feed()
            try:
                for i in range(self.iterations):
                    # input
                    _input, _silhouette, _keypoint, _, _ = sess.run(data)
                    _, k_out, s_out, k_loss, s_loss, sm = sess.run([train_ops, keypoint_output, silhouette_output,
                                                                    keypoint_loss, silhouette_loss, merge_op],
                                                                   feed_dict={self.training: True, self._input: _input,
                                                                              self._silhouette: _silhouette,
                                                                              self._keypoint: _keypoint})
                    if i % 10 == 0:
                        print('iteration %d: keypoint loss = %f, silhouette loss = %f' % (i, k_loss, s_loss))
                    if i % 200 == 0:
                        writer.add_summary(sm, i)
                        writer.flush()
                        cv.imwrite('visual/input.jpg', _input[0, :, :, ::-1])
                        cv.imwrite('visual/silhouette.jpg', Human2D.generate_silhouette(s_out[0]))
                        for j in range(14):
                            cv.imwrite('visual/keypoint_%d.jpg' % j, 255 * k_out[0, :, :, j])
                    if i % 500 == 0 and i != 0:
                        print('save at iteration %d' % i)
                        saver.save(sess, 'model/%s/human2d' %
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
        data = Reader(filenames=filenames, batch_size=1, num_epochs=1).feed(shuffle=False)
        print('start testing')
        # session
        with tf.Session(graph=graph) as sess:
            # restore the latest model
            file_list = os.listdir('model/')
            file_list.sort(key=lambda x: x)
            loader = tf.train.import_meta_graph('model/%s/human2d.meta' % file_list[-2])
            # get input tensor
            training_tensor = graph.get_tensor_by_name('human2d_init/training:0')
            input_tensor = graph.get_tensor_by_name('human2d_init/input:0')
            # get output tensor
            silhouette_output_tensor = graph.get_tensor_by_name('human2d_network/silhouette_output:0')
            keypoint_output_tensor = graph.get_tensor_by_name('human2d_network/keypoint_output:0')

            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
            loader.restore(sess, tf.train.latest_checkpoint('model/%s' % file_list[-2]))

            # output
            index = 0
            k_loss = 0
            s_loss = 0
            try:
                if not os.path.isdir('result'):
                    os.mkdir('result')
                while True:
                    _input, _silhouette, _keypoint, _, _ = sess.run(data)
                    silhouette_output, keypoint_output = sess.run([silhouette_output_tensor, keypoint_output_tensor],
                                                                  feed_dict={training_tensor: False,
                                                                             input_tensor: _input})

                    # post-process
                    silhouette_output = Human2D.generate_silhouette(silhouette_output)
                    silhouette_loss = np.average(np.sum(
                        np.square(_silhouette - silhouette_output) / 65025, axis=(1, 2)))
                    keypoint_loss = np.average(np.sum(
                        np.square(np.nan_to_num(_keypoint) - keypoint_output), axis=(1, 2)))
                    keypoint_output = Human2D.generate_keypoint(keypoint_output)

                    k_loss = k_loss + keypoint_loss
                    s_loss = s_loss + silhouette_loss

                    # optional operation
                    if to_quantify:
                        print('silhouette loss = %f, keypoint loss = %f' % (silhouette_loss, keypoint_loss))
                    if to_visualize:
                        cv.imwrite('result/silhouette_%05d.jpg' % index, silhouette_output[0])
                        # np.save('result/keypoint_%05d.npy' % index, keypoint_output)
                        for i in range(14):
                            cv.circle(_input[0], (int(4 * keypoint_output[0, 0, i]), int(4 * keypoint_output[0, 1, i])),
                                      5, (255, 255, 255), -1)
                        cv.imwrite('result/keypoint_%05d.jpg' % index, _input[0, :, :, ::-1])
                    index = index + 1
            except tf.errors.OutOfRangeError:
                print('Done testing -- epoch limit reached')
                print('keypoint loss = %f, silhouette loss = %f' % (k_loss / index, s_loss / index))

    @staticmethod
    def fine_tune(filenames='',
                  image_size=256,
                  batch_size=8,
                  num_epochs=None,
                  iterations=int(1e6),
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
            loader = tf.train.import_meta_graph('model/%s/human2d.meta' % file_list[-2])

            # get input tensor
            training_tensor = graph.get_tensor_by_name('human2d_init/training:0')
            input_tensor = graph.get_tensor_by_name('human2d_init/input:0')
            silhouette_tensor = graph.get_tensor_by_name('human2d_init/silhouette:0')
            keypoint_tensor = graph.get_tensor_by_name('human2d_init/keypoint:0')
            # get output tensor
            silhouette_output_tensor = graph.get_tensor_by_name('human2d_network/silhouette_output:0')
            keypoint_output_tensor = graph.get_tensor_by_name('human2d_network/keypoint_output:0')
            # get loss tensor
            silhouette_loss_tensor = graph.get_tensor_by_name('human2d_optimizer/silhouette_loss:0')
            # we use MSE loss while fine tuning to eliminate False Positive
            keypoint_loss_tensor = tf.reduce_mean(tf.reduce_sum(
                squared_l2_norm(keypoint_tensor, keypoint_output_tensor), axis=[1, 2]))

            # optimizer
            total_loss = 100 * keypoint_loss_tensor + silhouette_loss_tensor

            with tf.variable_scope(name_or_scope='fine_tune'):
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_ops = optimizer.minimize(total_loss)

            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
            loader.restore(sess, tf.train.latest_checkpoint('model/%s' % file_list[-2]))

            # saver
            saver = tf.train.Saver(max_to_keep=5)

            print('Start fine tuning Human2D')
            try:
                for i in range(iterations):
                    # input
                    _input, _silhouette, _keypoint, _, _ = sess.run(data)
                    _, k_out, s_out, k_loss, s_loss = sess.run([train_ops,
                                                                keypoint_output_tensor, silhouette_output_tensor,
                                                                keypoint_loss_tensor, silhouette_loss_tensor],
                                                               feed_dict={training_tensor: True, input_tensor: _input,
                                                                          silhouette_tensor: _silhouette,
                                                                          keypoint_tensor: _keypoint})
                    if i % 10 == 0:
                        print('iteration %d: keypoint loss = %f, silhouette loss = %f' % (i, k_loss, s_loss))
                    if i % 200 == 0:
                        cv.imwrite('visual/input.jpg', _input[0, :, :, ::-1])
                        cv.imwrite('visual/silhouette.jpg', Human2D.generate_silhouette(s_out[0]))
                        for j in range(14):
                            cv.imwrite('visual/keypoint_%d.jpg' % j, 255 * k_out[0, :, :, j])
                    if i % 500 == 0 and i != 0:
                        print('save at iteration %d' % i)
                        saver.save(sess, 'model/%s/human2d' %
                                   (time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))))
                    time.sleep(1)
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')

    @classmethod
    def generate_silhouette(cls, silhouette):
        """ Generates a silhouette from a prediction (probability of 2 classes)

            Args:
                silhouette: The prediction of silhouette (with shape of [1, 64, 64, 2])

            Returns:
                silhouette: A silhouette generated from a given prediction
        """
        silhouette = np.uint8(255 * np.argmax(silhouette, axis=-1))
        silhouette = np.reshape(silhouette, newshape=[-1, 64, 64, 1])
        return silhouette

    @classmethod
    def generate_keypoint(cls, keypoint):
        """ Generates keypoints from 14-channel heatmaps

            Args:
                keypoint: 14-channel heatmaps (with shape of [None, 64, 64, 14])

            Returns:
                keypoint: A keypoint ndarray generated from given heatmaps
        """
        batch_size = keypoint.shape[0]

        joints = np.zeros(shape=[batch_size, 3, 14])

        for i in range(batch_size):
            for j in range(14):
                coords = cv.minMaxLoc(keypoint[i, :, :, j])
                joints[i, 0, j] = coords[3][0]
                joints[i, 1, j] = coords[3][1]
                joints[i, 2, j] = coords[1]
        return joints


if __name__ == '__main__':
    old_v = tf.logging.get_verbosity()
    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.logging.set_verbosity(old_v)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    np.set_printoptions(threshold=np.inf)

    # h = Human2D(filenames='/home/laukyuen3/thesis/mpii.tfrecords', batch_size=8)
    # h.train(use_mini=False)
    # Human2D.fine_tune(filenames='/home/laukyuen3/thesis/mpii.tfrecords', batch_size=8, learning_rate=3e-6)
    Human2D.test(filenames='/home/laukyuen3/thesis/lspext.tfrecords', to_visualize=False, to_quantify=True)
