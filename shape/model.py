# -*- coding: utf-8 -*-

"""
shape.model.py

Created by Lau, KwanYuen on 2019/03/02.
Copyright© 2019. Lau, KwanYuen. All rights reserved.

Abstract:
    A tensorflow implementation of ShapePrior network proposed by Georgios Pavlakos et al.
    ShapePrior network estimates 3D shape from 2D silhouette.

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
from utils import res_layer, max_pool_layer, fc_layer, bilinear_layer
from smpl.smpl_webuser.serialization import load_model
import numpy as np
import time
import cv2 as cv
import os


class ShapePrior:
    """ Base class for building the ShapePrior Model. """
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
        with tf.variable_scope(name_or_scope='shape_prior_init'):
            self.filenames = filenames
            self.image_size = image_size
            self.batch_size = batch_size
            self.num_epochs = num_epochs
            self.iterations = int(iterations)
            self.learning_rate = learning_rate
            self.training = tf.placeholder(tf.bool, name='training')
            self._input = tf.placeholder(tf.float32, [None, 64, 64, 1], name='input')
            self._beta = tf.placeholder(tf.float32, [None, 10], name='beta')

    def model(self):
        """ Generates a model

            Returns:
                beta_output: The estimation of beta parameters of SMPL model
        """
        with tf.variable_scope(name_or_scope='shape_prior_network'):
            resi_0 = res_layer(self._input, 8, 1, 3, training=self.training, name='resi_0')
            pool_0 = max_pool_layer(resi_0, name='pool_0')

            resi_1 = res_layer(pool_0, 16, 1, 3, training=self.training, name='resi_1')
            pool_1 = max_pool_layer(resi_1, name='pool_1')

            resi_2 = res_layer(pool_1, 32, 1, 3, training=self.training, name='resi_2')
            pool_2 = max_pool_layer(resi_2, name='pool_2')

            resi_3 = res_layer(pool_2, 32, 1, 3, training=self.training, name='resi_3')
            pool_3 = max_pool_layer(resi_3, name='pool_3')

            resi_4 = res_layer(pool_3, 32, 1, 3, training=self.training, name='resi_4')
            pool_4 = max_pool_layer(resi_4, name='pool_4')

            fc_0 = fc_layer(tf.layers.flatten(pool_4), 512, training=self.training, name='fc_1')
            bi_0 = bilinear_layer(fc_0, 512, training=self.training, name='bi_1')
            beta_output = tf.layers.dense(bi_0, units=10, activation=None, use_bias=False, name='beta_output')

            return beta_output

    def train(self):
        """ Train a ShapePrior model

        """
        # generate model
        beta = self.model()

        with tf.variable_scope(name_or_scope='shape_prior_optimizer'):
            _vertex = tf.placeholder(tf.float32, [None, 6890, 3], name='vertex')
            _vertex_gt = tf.placeholder(tf.float32, [None, 6890, 3], name='vertex_gt')

            # loss function
            beta_loss = tf.reduce_mean(tf.reduce_sum(tf.square(beta - self._beta), axis=1), name='beta_loss')
            vertex_loss = tf.reduce_mean(tf.reduce_sum(tf.square(_vertex - _vertex_gt), axis=[1, 2]),
                                         name='vertex_loss')
            total_loss = beta_loss + vertex_loss

            # optimizer
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_ops = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(total_loss)

        # configuration
        if not os.path.isdir('model'):
            os.mkdir('model')
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.allow_soft_placement = True
        config.log_device_placement = False

        print('Start training ShapePrior')
        with tf.Session(config=config) as sess:
            # initialization
            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

            # saver
            saver = tf.train.Saver(max_to_keep=5)

            # store the network graph for tensorboard visualization
            writer = tf.summary.FileWriter('model/network_graph', sess.graph)
            writer.close()

            # dataset
            data = Reader(filenames=self.filenames, image_size=self.image_size,
                          batch_size=self.batch_size, num_epochs=self.num_epochs).feed()

            try:
                for i in range(self.iterations):
                    _, _silhouette, _, _, _beta = sess.run(data)

                    if i < 0.4 * self.iterations:
                        zero = np.zeros(shape=[self.batch_size, 6890, 3])
                        _, loss = sess.run([train_ops, total_loss],
                                           feed_dict={self.training: True,
                                                      self._input: _silhouette, self._beta: _beta,
                                                      _vertex: zero, _vertex_gt: zero})
                    else:
                        vertex, _ = ShapePrior.get_mesh(
                            sess.run(beta, feed_dict={self.training: False, self._input: _silhouette}))
                        vertex_gt, _ = ShapePrior.get_mesh(_beta)

                        _, loss = sess.run([train_ops, total_loss],
                                           feed_dict={self.training: True,
                                                      self._input: _silhouette, self._beta: _beta,
                                                      _vertex: vertex, _vertex_gt: vertex_gt})
                    if i % 10 == 0:
                        print('iteration %d: loss = %f' % (i, loss))
                    if i % 200 == 0:
                        ShapePrior.save_mesh(sess.run(beta, feed_dict={self.training: False, self._input: _silhouette})
                                             , 'visual/out.obj')
                        ShapePrior.save_mesh(_beta, 'visual/gt.obj')
                    if i % 500 == 0 and i != 0:
                        print('save at iteration %d' % i)
                        saver.save(sess, 'model/%s/shape' %
                                   (time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))))

                    # time.sleep(1)
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')

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
        data = Reader(filenames=filenames, batch_size=1324, num_epochs=1).feed(shuffle=True)
        # session
        with tf.Session(graph=graph) as sess:
            # restore the latest model
            file_list = os.listdir('model/')
            file_list.sort(key=lambda x: x)
            loader = tf.train.import_meta_graph('model/%s/shape.meta' % file_list[-2])
            # get input tensor
            training_tensor = graph.get_tensor_by_name('shape_prior_init/training:0')
            input_tensor = graph.get_tensor_by_name('shape_prior_init/input:0')
            vertex_tensor = graph.get_tensor_by_name('shape_prior_optimizer/vertex:0')
            vertex_gt_tensor = graph.get_tensor_by_name('shape_prior_optimizer/vertex_gt:0')
            # get output tensor
            beta_output_tensor = graph.get_tensor_by_name('shape_prior_network/beta_output/MatMul:0')
            # get loss tensor
            vertex_loss_tensor = graph.get_tensor_by_name('shape_prior_optimizer/vertex_loss:0')

            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
            loader.restore(sess, tf.train.latest_checkpoint('model/%s' % file_list[-2]))

            # output
            index = 0
            try:
                if not os.path.isdir('result'):
                    os.mkdir('result')
                while True:
                    _input, _silhouette, _, _, _beta = sess.run(data)

                    beta_output = sess.run(beta_output_tensor,
                                           feed_dict={training_tensor: False, input_tensor: _silhouette})

                    _vertex, _ = ShapePrior.get_mesh(beta_output)
                    _vertex_gt, _ = ShapePrior.get_mesh(_beta)
                    vertex_loss = sess.run(vertex_loss_tensor,
                                           feed_dict={vertex_tensor: _vertex, vertex_gt_tensor: _vertex_gt})
                    beta_loss = np.average(np.sum(np.square(_beta - beta_output), axis=1))

                    # optional operation
                    if to_quantify:
                        print('beta loss = %f, vertex loss = %f' % (beta_loss, vertex_loss))
                    if to_visualize:
                        cv.imwrite('result/input_%05d.jpg' % index, _input[0, :, :, ::-1])
                        ShapePrior.save_mesh(beta_output, 'result/out_%05d.obj' % index)
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
            loader = tf.train.import_meta_graph('model/%s/shape.meta' % file_list[-2])

            # get input tensor
            training_tensor = graph.get_tensor_by_name('shape_prior_init/training:0')
            input_tensor = graph.get_tensor_by_name('shape_prior_init/input:0')
            beta_tensor = graph.get_tensor_by_name('shape_prior_init/beta:0')
            # get output tensor
            beta_output_tensor = graph.get_tensor_by_name('shape_prior_network/beta_output/output/cond/Merge:0')
            # get loss tensor
            _vertex = tf.placeholder(tf.float32, [None, 6890, 3])
            _vertex_gt = tf.placeholder(tf.float32, [None, 6890, 3])
            # loss function
            beta_loss = tf.reduce_mean(tf.reduce_sum(tf.square(beta_tensor - beta_output_tensor), axis=1))
            vertex_loss = tf.reduce_mean(tf.reduce_sum(tf.square(_vertex - _vertex_gt), axis=[1, 2]))
            total_loss = beta_loss + vertex_loss

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

            print('Start fine tuning ShapePrior')
            try:
                for i in range(iterations):
                    # input
                    _, _silhouette, _, _, _beta = sess.run(data)
                    vertex, _ = ShapePrior.get_mesh(
                        sess.run(beta_output_tensor, feed_dict={training_tensor: False, input_tensor: _silhouette}))
                    vertex_gt, _ = ShapePrior.get_mesh(_beta)
                    _, b_out, b_loss, v_loss = sess.run([train_ops,
                                                         beta_output_tensor, beta_loss, vertex_loss],
                                                        feed_dict={training_tensor: True, input_tensor: _silhouette,
                                                                   beta_tensor: _beta,
                                                                   _vertex: vertex, _vertex_gt: vertex_gt})
                    if i % 10 == 0:
                        print('iteration %d: beta loss = %f, vertex loss = %f' % (i, b_loss, v_loss))
                    if i % 200 == 0:
                        ShapePrior.save_mesh(sess.run(beta_output_tensor,
                                                      feed_dict={training_tensor: False, input_tensor: _silhouette})
                                             , 'visual/out.obj')
                        ShapePrior.save_mesh(_beta, 'visual/gt.obj')
                    if i % 500 == 0 and i != 0:
                        print('save at iteration %d' % i)
                        saver.save(sess, 'model/%s/shape' %
                                   (time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))))
                    # time.sleep(1)
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')

    @classmethod
    def get_mesh(cls, beta):
        """ Generates a 3D mesh from a set of beta

            Args:
                beta: A set of beta parameters (with shape of [None, 10])

            Returns:
                vertex: The vertices of SMPL meshes
                joint: The joints of SMPL messsshes
        """
        batch_size = beta.shape[0]

        vertex = np.zeros(shape=[batch_size, 6890, 3])
        joint = np.zeros(shape=[batch_size, 24, 3])

        for i in range(batch_size):
            ShapePrior.model_template.betas[:] = beta[i]
            vertex[i] = ShapePrior.model_template.r
            joint[i] = ShapePrior.model_template.J

        return vertex, joint

    @classmethod
    def save_mesh(cls, beta, filename=''):
        ShapePrior.model_template.betas[:] = beta[0]

        with open(filename, 'w') as fp:
            for v in ShapePrior.model_template.r:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

            for f in ShapePrior.model_template.f + 1:  # Faces are 1-based, not 0-based in obj files
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


if __name__ == '__main__':
    old_v = tf.logging.get_verbosity()
    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.logging.set_verbosity(old_v)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    np.set_printoptions(threshold=np.inf)

    # s = ShapePrior(filenames='/home/laukyuen3/thesis/mpii.tfrecords', batch_size=32, learning_rate=3e-5)
    # s.train()
    # ShapePrior.fine_tune(filenames='/home/laukyuen3/thesis/mpii.tfrecords', batch_size=8, learning_rate=3e-5)

    ShapePrior.test(filenames='/home/laukyuen3/thesis/lspext.tfrecords', to_quantify=True, to_visualize=False)
