# -*- coding: utf-8 -*-

"""
pipeline.py

Created by Lau, KwanYuen on 2019/03/02.
Copyright© 2019. Lau, KwanYuen. All rights reserved.

Abstract:
    A tensorflow implementation of model proposed by Georgios Pavlakos et al.
    This model is used for Learning to Estimate 3D Human Pose and Shape from a Single Color Image.

    @inproceedings {
        pavlakos2018humanshape,
        Author = {Pavlakos, Georgios and Zhu, Luyang and Zhou, Xiaowei and Daniilidis, Kostas},
        Title = {Learning to Estimate 3{D} Human Pose and Shape from a Single Color Image},
        Booktitle = {CVPR},
        Year = {2018}
    }

    The pipeline is trained with the dataset collected by Lassner et al.

    @inproceedings {
        Lassner:UP:2017,
        title = {Unite the People: Closing the Loop Between 3D and 2D Human Representations},
        author = {Lassner, Christoph and Romero, Javier and Kiefel, Martin and Bogo,
                  Federica and Black, Michael J. and Gehler, Peter V.},
        booktitle = {IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
        month = jul,
        year = {2017},
        url = {http://up.is.tuebingen.mpg.de},
        month_numeric = {7}
    }

    The 3D mesh(es) involved is(are) powered by SMPL, which was proposed by Loper et al.

    @article {
        SMPL:2015,
        author = {Loper, Matthew and Mahmood, Naureen and Romero, Javier and Pons-Moll, Gerard and Black, Michael J.},
        title = {{SMPL}: A Skinned Multi-Person Linear Model},
        journal = {ACM Trans. Graphics (Proc. SIGGRAPH Asia)},
        month = oct,
        number = {6},
        pages = {248:1--248:16},
        publisher = {ACM},
        volume = {34},
        year = {2015}
    }

    The template neutral SMPL model used in the pipeline was trained by Bogo et al.

    @inproceedings {
        Bogo:ECCV:2016,
        title = {Keep it {SMPL}: Automatic Estimation of {3D} Human Pose and Shape from a Single Image},
        author = {Bogo, Federica and Kanazawa, Angjoo and Lassner, Christoph and
        Gehler, Peter and Romero, Javier and Black, Michael J.},
        booktitle = {Computer Vision -- ECCV 2016},
        series = {Lecture Notes in Computer Science},
        publisher = {Springer International Publishing},
        month = oct,
        year = {2016}
    }

"""


from reader import Reader
from smpl.smpl_webuser.serialization import load_model
from opendr.camera import ProjectPoints
from utils import squared_l2_norm, overlap_distance
import tensorflow as tf
import numpy as np
import cv2 as cv
import os
import time


class Prior:
    """ Base class for building the Prior Model. """
    model_template = load_model('smpl/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')

    def __init__(self,
                 filenames='',
                 image_size=256,
                 batch_size=256,
                 num_epochs=None,
                 iterations=1e5,
                 learning_rate=3e-4):
        self.filenames = filenames
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.iterations = int(iterations)
        self.learning_rate = learning_rate

    def train(self):
        graph_shape = tf.get_default_graph()
        with tf.Session(graph=graph_shape) as sess_shape:
            # restore the latest model
            file_list = os.listdir('shape/model/')
            file_list.sort(key=lambda x: x)
            loader = tf.train.import_meta_graph('shape/model/%s/shape.meta' % file_list[-2])

            sess_shape.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
            loader.restore(sess_shape, tf.train.latest_checkpoint('shape/model/%s' % file_list[-2]))

        graph_pose = tf.get_default_graph()
        with tf.Session(graph=graph_pose) as sess_pose:
            # restore the latest model
            file_list = os.listdir('pose/model/')
            file_list.sort(key=lambda x: x)
            loader = tf.train.import_meta_graph('pose/model/%s/pose.meta' % file_list[-2])

            sess_pose.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
            loader.restore(sess_pose, tf.train.latest_checkpoint('pose/model/%s' % file_list[-2]))

        # configuration
        if not os.path.isdir('prior'):
            os.mkdir('prior')
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.allow_soft_placement = True
        config.log_device_placement = False

        print('Start training Prior')
        graph_prior = tf.get_default_graph()
        with tf.Session(graph=graph_prior, config=config) as sess_prior:

            with tf.variable_scope(name_or_scope='prior_init'):
                training = tf.placeholder(tf.bool, name='training')
                shape_input = tf.placeholder(tf.float32, [None, 256, 256, 1], name='shape_input')
                pose_input = tf.placeholder(tf.float32, [None, 3, 14], name='pose_input')
                beta = tf.placeholder(tf.float32, [None, 10], name='beta')
                theta = tf.placeholder(tf.float32, [None, 72], name='theta')

            with tf.variable_scope(name_or_scope='prior_network'):
                beta_output = tf.import_graph_def(graph_shape.as_graph_def(),
                                                  input_map={'shape_prior_init/input:0': shape_input,
                                                             'shape_prior_init/training:0': training,
                                                             'shape_prior_init/beta:0': beta},
                                                  return_elements=['shape_prior_network/beta_output:0'],
                                                  name='beta_output')
                theta_output = tf.import_graph_def(graph_pose.as_graph_def(),
                                                   input_map={'pose_prior_init/input:0': pose_input,
                                                              'pose_prior_init/training:0': training,
                                                              'pose_prior_init/theta:0': theta},
                                                   return_elements=['pose_prior_network/theta_output:0'],
                                                   name='theta_output')

            with tf.variable_scope(name_or_scope='prior_optimizer'):
                _vertex = tf.placeholder(tf.float32, [None, 6890, 3], name='vertex')
                _vertex_gt = tf.placeholder(tf.float32, [None, 6890, 3], name='vertex_gt')

                # loss function
                total_loss = tf.reduce_mean(tf.reduce_sum(tf.square(_vertex - _vertex_gt), axis=[1, 2]),
                                            name='vertex_loss')

                # optimizer
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_ops = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(total_loss)

            # saver
            saver = tf.train.Saver(max_to_keep=5)

            # store the network graph for tensorboard visualization
            writer = tf.summary.FileWriter('prior/network_graph', sess_prior.graph)
            writer.close()

            # dataset
            data = Reader(filenames=self.filenames, image_size=self.image_size,
                          batch_size=self.batch_size, num_epochs=self.num_epochs).feed()

            # initialization
            sess_prior.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

            try:
                for i in range(self.iterations):
                    _, _silhouette, _keypoint, _theta, _beta = sess_prior.run(data)
                    _keypoint = Prior.generate_keypoint(_keypoint)

                    vertex, _ = Prior.get_mesh(sess_prior.run([beta_output, theta_output],
                                                              feed_dict={training: False,
                                                                         shape_input: _silhouette,
                                                                         pose_input: _keypoint}))
                    vertex_gt, _ = Prior.get_mesh([_beta, _theta])

                    _, loss = sess_prior.run([train_ops, total_loss],
                                             feed_dict={training: True,
                                                        shape_input: _silhouette, beta: _beta,
                                                        pose_input: _keypoint, theta: _theta,
                                                        _vertex: vertex, _vertex_gt: vertex_gt})
                    if i % 100 == 0:
                        print('iteration %d: loss = %f' % (i, loss))
                    if i % 500 == 0:
                        print('save at iteration %d' % i)
                        saver.save(sess_prior, 'prior/%s/prior' %
                                   (time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))))

                        vertex, _ = Prior.get_mesh(sess_prior.run([beta_output, theta_output],
                                                                  feed_dict={training: False,
                                                                             shape_input: _silhouette,
                                                                             pose_input: _keypoint}))
                        vertex_gt, _ = Prior.get_mesh([_beta, _theta])
                        Prior.save_mesh(vertex[0], 'visual/out.obj')
                        Prior.save_mesh(vertex_gt[0], 'visual/gt.obj')
                    time.sleep(3)
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')

    @staticmethod
    def test(filenames='', to_quantify=True, to_visualize=True):
        graph_shape = tf.Graph()
        with graph_shape.as_default():
            sess_shape = tf.Session(graph=graph_shape)
            with sess_shape.as_default():
                # restore the latest model
                file_list = os.listdir('shape/model/')
                file_list.sort(key=lambda x: x)
                shape_path = file_list[-2]
                loader_shape = tf.train.import_meta_graph('shape/model/%s/shape.meta' % shape_path)

                shape_input = graph_shape.get_tensor_by_name('shape_prior_init/input:0')
                shape_training = graph_shape.get_tensor_by_name('shape_prior_init/training:0')
                beta_output = graph_shape.get_tensor_by_name('shape_prior_network/beta_output/MatMul:0')

        graph_pose = tf.Graph()
        with graph_pose.as_default():
            sess_pose = tf.Session(graph=graph_pose)
            with sess_pose.as_default():
                # restore the latest model
                file_list = os.listdir('pose/model/')
                file_list.sort(key=lambda x: x)
                pose_path = file_list[-2]
                loader_pose = tf.train.import_meta_graph('pose/model/%s/pose.meta' % pose_path)

                pose_input = graph_pose.get_tensor_by_name('pose_prior_init/input:0')
                pose_training = graph_pose.get_tensor_by_name('pose_prior_init/training:0')
                theta_output = graph_pose.get_tensor_by_name('pose_prior_network/theta_output/MatMul:0')

        # configuration
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.allow_soft_placement = True
        config.log_device_placement = False

        with tf.Session(config=config) as sess_prior:
            _vertex = tf.placeholder(tf.float32, [None, 6890, 3], name='vertex')
            _vertex_gt = tf.placeholder(tf.float32, [None, 6890, 3], name='vertex_gt')

            # loss function
            total_loss = tf.reduce_mean(tf.reduce_sum(tf.square(_vertex - _vertex_gt), axis=[1, 2]), name='vertex_loss')

            # dataset
            data = Reader(filenames=filenames, image_size=256, batch_size=1, num_epochs=1).feed(shuffle=True)

            # initialization
            sess_prior.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
            loader_shape.restore(sess_shape, tf.train.latest_checkpoint('shape/model/%s' % shape_path))
            loader_pose.restore(sess_pose, tf.train.latest_checkpoint('pose/model/%s' % pose_path))

            index = 0
            v_loss = 0
            try:
                while True:
                    _input, _silhouette, _keypoint, _theta, _beta = sess_prior.run(data)
                    _keypoint = Prior.generate_keypoint(_keypoint)

                    with sess_shape.as_default():
                        _beta_output = sess_shape.run(beta_output, feed_dict={shape_input: _silhouette,
                                                                              shape_training: False})
                    with sess_pose.as_default():
                        _theta_output = sess_pose.run(theta_output, feed_dict={pose_input: _keypoint,
                                                                               pose_training: False}) / 500
                    vertex, _ = Prior.get_mesh([_beta_output, _theta_output])
                    vertex_gt, _ = Prior.get_mesh([_beta, _theta])

                    loss = sess_prior.run(total_loss, feed_dict={_vertex: vertex, _vertex_gt: vertex_gt})
                    v_loss = v_loss + loss
                    if to_quantify:
                        print('vertex loss = %f' % loss)
                    if to_visualize:
                        cv.imwrite('prior/result/input_%05d.jpg' % index, _input[0, :, :, ::-1])
                        Prior.save_mesh(_beta_output[0], _theta_output[0], 'prior/result/out_%05d.obj' % index)
                    index = index + 1
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
                print('vertex loss = %f' % (v_loss / index))

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

    @classmethod
    def get_mesh(cls, param):
        """ Generates a 3D mesh from a set of beta

            Args:
                param: A list containing beta parameters and theta parameters, in which,
                       beta is a set of beta parameters (with shape of [None, 10]) and
                       theta is a set of theta parameters (with shape of [None, 72])

            Returns:
                vertex: The vertices of SMPL meshes
                joint: The joints of SMPL messsshesss
        """
        assert len(param) == 2
        assert param[0].shape[0] == param[1].shape[0]

        batch_size = param[0].shape[0]

        vertex = np.zeros(shape=[batch_size, 6890, 3])
        joint = np.zeros(shape=[batch_size, 24, 3])

        for i in range(batch_size):
            Prior.model_template.betas[:] = param[0][i]
            Prior.model_template.pose[3:] = param[1][i, 3:]
            vertex[i] = Prior.model_template.r
            joint[i] = Prior.model_template.J

        return vertex, joint

    @classmethod
    def save_mesh(cls, beta, theta, filename=''):
        Prior.model_template.betas[:] = beta[:]
        Prior.model_template.pose[3:] = theta[3:]
        with open(filename, 'w') as fp:
            for v in Prior.model_template.r:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

            for f in Prior.model_template.f + 1:  # Faces are 1-based, not 0-based in obj files
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


class Pipeline:
    """ Base class for building the Pipeline. """
    model_template = load_model('smpl/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')

    def __init__(self,
                 filenames='',
                 image_size=256,
                 batch_size=8,
                 num_epochs=None,
                 iterations=1e4,
                 learning_rate=3e-4):
        self.filenames = filenames
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.iterations = int(iterations)
        self.learning_rate = learning_rate

    def train(self):
        graph_human2d = tf.get_default_graph()
        with tf.Session(graph=graph_human2d) as sess_human2d:
            # restore the latest model
            file_list = os.listdir('human2d/model/')
            file_list.sort(key=lambda x: x)
            loader = tf.train.import_meta_graph('human2d/model/%s/human2d.meta' % file_list[-2])

            sess_human2d.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
            loader.restore(sess_human2d, tf.train.latest_checkpoint('pose/model/%s' % file_list[-2]))

        graph_prior = tf.get_default_graph()
        with tf.Session(graph=graph_prior) as sess_prior:
            # restore the latest model
            file_list = os.listdir('prior/model/')
            file_list.sort(key=lambda x: x)
            loader = tf.train.import_meta_graph('prior/model/%s/prior.meta' % file_list[-2])

            sess_prior.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
            loader.restore(sess_prior, tf.train.latest_checkpoint('prior/model/%s' % file_list[-2]))

        # configuration
        if not os.path.isdir('pipeline'):
            os.mkdir('pipeline')
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.allow_soft_placement = True
        config.log_device_placement = False

        print('Start training Pipeline')
        graph_pipeline = tf.get_default_graph()
        with tf.Session(graph=graph_pipeline, config=config) as sess_pipeline:

            with tf.variable_scope(name_or_scope='pipeline_init'):
                training = tf.placeholder(tf.bool, name='training')
                human2d_input = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='input')

            with tf.variable_scope(name_or_scope='pipeline_network'):
                sil_output, kpt_output = tf.import_graph_def(graph_human2d.as_graph_def(),
                                                             input_map={'human2d_init/input:0': human2d_input,
                                                                        'human2d_init/training:0': training},
                                                             return_elements=['human2d_network/silhouette_output:0',
                                                                              'human2d_network/keypoint_output:0'])
                beta_output, theta_output = tf.import_graph_def(graph_prior.as_graph_def(),
                                                                input_map={'prior_init/input:0': sil_output,
                                                                           'prior_init/training:0': training},
                                                                return_elements=['prior_network/beta_output:0',
                                                                                 'prior_network/theta_output:0'])

            with tf.variable_scope(name_or_scope='pipeline_optimizer'):
                kpt_project = tf.placeholder(tf.float32, shape=[None, 2, 14])
                kpt_gt = tf.placeholder(tf.float32, shape=[None, 2, 14])
                overlap_loss = tf.placeholder(tf.float32)

                # loss function
                total_loss = squared_l2_norm(kpt_gt, kpt_project) + overlap_loss

                # optimizer
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_ops = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(total_loss)

            # saver
            saver = tf.train.Saver(max_to_keep=5)

            # store the network graph for tensorboard visualization
            writer = tf.summary.FileWriter('prior/network_graph', sess_pipeline.graph)
            writer.close()

            # dataset
            data = Reader(filenames=self.filenames, image_size=self.image_size,
                          batch_size=self.batch_size, num_epochs=self.num_epochs).feed()

            # initialization
            sess_pipeline.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

            try:
                for i in range(self.iterations):
                    _input, _silhouette, _keypoint, _theta, _beta = sess_pipeline.run(data)
                    _keypoint = Pipeline.generate_keypoint(_keypoint)

                    vertex, joint = Pipeline.get_mesh(sess_pipeline.run([beta_output, theta_output],
                                                                        feed_dict={training: False,
                                                                                   human2d_input: _input}))
                    sil, kpt = Pipeline.project(vertex, joint)

                    _, loss = sess_pipeline.run([train_ops, total_loss],
                                                feed_dict={training: True,
                                                           human2d_input: _input,
                                                           kpt_project: kpt, kpt_gt: _keypoint,
                                                           overlap_loss: overlap_distance(_silhouette, sil)})
                    if i % 100 == 0:
                        print('iteration %d: loss = %f' % (i, loss))
                    if i % 500 == 0:
                        print('save at iteration %d' % i)
                        saver.save(sess_pipeline, 'prior/%s/prior' %
                                   (time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))))

                        vertex, joint = Pipeline.get_mesh(sess_pipeline.run([beta_output, theta_output],
                                                                            feed_dict={training: False,
                                                                                       human2d_input: _input}))
                        vertex_gt, _ = Pipeline.get_mesh([_beta, _theta])
                        Prior.save_mesh(vertex[0], 'visual/out.obj')
                        Prior.save_mesh(vertex_gt[0], 'visual/gt.obj')
                    time.sleep(3)
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')

    @staticmethod
    def test(filenames='', to_quantify=True, to_visualize=True):
        graph_human2d = tf.Graph()
        with graph_human2d.as_default():
            sess_human2d = tf.Session(graph=graph_human2d)
            with sess_human2d.as_default():
                # restore the latest model
                file_list = os.listdir('human2d/model/')
                file_list.sort(key=lambda x: x)
                human2d_path = file_list[-2]
                loader_human2d = tf.train.import_meta_graph('human2d/model/%s/human2d.meta' % human2d_path)

                human2d_input = graph_human2d.get_tensor_by_name('human2d_init/input:0')
                human2d_training = graph_human2d.get_tensor_by_name('human2d_init/training:0')
                silhouette_output = graph_human2d.get_tensor_by_name('human2d_network/silhouette_output:0')
                keypoint_output = graph_human2d.get_tensor_by_name('human2d_network/keypoint_output:0')

        graph_shape = tf.Graph()
        with graph_shape.as_default():
            sess_shape = tf.Session(graph=graph_shape)
            with sess_shape.as_default():
                # restore the latest model
                file_list = os.listdir('shape/model/')
                file_list.sort(key=lambda x: x)
                shape_path = file_list[-2]
                loader_shape = tf.train.import_meta_graph('shape/model/%s/shape.meta' % shape_path)

                shape_input = graph_shape.get_tensor_by_name('shape_prior_init/input:0')
                shape_training = graph_shape.get_tensor_by_name('shape_prior_init/training:0')
                beta_output = graph_shape.get_tensor_by_name('shape_prior_network/beta_output/MatMul:0')

        graph_pose = tf.Graph()
        with graph_pose.as_default():
            sess_pose = tf.Session(graph=graph_pose)
            with sess_pose.as_default():
                # restore the latest model
                file_list = os.listdir('pose/model/')
                file_list.sort(key=lambda x: x)
                pose_path = file_list[-2]
                loader_pose = tf.train.import_meta_graph('pose/model/%s/pose.meta' % pose_path)

                pose_input = graph_pose.get_tensor_by_name('pose_prior_init/input:0')
                pose_training = graph_pose.get_tensor_by_name('pose_prior_init/training:0')
                theta_output = graph_pose.get_tensor_by_name('pose_prior_network/theta_output/MatMul:0')

        # configuration
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.allow_soft_placement = True
        config.log_device_placement = False

        with tf.Session(config=config) as sess_prior:
            _vertex = tf.placeholder(tf.float32, [None, 6890, 3], name='vertex')
            _vertex_gt = tf.placeholder(tf.float32, [None, 6890, 3], name='vertex_gt')

            # loss function
            total_loss = tf.reduce_mean(tf.reduce_sum(tf.square(_vertex - _vertex_gt), axis=[1, 2]), name='vertex_loss')

            # dataset
            data = Reader(filenames=filenames, image_size=256, batch_size=1, num_epochs=1).feed(shuffle=True)

            # initialization
            sess_prior.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
            loader_human2d.restore(sess_human2d, tf.train.latest_checkpoint('human2d/model/%s' % human2d_path))
            loader_shape.restore(sess_shape, tf.train.latest_checkpoint('shape/model/%s' % shape_path))
            loader_pose.restore(sess_pose, tf.train.latest_checkpoint('pose/model/%s' % pose_path))

            index = 0
            v_loss = 0
            try:
                while True:
                    _input, _, _, _theta, _beta = sess_prior.run(data)

                    with sess_human2d.as_default():
                        _silhouette, _keypoint = sess_human2d.run([silhouette_output, keypoint_output],
                                                                  feed_dict={human2d_input: _input,
                                                                             human2d_training: False})
                        _silhouette = Pipeline.generate_silhouette(_silhouette)
                        _keypoint = Pipeline.generate_keypoint(_keypoint)
                    with sess_shape.as_default():
                        _beta_output = sess_shape.run(beta_output, feed_dict={shape_input: _silhouette,
                                                                              shape_training: False})
                    with sess_pose.as_default():
                        _theta_output = sess_pose.run(theta_output, feed_dict={pose_input: _keypoint,
                                                                               pose_training: False}) / 500
                    vertex, _ = Prior.get_mesh([_beta_output, _theta_output])
                    vertex_gt, _ = Prior.get_mesh([_beta, _theta])

                    loss = sess_prior.run(total_loss, feed_dict={_vertex: vertex, _vertex_gt: vertex_gt})
                    v_loss = v_loss + loss
                    if to_quantify:
                        print('vertex loss = %f' % loss)
                    if to_visualize:
                        cv.imwrite('pipeline/result/input_%05d.jpg' % index, _input[0, :, :, ::-1])
                        Prior.save_mesh(_beta_output[0], _theta_output[0], 'pipeline/result/out_%05d.obj' % index)
                    index = index + 1
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
                print('vertex loss = %f' % (v_loss / index))

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

    @classmethod
    def get_mesh(cls, param):
        """ Generates a 3D mesh from a set of beta

            Args:
                param: A list containing beta parameters and theta parameters, in which,
                       beta is a set of beta parameters (with shape of [None, 10]) and
                       theta is a set of theta parameters (with shape of [None, 72])

            Returns:
                vertex: The vertices of SMPL meshes
                joint: The joints of SMPL messsshesss
        """
        assert len(param) == 2
        assert param[0].shape[0] == param[1].shape[0]

        batch_size = param[0].shape[0]

        vertex = np.zeros(shape=[batch_size, 6890, 3])
        joint = np.zeros(shape=[batch_size, 24, 3])

        for i in range(batch_size):
            Prior.model_template.betas[:] = param[0][i]
            Prior.model_template.pose[:] = param[1][i]
            vertex[i] = Prior.model_template.r
            joint[i] = Prior.model_template.J

        return vertex, joint

    @classmethod
    def save_mesh(cls, mesh, filename=''):
        with open(filename, 'w') as fp:
            for v in mesh.r:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

            for f in mesh.f + 1:  # Faces are 1-based, not 0-based in obj files
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    @classmethod
    def project(cls, vertex, joint):
        return vertex, joint


if __name__ == '__main__':
    old_v = tf.logging.get_verbosity()
    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.logging.set_verbosity(old_v)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    np.set_printoptions(threshold=np.inf)

    Pipeline.test(filenames='/home/laukyuen3/thesis/fashionpose.tfrecords', to_visualize=False, to_quantify=True)
