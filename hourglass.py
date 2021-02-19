# -*- coding: utf-8 -*-

"""
Deep Human Pose Estimation

Project by Walid Benbihi
MSc Individual Project
Imperial College
Created on Mon Jul 10 19:13:56 2017

@author: Walid Benbihi
@mail : w.benbihi(at)gmail.com
@github : https://github.com/wbenbihi/hourglasstensorlfow/

Abstract:
    This python code creates a Stacked Hourglass Model
    (Credits : A.Newell et al.)
    (Paper : https://arxiv.org/abs/1603.06937)

    Code translated from 'anewell' github
    Torch7(LUA) --> TensorFlow(PYTHON)
    (Code : https://github.com/anewell/pose-hg-train)

    Modification are made and explained in the report
    Goal : Achieve Real Time detection (Webcam)
    ----- Modifications made to obtain faster results (trade off speed/accuracy)

    This work is free of use, please cite the author if you use it!
"""

import tensorflow as tf
import numpy as np


def _graph_hourglass(inputs, nFeat, nStack, nLow, outDim=16, dropout_rate=0.2, training=True, tiny=False):
    """Create the Network
    Args:
        inputs : TF Tensor (placeholder) of shape (None, 256, 256, 3) # TODO : Create a parameter for customize size
    """
    with tf.name_scope('model'):
        """
        with tf.name_scope('preprocessing'):
            # Input Dim : nbImages x 256 x 256 x 3
            pad1 = tf.pad(inputs, [[0, 0], [2, 2], [2, 2], [0, 0]], name='pad_1')
            # Dim pad1 : nbImages x 260 x 260 x 3
            conv1 = _conv_bn_relu(pad1, filters=64, kernel_size=6, strides=2, name='conv_256_to_128')
            # Dim conv1 : nbImages x 128 x 128 x 64
            r1 = _residual(conv1, numOut=128, name='r1')
            # Dim pad1 : nbImages x 128 x 128 x 128
            pool1 = tf.contrib.layers.max_pool2d(r1, [2, 2], [2, 2], padding='VALID')
            # Dim pool1 : nbImages x 64 x 64 x 128
            if tiny:
                r3 = _residual(pool1, numOut=nFeat, name='r3')
            else:
                r2 = _residual(pool1, numOut=int(nFeat / 2), name='r2')
                r3 = _residual(r2, numOut=nFeat, name='r3')
        """
        # The code commented above is used for regular procedure (with initialization),
        # however, in Human2D, we should drop this operation (to replace by the following statement).
        r3 = inputs
        # Storage Table
        hg = [None] * nStack
        ll = [None] * nStack
        ll_ = [None] * nStack
        drop = [None] * nStack
        out = [None] * nStack
        out_ = [None] * nStack
        sum_ = [None] * nStack
        if tiny:
            with tf.name_scope('stacks'):
                with tf.name_scope('stage_0'):
                    hg[0] = _hourglass(r3, nLow, nFeat, 'hourglass')
                    drop[0] = tf.layers.dropout(hg[0], rate=dropout_rate, training=training, name='dropout')
                    ll[0] = _conv_bn_relu(drop[0], nFeat, 1, 1, name='ll')
                    out[0] = _conv(ll[0], outDim, 1, 1, 'VALID', 'out')
                    out_[0] = _conv(out[0], nFeat, 1, 1, 'VALID', 'out_')
                    sum_[0] = tf.add_n([out_[0], ll[0], r3], name='merge')
                for i in range(1, nStack - 1):
                    with tf.name_scope('stage_' + str(i)):
                        hg[i] = _hourglass(sum_[i - 1], nLow, nFeat, 'hourglass')
                        drop[i] = tf.layers.dropout(hg[i], rate=dropout_rate, training=training, name='dropout')
                        ll[i] = _conv_bn_relu(drop[i], nFeat, 1, 1, name='ll')
                        out[i] = _conv(ll[i], outDim, 1, 1, 'VALID', 'out')
                        out_[i] = _conv(out[i], nFeat, 1, 1, 'VALID', 'out_')
                        sum_[i] = tf.add_n([out_[i], ll[i], sum_[i - 1]], name='merge')
                with tf.name_scope('stage_' + str(nStack - 1)):
                    hg[nStack - 1] = _hourglass(sum_[nStack - 2], nLow, nFeat, 'hourglass')
                    drop[nStack - 1] = tf.layers.dropout(hg[nStack - 1], rate=dropout_rate, training=training,
                                                         name='dropout')
                    ll[nStack - 1] = _conv_bn_relu(drop[nStack - 1], nFeat, 1, 1, 'VALID', 'conv')
                    out[nStack - 1] = _conv(ll[nStack - 1], outDim, 1, 1, 'VALID', 'out')
            """
            return tf.stack(out, axis=1, name='final_output')
            """
            # The code commented above is used for regular procedure (to obtain serveral heatmaps),
            # however, in Human2D, we expect an output with channel of nFeat.
            return ll[nStack - 1]
        else:
            with tf.name_scope('stacks'):
                with tf.name_scope('stage_0'):
                    hg[0] = _hourglass(r3, nLow, nFeat, 'hourglass')
                    drop[0] = tf.layers.dropout(hg[0], rate=dropout_rate, training=training, name='dropout')
                    ll[0] = _conv_bn_relu(drop[0], nFeat, 1, 1, 'VALID', name='conv')
                    ll_[0] = _conv(ll[0], nFeat, 1, 1, 'VALID', 'll')
                    out[0] = _conv(ll[0], outDim, 1, 1, 'VALID', 'out')
                    out_[0] = _conv(out[0], nFeat, 1, 1, 'VALID', 'out_')
                    sum_[0] = tf.add_n([out_[0], r3, ll_[0]], name='merge')
                for i in range(1, nStack - 1):
                    with tf.name_scope('stage_' + str(i)):
                        hg[i] = _hourglass(sum_[i - 1], nLow, nFeat, 'hourglass')
                        drop[i] = tf.layers.dropout(hg[i], rate=dropout_rate, training=training, name='dropout')
                        ll[i] = _conv_bn_relu(drop[i], nFeat, 1, 1, 'VALID', name='conv')
                        ll_[i] = _conv(ll[i], nFeat, 1, 1, 'VALID', 'll')
                        out[i] = _conv(ll[i], outDim, 1, 1, 'VALID', 'out')
                        out_[i] = _conv(out[i], nFeat, 1, 1, 'VALID', 'out_')
                        sum_[i] = tf.add_n([out_[i], sum_[i - 1], ll_[0]], name='merge')
                with tf.name_scope('stage_' + str(nStack - 1)):
                    hg[nStack - 1] = _hourglass(sum_[nStack - 2], nLow, nFeat, 'hourglass')
                    drop[nStack - 1] = tf.layers.dropout(hg[nStack - 1], rate=dropout_rate, training=training,
                                                         name='dropout')
                    ll[nStack - 1] = _conv_bn_relu(drop[nStack - 1], nFeat, 1, 1, 'VALID', 'conv')
                    out[nStack - 1] = _conv(ll[nStack - 1], outDim, 1, 1, 'VALID', 'out')
            """
            return tf.stack(out, axis=1, name='final_output')
            """
            # The code commented above is used for regular procedure (to obtain serveral heatmaps),
            # however, in Human2D, we expect an output with channel of nFeat.
            return ll[nStack - 1]


def _conv(inputs, filters, kernel_size=1, strides=1, pad='VALID', name='conv'):
    """ Spatial Convolution (CONV2D)
    Args:
        inputs      : Input Tensor (Data Type : NHWC)
        filters	    : Number of filters (channels)
        kernel_size	: Size of kernel
        strides		: Stride
        pad			: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
        name		: Name of the block
    Returns:
        conv		: Output Tensor (Convolved Input)
    """
    with tf.name_scope(name):
        # Kernel for convolution, Xavier Initialisation
        kernel = tf.Variable(tf.contrib.layers.xavier_initializer
                             (uniform=False)([kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters]),
                             name='weights')
        conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], padding=pad, data_format='NHWC')
        return conv


def _conv_bn_relu(inputs, filters, kernel_size=1, strides=1, pad='VALID', name='conv_bn_relu', training=True):
    """ Spatial Convolution (CONV2D) + BatchNormalization + ReLU Activation
    Args:
        inputs		: Input Tensor (Data Type : NHWC)
        filters		: Number of filters (channels)
        kernel_size	: Size of kernel
        strides		: Stride
        pad			: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
        name		: Name of the block
    Returns:
        norm		: Output Tensor
    """
    with tf.name_scope(name):
        kernel = tf.Variable(tf.contrib.layers.xavier_initializer
                             (uniform=False)([kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters]),
                             name='weights')
        conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], padding=pad, data_format='NHWC')
        norm = tf.contrib.layers.batch_norm(conv, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu, is_training=training)
        return norm


def _conv_block(inputs, numOut, name='conv_block', training=True, tiny=False):
    """ Convolutional Block
    Args:
        inputs	: Input Tensor
        numOut	: Desired output number of channel
        name	: Name of the block
    Returns:
        conv_3	: Output Tensor
    """
    if tiny:
        with tf.name_scope(name):
            norm = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
                                                is_training=training)
            pad = tf.pad(norm, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name='pad')
            conv = _conv(pad, int(numOut), kernel_size=3, strides=1, pad='VALID', name='conv')
            return conv
    else:
        with tf.name_scope(name):
            with tf.name_scope('norm_1'):
                norm_1 = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
                                                      is_training=training)
                conv_1 = _conv(norm_1, int(numOut / 2), kernel_size=1, strides=1, pad='VALID', name='conv')
            with tf.name_scope('norm_2'):
                norm_2 = tf.contrib.layers.batch_norm(conv_1, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
                                                      is_training=training)
                pad = tf.pad(norm_2, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name='pad')
                conv_2 = _conv(pad, int(numOut / 2), kernel_size=3, strides=1, pad='VALID', name='conv')
            with tf.name_scope('norm_3'):
                norm_3 = tf.contrib.layers.batch_norm(conv_2, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
                                                      is_training=training)
                conv_3 = _conv(norm_3, int(numOut), kernel_size=1, strides=1, pad='VALID', name='conv')
            return conv_3


def _skip_layer(inputs, numOut, name='skip_layer'):
    """ Skip Layer
    Args:
        inputs	: Input Tensor
        numOut	: Desired output number of channel
        name	: Name of the bloc
    Returns:
        Tensor of shape (None, inputs.height, inputs.width, numOut)
    """
    with tf.name_scope(name):
        if inputs.get_shape().as_list()[3] == numOut:
            return inputs
        else:
            conv = _conv(inputs, numOut, kernel_size=1, strides=1, name='conv')
            return conv


def _residual(inputs, numOut, training=True, tiny=True, name='residual_block'):
    """ Residual Unit
    Args:
        inputs  : Input Tensor
        numOut  : Number of Output Features (channels)
        name    : Name of the block
    """
    with tf.name_scope(name):
        convb = _conv_block(inputs, numOut, training=training, tiny=tiny)
        skipl = _skip_layer(inputs, numOut)
        return tf.add_n([convb, skipl], name='res_block')


def _hourglass(inputs, n, numOut, training=True, tiny=True, name='hourglass'):
    """ Hourglass Module
    Args:
        inputs	: Input Tensor
        n		: Number of downsampling step
        numOut	: Number of Output Features (channels)
        name	: Name of the block
    """
    with tf.name_scope(name):
        # Upper Branch
        up_1 = _residual(inputs, numOut, training, tiny, name='up_1')
        # Lower Branch
        low_ = tf.contrib.layers.max_pool2d(inputs, [2, 2], [2, 2], padding='VALID')
        low_1 = _residual(low_, numOut, training, tiny, name='low_1')

        if n > 0:
            low_2 = _hourglass(low_1, n - 1, numOut, training, tiny, name='low_2')
        else:
            low_2 = _residual(low_1, numOut, training, tiny, name='low_2')

        low_3 = _residual(low_2, numOut, training, tiny, name='low_3')
        up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(low_3)[1:3] * 2, name='upsampling')
        return tf.add_n([up_2, up_1], name='out_hg')
