# -*- coding: utf-8 -*-

import tensorflow as tf
import resnet
import hourglass


_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def batch_norm(x, axis=-1, training=True):
    return tf.layers.batch_normalization(
        inputs=x, axis=axis,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=training, fused=True)


def max_pool_layer(x, pool_size=3, strides=2, name=''):
    with tf.variable_scope(name_or_scope=name):
        return tf.layers.max_pooling2d(x, pool_size=pool_size, strides=strides,
                                       padding='SAME', data_format='channels_last', name='output')


def res_layer(x, filters, strides, kernel_size, training=True, bottleneck=True, name=''):
    with tf.variable_scope(name_or_scope=name):
        if bottleneck:
            block_fn = resnet._bottleneck_block_v2
        else:
            block_fn = resnet._building_block_v2
        return resnet.block_layer(x, filters=filters, bottleneck=bottleneck,
                                  block_fn=block_fn, blocks=1,
                                  strides=strides, training=training, data_format='channels_last',
                                  kernel_size=kernel_size, name='output')


def hourglass_layer(x, training=True, tiny=True, name=''):
    with tf.variable_scope(name_or_scope=name):
        return hourglass._hourglass(x, n=3, numOut=256, training=training, tiny=tiny)


def fc_layer(x, units, training=True, dropout=True, name=''):
    with tf.variable_scope(name_or_scope=name):
        inputs = tf.layers.dense(x, units=units, activation=None, use_bias=False)
        inputs = tf.nn.relu(batch_norm(inputs, training=training))
        if dropout:
            return tf.layers.dropout(inputs, rate=0.25, training=training, name='output')
        else:
            return inputs


def bilinear_layer(x, units, training=True, name=''):
    with tf.variable_scope(name_or_scope=name):
        shortcut = x
        inputs = fc_layer(x, units, training=training, name='fc_0')
        inputs = fc_layer(inputs, units, training=training, name='fc_1')
        return tf.add_n([inputs, shortcut], name='output')


def squared_l2_norm(ref, val):
    return tf.square(tf.clip_by_value(ref, 0., 1.) - tf.clip_by_value(val, 0., 1.))


def kl_divergence(p_prop, q_prop):
    return tf.clip_by_value(p_prop, 1e-1, 1.) * \
           tf.abs(tf.log(tf.clip_by_value(p_prop, 1e-10, 1.)) - tf.log(tf.clip_by_value(q_prop, 1e-10, 1.)))


def cross_entropy(label, prop):
    label = tf.one_hot(tf.cast(label / 255, dtype=tf.int32), 2, 1, 0)
    label = tf.cast(tf.reshape(label, shape=[-1, 64, 64, 2]), dtype=tf.float32)
    return -(label * tf.log(tf.clip_by_value(prop, 1e-10, 1.)) +
             (1 - label) * tf.log(tf.clip_by_value(1. - prop, 1e-10, 1.)))


def overlap_distance(ref, val):
    assert ref.shape == val.shape
    # prevent over-coverage
    for b in range(val.shape[0]):
        for x in range(val.shape[1]):
            for y in range(val.shape[2]):
                pass
    # prevent under-coverage

    pass
