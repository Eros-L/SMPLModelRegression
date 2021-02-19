from smpl.serialization import load_model
import numpy as np
import tensorflow as tf
import pickle
import cv2 as cv
import os


# with tf.variable_scope(name_or_scope='model_template'):
#     template = load_model('smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
#     # vertex template
#     v_template = tf.Variable(template.v_template, dtype=tf.float32, name='v_template')
#     # shape and pose parameters
#     posedirs = tf.Variable(template.posedirs, dtype=tf.float32, name='posedirs')
#     theta = tf.Variable(template.pose, dtype=tf.float32, name='theta')
#     shapedirs = tf.Variable(template.shapedirs, dtype=tf.float32, name='shapedirs')
#     beta = tf.Variable(template.betas, dtype=tf.float32, name='beta')
#     # joints
#     J_regressor_prior = tf.Variable(template.J_regressor_prior.toarray(), dtype=tf.float32, name='J_regressor_prior')
#     J_regressor = tf.Variable(template.J_regressor.toarray(), dtype=tf.float32, name='J_regressor')
#     J = tf.Variable(template.J, dtype=tf.float32, name='J')
#     # weights
#     weights_prior = tf.Variable(template.weights_prior, dtype=tf.float32, name='weights_prior')
#     weights = tf.Variable(template.weights, dtype=tf.float32, name='weights')
#     # etc
#     kintree_table = tf.Variable(template.kintree_table, dtype=tf.float32, name='kintree_table')
#     f = tf.Variable(template.f, dtype=tf.float32, name='f')

if __name__ == '__main__':
    mesh = pickle.load(open('/Users/Eddie/Desktop/tmp/00000_body.pkl', 'rb'), encoding='iso-8859-1')

    template = load_model('smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
    template.pose = mesh['pose']
    template.betas = mesh['betas']

    with open('/Users/Eddie/Desktop/test.obj', 'w') as fp:
        for v in template.r:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        for f in template.f + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


    # def canny(input_batch_image):
    #     #   压缩channel
    #     squeeze_input = np.squeeze(input_batch_image, axis=-1)
    #     #   提取边缘图像
    #     canny_list = []
    #     for i in range(squeeze_input.shape[0]):
    #         canny_img = cv.Canny(squeeze_input[0], 0, 100)
    #         canny_img = np.expand_dims(canny_img, axis=0)
    #         canny_list.append(canny_img)
    #     #   转化矩阵
    #     canny_output = np.concatenate(canny_list, axis=0)
    #     canny_output = np.expand_dims(canny_output, axis=-1)
    #
    #     return canny_output

    # files = os.listdir('/Users/Eddie/Desktop/dfn/data/train_labels/')
    #
    # for filename in files:
    #     img = cv.imread('/Users/Eddie/Desktop/dfn/data/train_labels/%s' % filename)
    #     edge = cv.Canny(img, 0, 1)
    #     cv.imwrite('/Users/Eddie/Desktop/dfn/data/edge/%s' % filename, edge)
