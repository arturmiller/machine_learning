from __future__ import absolute_import, division, print_function, unicode_literals

'''
Created on 09.04.2018

@author: amiller
'''

import tensorflow as tf

from src.loss_net.loss_net import LossNet


def tf_convolve_2d(image, psf):
    tf_image = tf.reshape(image, (1, tf.shape(image)[0], tf.shape(image)[1], 1))
    tf_psf = tf.reshape(psf, (tf.shape(psf)[0], tf.shape(psf)[1], 1, 1))
    tf_output = tf.nn.conv2d(tf_image, tf_psf, [1, 1, 1, 1], 'SAME')[0, :, :, 0]
    return tf_output


def calc_loss(tf_output, tf_input, tf_gt_output):
    output_loss = tf.sqrt(tf.nn.l2_loss(tf_output - tf_gt_output))
    regularization = tf.sqrt(tf.nn.l2_loss(tf_input[:-1, :] - tf_input[1:, :]) + tf.nn.l2_loss(tf_input[:, :-1] - tf_input[:, 1:]))
    loss = output_loss + 0.2 * regularization # mu=0.2
    return loss


def tf_gen_model(latent_params, model_params, X):
    tf_output = tf_convolve_2d(latent_params, model_params)
    loss = calc_loss(tf_output, latent_params, X)

    next_latent_params = latent_params - tf.multiply(0.5, tf.gradients(loss, [latent_params]))[0, :, :]
    return next_latent_params


class BlindDeconvolution(LossNet):
    def __init__(self, psf_shape, learning_rate=5e-4, seed=42, iter_inner_loop=5, iter_outer_loop=1):
        LossNet.__init__(self, tf_gen_model, psf_shape, learning_rate=5e-4, seed=42, iter_inner_loop=500, iter_outer_loop=50)
