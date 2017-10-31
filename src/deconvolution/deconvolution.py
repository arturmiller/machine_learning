from __future__ import absolute_import, division, print_function, unicode_literals

'''
Created on 28.10.2017

@author: amiller
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage import color, data
from skimage.transform import rescale
import scipy.ndimage.filters as fi


def gaussian(kernlen, nsig_x, nsig_y):
    inp = np.zeros((kernlen, kernlen))
    inp[kernlen//2, kernlen//2] = 1
    return fi.gaussian_filter(inp, (nsig_x, nsig_y))


def tf_convolve_2d(image, psf):
    tf_image = tf.reshape(image, (1, tf.shape(image)[0], tf.shape(image)[1], 1))
    tf_psf = tf.reshape(psf, (tf.shape(psf)[0], tf.shape(psf)[1], 1, 1))
    tf_output = tf.nn.conv2d(tf_image, tf_psf, [1, 1, 1, 1], 'SAME')[0, :, :, 0]
    return tf_output


def convolve(image, psf):
    sess = tf.Session()
    tf_image = tf.constant(image, dtype=tf.float32)
    tf_psf = tf.constant(psf, dtype=tf.float32)
    tf_output = tf_convolve_2d(tf_image, tf_psf)
    output = sess.run(tf_output)
    return output


def deconvolve(image, psf, learning_rate=1e-2):
    loss_history = []

    tf_input = tf.Variable(np.random.randn(*image.shape) / 10.0, dtype=tf.float32)
    tf_psf = tf.constant(psf, dtype=tf.float32)
    tf_gt_output = tf.constant(image, dtype=tf.float32)
    tf_output = tf_convolve_2d(tf_input, tf_psf)

    loss = tf.reduce_sum(tf.square(tf_output - tf_gt_output))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    train = optimizer.minimize(loss)

    # training loop
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    def eval_step():
        tmp_loss = sess.run(loss)
        loss_history.append(tmp_loss)
        print('loss: {}'.format(tmp_loss))

    eval_step()
    for _ in range(500):
        sess.run(train)
        eval_step()
    res_input = sess.run(tf_input)

    return res_input, loss_history


if __name__ == '__main__':
    astro_org = rescale(color.rgb2gray(data.astronaut()), 1.0/3.0)

    psf = gaussian(kernlen=13, nsig_x=1, nsig_y=3)
    psf /= np.sum(psf)

    astro_conv = convolve(astro_org, psf)
    plt.imshow(astro_conv)
    plt.show()

    astro_deconv, loss = deconvolve(astro_conv, psf)
    plt.plot(loss)
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 5))
    plt.gray()

    for a in ax:
        a.axis('off')

    ax[0].imshow(astro_org, vmin=astro_org.min(), vmax=astro_org.max())
    ax[0].set_title('Original data')

    ax[1].imshow(astro_conv, vmin=astro_org.min(), vmax=astro_org.max())
    ax[1].set_title('Convolved data')

    ax[2].imshow(astro_deconv, vmin=astro_org.min(), vmax=astro_org.max())
    ax[2].set_title('Deconvolved data')

    fig.subplots_adjust(wspace=0.02, hspace=0.2,
                        top=0.9, bottom=0.05, left=0, right=1)
    plt.show()
