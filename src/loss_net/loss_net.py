from __future__ import absolute_import, division, print_function, unicode_literals

'''
Created on 09.04.2018

@author: amiller
'''

import numpy as np
import tensorflow as tf


class LossNet(object):
    def __init__(self, tf_gen_model, psf_shape, learning_rate=5e-4, seed=42, iter_inner_loop=500, iter_outer_loop=50):
        self.tf_gen_model = tf_gen_model
        self.learning_rate = learning_rate
        self.psf_shape = psf_shape

        np.random.seed(seed)
        self.loss_history = []
        self.iter_inner_loop = iter_inner_loop
        self.iter_outer_loop = iter_outer_loop

    def fit(self, X, y):
        tf_latent_params = tf.constant(np.random.randn(*X.shape) / 10.0, dtype=tf.float32)
        tf_y = tf.constant(y, dtype=tf.float32)

        tf_model_params = tf.Variable(np.zeros(self.psf_shape), dtype=tf.float32)
        tf_X = tf.constant(X, dtype=tf.float32)

        for i in range(self.iter_inner_loop):
            tf_latent_params = self.tf_gen_model(tf_latent_params, tf_model_params, tf_X)

        L2 = tf.sqrt(tf.nn.l2_loss(tf_latent_params - tf_y))

        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        train = optimizer.minimize(L2)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        def eval_step(i=0):
            tmp_loss = sess.run(L2)
            self.loss_history.append(tmp_loss)
            if i % 1 == 0:
                print('loss: {}'.format(tmp_loss))

        for i in range(self.iter_outer_loop):
            eval_step(i)
            sess.run(train)

        res_model_params = sess.run(tf_model_params)

        return res_model_params

    def predict(self, X):
        pass
