from __future__ import absolute_import, division, print_function, unicode_literals

'''
Created on 09.04.2018

@author: amiller
'''

from functools import partial

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


class LossNet2(object):
    def __init__(self, tf_gen_model, num_latent_params, learning_rate=5e-4, seed=42, iter_inner_loop=500, iter_outer_loop=50, print_loss_number=20):
        self.tf_gen_model = tf_gen_model
        self.num_latent_params = num_latent_params
        self.learning_rate = learning_rate

        np.random.seed(seed)
        self.loss_history = []
        self.iter_inner_loop = iter_inner_loop
        self.iter_outer_loop = iter_outer_loop
        self.print_loss_number = print_loss_number

    def tf_iter_latent_params(self, X, latent_params, model_params):
        for i in range(self.iter_inner_loop):
            latent_params = self.tf_gen_model(latent_params, model_params, X)
        return latent_params

    def tf_calc_outer_loss(self, X, y, latent_params, model_params, output_model):
        latent_params = self.tf_iter_latent_params(X, latent_params, model_params)
        output_params = tf.matmul(output_model, latent_params)
        #loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.transpose(y), logits=tf.transpose(latent_params))
        loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=tf.transpose(output_params))
        #loss = tf.sqrt(tf.nn.l2_loss(latent_params - y))
        return loss

    def calc_outer_loss(self, X, y, latent_params, model_params):
        tf_X = tf.constant(X, dtype=tf.float32)
        tf_y = tf.constant(y, dtype=tf.float32)
        tf_latent_params = tf.constant(latent_params, dtype=tf.float32)
        tf_model_params = tf.constant(model_params, dtype=tf.float32)
        tf_loss = self.tf_calc_outer_loss(tf_X, tf_y, tf_latent_params, tf_model_params)

        sess = tf.Session()
        loss = sess.run(tf_loss)
        return loss

    def fit(self, X, y):
        latent_shape = (self.num_latent_params, np.size(X, axis=0))
        tf_latent_params = tf.constant(np.abs(np.random.randn(*latent_shape)) / 100.0, dtype=tf.float32)
        tf_y = tf.constant(y, dtype=tf.float32)

        #positive = partial(tf.clip_by_value, clip_value_min=0.0, clip_value_max=np.inf)

        model_shape = (np.size(X, axis=1), self.num_latent_params)
        tf_model_params = tf.Variable(np.abs(np.random.randn(*model_shape)) / 100.0, dtype=tf.float32)#, constraint=positive)

        output_shape = (np.size(y, axis=1), self.num_latent_params)
        tf_output_model = tf.Variable(np.abs(np.random.randn(*output_shape)) / 100.0, dtype=tf.float32)
        tf_X = tf.constant(X, dtype=tf.float32)
        L2 = self.tf_calc_outer_loss(tf_X, tf_y, tf_latent_params, tf_model_params, tf_output_model)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        train = optimizer.minimize(L2)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        def eval_step(i=0):
            tmp_loss = sess.run(L2)
            self.loss_history.append(tmp_loss)
            if i % self.print_loss_number == 0:
                print('index: {} loss: {}'.format(i, tmp_loss))

        for i in range(self.iter_outer_loop):
            eval_step(i)
            sess.run(train)

        self.model_params = sess.run(tf_model_params)
        self.output_model = sess.run(tf_output_model)

    def predict(self, X):
        tf_X = tf.constant(X, dtype=tf.float32)
        latent_shape = (self.num_latent_params, np.size(X, axis=0))
        tf_latent_params = tf.constant(np.abs(np.random.randn(*latent_shape)) / 100.0, dtype=tf.float32)
        tf_model_params = tf.constant(self.model_params, dtype=tf.float32)
        tf_res_latent_params = self.tf_iter_latent_params(tf_X, tf_latent_params, tf_model_params)
        tf_output_model = tf.constant(self.output_model, dtype=tf.float32)
        tf_output_params = tf.matmul(tf_output_model, tf_res_latent_params)
        tf_label = tf.argmax(tf_output_params, axis=0)
        sess = tf.Session()
        label = sess.run(tf_label)
        return label

if __name__ == '__main__':
    from sklearn import datasets
    import matplotlib.pyplot as plt

    digits = datasets.load_digits(10)
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    import numpy as np
    import tensorflow as tf

    def tf_calc_loss(output, gt_output):
        loss = tf.sqrt(tf.nn.l2_loss(output - gt_output))
        return loss

    def tf_gen_model_grad(latent_params, model_params, X):
        tf_output = tf.transpose(tf.matmul(model_params, latent_params))
        loss = tf_calc_loss(tf_output, X)
        return tf.gradients(loss, [latent_params])[0]

    def tf_gen_model(latent_params, model_params, X):
        next_latent_params = latent_params - tf.multiply(0.01, tf_gen_model_grad(latent_params, model_params, X))
        return next_latent_params

    def one_hot(vec, depth):
        length = np.size(vec)
        arr = np.zeros((length, depth))
        arr[np.arange(length), vec] = 1
        return arr

    latent_params = 20
    num_train = 1000

    results = []
    #learning_rates = [1e0, 3e0, 1e1, 3e1, 1e2, 3e2, 1e2, 3e2]
    learning_rates = [1e1,]
    for learning_rate in learning_rates:
        loss_net = LossNet2(tf_gen_model, latent_params, learning_rate=learning_rate, seed=42, iter_inner_loop=250, iter_outer_loop=5000, print_loss_number=10)
        loss_net.fit(data[:num_train, :], one_hot(digits.target[:num_train], 10))
        predicted = loss_net.predict(data[1000:1500, :])
        results.append(np.sum(digits.target[1000:1500] == predicted))
        print(results[-1]) #best 97%

    plt.plot(learning_rates, results)
    #for i in range(10):
    #    plt.figure()
    #    plt.imshow(loss_net.model_params[:, i].reshape((8, 8)))
    #for i in range(5):
    #    plt.figure()
    #    plt.imshow(data[i, :].reshape((8, 8)))
    plt.show()
    #index: 4990 loss: 0.3917354643344879