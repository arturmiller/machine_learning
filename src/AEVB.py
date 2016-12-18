'''
Created on 06.08.2016

@author: Artur
'''

import random

import numpy as np
import theano
import theano.tensor as T

theano.config.optimizer = 'fast_compile'
theano.config.exception_verbosity = 'high'
theano.config.compute_test_value = 'warn'
theano.config.NanGuardMode.nan_is_error = True
theano.config.floatX = 'float64'


class HiddenLayer(object):
    def __init__(self, rv_input, n_in, n_out, W=None, b=None,
                 activation=T.tanh, name=''):
        if W is None:
            W_values = np.asarray(
                np.random.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX  # @UndefinedVariable
            )

            W = theano.shared(value=W_values, name='W_' + name, borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)  # @UndefinedVariable
            b = theano.shared(value=b_values, name='b_' + name, borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(rv_input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


class Encoder(object):

    def __init__(self, x, n_points, n_hidden, n_features):
        self.hiddenLayer = HiddenLayer(
            rv_input=x,
            n_in=n_points,
            n_out=n_hidden,
            activation=T.tanh,
            name='encoder_hidden'
        )

        self.mu = HiddenLayer(
            rv_input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_features,
            activation=None,
            name='encoder_mu'
        )

        self.log_var = HiddenLayer(
            rv_input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_features,
            activation=None,
            name='encoder_log_var'
        )

        self.params = self.hiddenLayer.params + self.mu.params + self.log_var.params

        srng = T.shared_randomstreams.RandomStreams()
        eps = srng.normal(self.mu.output.shape)
        self.z = self.mu.output + T.exp(0.5 * self.log_var.output) * eps


class Decoder(object):

    def __init__(self, rv_input, n_in, n_hidden, n_out):
        self.hiddenLayer = HiddenLayer(
            rv_input=rv_input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh,
            name='decoder_hidden'
        )

        self.mu = HiddenLayer(
            rv_input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out,
            activation=None,
            name='decoder_mu'
        )

        self.log_var = HiddenLayer(
            rv_input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out,
            activation=None,
            name='decoder_log_var'
        )

        self.params = self.hiddenLayer.params + self.mu.params + self.log_var.params


class AEVB():
    def __init__(self, n_components=None, iterations=20, learning_rate=1e-3, batch_size=100,
                 n_epochs=10, n_hidden_encoder=30, n_hidden_decoder=30):
        self.n_components = n_components
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_hidden_encoder = n_hidden_encoder
        self.n_hidden_decoder = n_hidden_decoder

    def fit(self, X, y=None):
        self._fit(X)
        return self

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def transform(self, X):
        return self._encode(X)

    def getCurrentElbo(self):
        return np.mean([self.elboFunc(minibatch_index) for minibatch_index in range(self.n_train_batches)])

    def _fit(self, X):
        train_set_x = theano.shared(np.asarray(X, dtype=theano.config.floatX), borrow=True)  # @UndefinedVariable

        n_points = np.size(X, axis=1)  # dim x

        x = T.matrix('x')
        x.tag.test_value = np.random.rand(self.batch_size, n_points)

        encoder = Encoder(x, n_points, self.n_hidden_encoder, self.n_components)
        decoder = Decoder(encoder.z, self.n_components, self.n_hidden_decoder, n_points)
        params = decoder.params + encoder.params

        log_likelihood = -0.5 * T.sum(np.log(2 * np.pi) + decoder.log_var.output +
                                      (x - decoder.mu.output) ** 2 / T.exp(decoder.log_var.output), axis=1)
        elbo = T.sum(0.5 * T.sum(1.0 + encoder.log_var.output - encoder.mu.output ** 2 -
                                 T.exp(encoder.log_var.output), axis=1) + log_likelihood) / x.shape[0]
        grad_elbo = [T.grad(elbo, param) for param in params]

        index = T.lscalar()
        index.tag.test_value = 0

        self.n_train_batches = train_set_x.get_value(borrow=True).shape[0] // self.batch_size

        train_model = theano.function(
            inputs=[index],
            updates=[(param, param + self.learning_rate * gparam) for param, gparam in zip(params, grad_elbo)],
            givens={x: train_set_x[index * self.batch_size: (index + 1) * self.batch_size]}
        )

        self.elbos = []

        self.elboFunc = theano.function(inputs=[index], givens={x: train_set_x[index * self.batch_size: (index + 1) * self.batch_size]},
                                        outputs=elbo)

        for epoch in range(self.n_epochs):
            print('epoch: ' + str(epoch))
            minibatch_indices = list(range(self.n_train_batches))
            random.shuffle(minibatch_indices)
            for minibatch_index in minibatch_indices:
                train_model(minibatch_index)

            self.elbos.append(self.getCurrentElbo())

        sample_z = T.shared_randomstreams.RandomStreams().normal((1, self.n_components))
        self.sample = theano.function(inputs=[], outputs=decoder.mu.output, givens={encoder.z: sample_z})
        self._encode = theano.function(inputs=[x], outputs=encoder.z)


