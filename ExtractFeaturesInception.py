'''
Created on 28.08.2016

@author: Artur
'''

import pickle

import numpy as np
import lasagne
from modelzoo.inception_v3 import build_network as inception_model

if __name__ == '__main__':
    model = inception_model()
    path = r'C:\Users\Artur\Development\TrainedNetworks\inception_v3.pkl'
    with open(path, 'rb') as f:
        pickled_model = pickle.load(f, encoding='latin-1')

    lasagne.layers.set_all_param_values(model['softmax'], pickled_model['param values'])

    im = np.random.randn(1, 3, 299, 299)
    prob = np.array(lasagne.layers.get_output(model['pool3'], im, deterministic=True).eval())

    print(prob)
