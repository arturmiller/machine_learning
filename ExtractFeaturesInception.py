'''
Created on 28.08.2016

@author: Artur
'''
import os
import pickle

import numpy as np
from skimage.io import imread
from skimage.transform import resize
import pandas as pd
import lasagne
from modelzoo.inception_v3 import build_network as inception_model

if __name__ == '__main__':
    model = inception_model()
    model_path = os.path.join(os.environ['MODELS'], r'TrainedNetworks\inception_v3.pkl')
    dataset_path = os.path.join(os.environ['DATASETS'], r'ktosexy\dataset.csv')
    features_path = os.path.join(os.environ['DATASETS'], r'ktosexy\features\inception_v3.npy')

    dataset = pd.read_csv(dataset_path, sep=';')

    with open(model_path, 'rb') as f:
        pickled_model = pickle.load(f, encoding='latin-1')

    lasagne.layers.set_all_param_values(model['softmax'], pickled_model['param values'])

    num_features = model['pool3'].output_shape[1]
    num_images = len(dataset)
    features = np.zeros((num_images, num_features))

    print('start feature extraction...')
    print('{:d}/{:d}'.format(0, num_images))
    for tmp_dataset in dataset.iterrows():
        index = tmp_dataset[0]

        image_path = tmp_dataset[1][1]
        gender = tmp_dataset[1][3]

        image_data = imread(image_path)
        image_data_resized = resize(image_data, (299, 299, 3))
        image_data_resized = np.rollaxis(image_data_resized, axis=2)
        image_data_resized = image_data_resized.reshape((1, 3, 299, 299))

        features[index, :] = np.array(lasagne.layers.get_output(model['pool3'], image_data_resized, deterministic=True).eval())
        print('{:d}/{:d}'.format(index+1, num_images))

    print('write to file...')
    np.save(features_path, features)

    print('finished')
