'''
Created on 26.11.2016

@author: Artur
'''
import pickle
from bson.binary import Binary
import numpy as np
from pymongo import MongoClient
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.ndimage.filters import gaussian_filter1d


TRAIN_COLLECTION_NAME = 'train_collection'
TEST_COLLECTION_NAME = 'test_collection'
SOURCE = 'source'
TARGET = 'target'


class SmoothFunctionCreator():
    def __init__(self, seed=42):
        self._gp = GaussianProcessRegressor()
        x_train = np.array([0.0, 2.0, 6.0, 10.0])[:, np.newaxis]
        source_train = np.array([0.0, 1.0, -1.0, 0.0])
        self._gp.fit(x_train, source_train)
        self._random_state = np.random.RandomState(seed)

    def sample(self, n_samples):
        x = np.linspace(0.0, 10.0, 100)[:, np.newaxis]
        source = self._gp.sample_y(x, n_samples, random_state=self._random_state)
        target = gaussian_filter1d(source, 1, order=1, axis=0)
        target = np.tanh(10.0 * target)
        return source, target


def create_collection(collection_name, samples):
    client = MongoClient()
    db = client[collection_name]
    db.drop_collection('collection')
    collection = db.collection

    creator = SmoothFunctionCreator()

    for _ in range(samples):
        y_original, y_transformed = creator.sample(1)
        collection.insert({SOURCE: Binary(pickle.dumps(y_original, protocol=2)),
                           TARGET: Binary(pickle.dumps(y_transformed, protocol=2))})

    print(len(list(collection.find())))


if __name__ == '__main__':
    create_collection(TRAIN_COLLECTION_NAME, 1000)
    create_collection(TEST_COLLECTION_NAME, 200)
