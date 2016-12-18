'''
Created on 27.11.2016

@author: Artur
'''
import pickle
from functools import partial

import numpy as np
from pymongo import MongoClient
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pylab as plt
from sklearn.base import clone
from sklearn.utils import check_X_y
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils.validation import has_fit_parameter


def _partial_fit_estimator(estimator, X, y, sample_weight=None):
    estimator = clone(estimator)
    if sample_weight is not None:
        estimator.fit(X, y, sample_weight=sample_weight)
    else:
        estimator.fit(X, y)
    return estimator


class PartialMultiOutputRegressor(MultiOutputRegressor):
    def partial_fit(self, X, y, sample_weight=None):
        """ Fit the model to data.
        Fit a separate model for each output variable.

        Parameters
        ----------
        X : (sparse) array-like, shape (n_samples, n_features)
            Data.

        y : (sparse) array-like, shape (n_samples, n_outputs)
            Multi-output targets. An indicator matrix turns on multilabel
            estimation.

        sample_weight : array-like, shape = (n_samples) or None
            Sample weights. If None, then samples are equally weighted.
            Only supported if the underlying regressor supports sample
            weights.

        Returns
        -------
        self : object
            Returns self.
        """

        if not hasattr(self.estimator, "fit"):
            raise ValueError("The base estimator should implement a fit method")

        X, y = check_X_y(X, y,
                         multi_output=True,
                         accept_sparse=True)

        if y.ndim == 1:
            raise ValueError("y must have at least two dimensions for "
                             "multi target regression but has only one.")

        if (sample_weight is not None and
                not has_fit_parameter(self.estimator, 'sample_weight')):
            raise ValueError("Underlying regressor does not support"
                             " sample weights.")

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(delayed(_partial_fit_estimator)(
            self.estimator, X, y[:, i], sample_weight) for i in range(y.shape[1]))
        return self


TRAIN_COLLECTION_NAME = 'train_collection'
TEST_COLLECTION_NAME = 'test_collection'
SOURCE = 'source'
TARGET = 'target'


def plot_functions(y_transformed_test, y_transformed_predicted):
    for index in range(np.size(y_transformed_test, axis=0)):
        plt.plot(y_transformed_predicted[index, :], color='red', label='predicted')
        plt.plot(y_transformed_test[index, :], color='blue', label='ground truth')
        plt.legend(loc='upper center')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.show()


def unpickle_data(doc, first_column, second_column=None):

    feature = pickle.loads(doc[first_column])
    if second_column is None:
        return feature
    else:
        value = pickle.loads(doc[second_column])
        return feature, value


def train_pca_model(collection_name, feature_name, n_components, iterations=100, batch_size=20):
    collection = collection_from_name(collection_name)
    model = IncrementalPCA(n_components=n_components)

    partial_unpickle_data = partial(unpickle_data, feature_name=feature_name)

    for _ in range(iterations):
        feature = map(partial_unpickle_data, collection.aggregate([{'$sample': {'size': batch_size}}]))
        feature = np.hstack(feature).T

        model.partial_fit(feature)

    return model


def train_regression_model(collection_name, iterations=100, batch_size=20, pca_original=None, pca_transformed=None):
    collection = collection_from_name(collection_name)
    regressor = PartialMultiOutputRegressor(SGDRegressor())

    partial_unpickle_data = partial(unpickle_data, feature_name=SOURCE, value_name=TARGET)
    for _ in range(iterations):
        feature, value = zip(*map(partial_unpickle_data, collection.aggregate([{'$sample': {'size': batch_size}}])))
        feature = np.hstack(feature).T
        value = np.hstack(value).T
        if pca_original is not None:
            feature = pca_original.transform(feature)
        if pca_transformed is not None:
            value = pca_transformed.transform(value)

        regressor.partial_fit(feature, value)

    return regressor


def predict_from_doc(doc, regressor, partial_unpickle_data, pca_original=None, pca_transformed=None):
    feature, value_gt = partial_unpickle_data(doc)
    if pca_original is not None:
        feature = pca_original.transform(feature.T)
        value_predict = regressor.predict(feature)
        value_predict = pca_transformed.inverse_transform(value_predict)
    else:
        value_predict = regressor.predict(feature.T)
    #plt.plot(value_predict[0, :], color='blue')
    #plt.plot(value_gt[:, 0], color='red')
    #plt.show()
    return value_predict - value_gt.T


def test_regression_model(regressor, collection_name, pca_original=None, pca_transformed=None):
    collection = collection_from_name(collection_name)

    partial_unpickle_data = partial(unpickle_data, feature_name=SOURCE, value_name=TARGET)
    partial_predict_from_doc = partial(predict_from_doc, regressor=regressor, partial_unpickle_data=partial_unpickle_data,
                                       pca_original=pca_original, pca_transformed=pca_transformed)
    difference = np.array(list(map(partial_predict_from_doc, collection.find())))

    print(np.linalg.norm(difference))


def collection_from_name(collection_name):
    client = MongoClient()
    db = client[collection_name]
    collection = db.collection
    return collection
 
 
def train_model_dr():
    train_collection_name = 'train_collection'
    test_collection_name = 'test_collection'

    pca_original = train_pca_model(train_collection_name, SOURCE, 15, batch_size=20, iterations=100)
    pca_transformed = train_pca_model(train_collection_name, TARGET, 10, batch_size=20, iterations=100)

    regressor = train_regression_model(train_collection_name, pca_original=pca_original, pca_transformed=pca_transformed)
    test_regression_model(regressor, test_collection_name, pca_original=pca_original, pca_transformed=pca_transformed)


def train_model_no_dr():
    train_collection_name = 'train_collection'
    test_collection_name = 'test_collection'
    regressor = train_regression_model(train_collection_name)
    test_regression_model(regressor, test_collection_name)


if __name__ == '__main__':
    train_model_dr()
    train_model_no_dr()
