'''
Created on 27.11.2016

@author: Artur
'''

import pickle

import numpy as np
from pymongo import MongoClient
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt


def plot_functions(y_transformed_test, y_transformed_predicted):
    for index in range(np.size(y_transformed_test, axis=0)):
        plt.plot(y_transformed_predicted[index, :], color='red', label='predicted')
        plt.plot(y_transformed_test[index, :], color='blue', label='ground truth')
        plt.legend(loc='upper center')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.show()


def calc_error(y_transformed_predicted, y_transformed_test):
    error = np.mean(np.sqrt(np.mean((y_transformed_predicted - y_transformed_test)**2, axis=1)))
    return error


def train_model_dr(y_original_train, y_original_test, y_transformed_train, y_transformed_test):
    pca_original = PCA(15)
    pca_original.fit(y_original_train)
    pca_transformed = PCA(10)
    pca_transformed.fit(y_transformed_train)

    y_original_reduced_train = pca_original.transform(y_original_train)
    y_original_reduced_test = pca_original.transform(y_original_test)
    y_transformed_reduced_train = pca_transformed.transform(y_transformed_train)

    regression_reduced = LinearRegression()
    regression_reduced.fit(y_original_reduced_train, y_transformed_reduced_train)
    y_transformed_reduced_predicted = regression_reduced.predict(y_original_reduced_test)
    y_transformed_predicted = pca_transformed.inverse_transform(y_transformed_reduced_predicted)

    return y_transformed_predicted


def train_model_no_dr(y_original_train, y_original_test, y_transformed_train, y_transformed_test):
    regression = LinearRegression()
    regression.fit(y_original_train, y_transformed_train)
    y_transformed_predicted = regression.predict(y_original_test)
    return y_transformed_predicted


if __name__ == '__main__':
    client = MongoClient()
    db = client.test_database
    collection = db.collection

    num_samples = 100#db.collection.count()
    dimension = 100

    y_original = np.zeros((num_samples, dimension))
    y_transformed = np.zeros((num_samples, dimension))

    for index, doc in enumerate(collection.find()[:num_samples]):
        y_original[index, :] = pickle.loads(doc['y_original'])[:, 0]
        y_transformed[index, :] = pickle.loads(doc['y_transformed'])[:, 0]

    y_original_train, y_original_test, y_transformed_train, y_transformed_test = train_test_split(y_original, y_transformed, test_size=0.2, random_state=42)

    y_transformed_predicted = train_model_dr(y_original_train, y_original_test, y_transformed_train, y_transformed_test)
    error = calc_error(y_transformed_predicted, y_transformed_test)
    print('error reduced: {}'.format(error))
    plot_functions(y_transformed_test, y_transformed_predicted)

    y_transformed_predicted = train_model_no_dr(y_original_train, y_original_test, y_transformed_train, y_transformed_test)
    error = calc_error(y_transformed_predicted, y_transformed_test)
    print('error not reduced: {}'.format(error))
