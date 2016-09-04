'''
Created on 23.08.2016

@author: Artur
'''

import matplotlib.pylab as plt
from sklearn import datasets
from sklearn.preprocessing import normalize

from AEVB import AEVB


def showElbo(aevb):
    plt.figure()
    plt.plot(aevb.elbos)
    plt.xlabel('Epochs')
    plt.ylabel('ELBO')
    plt.title('ELBO')


def showSamples(aevb):
    f, ax = plt.subplots(2, 5)
    f.suptitle('Generated Samples', fontsize=14)

    for i in range(2):
        for j in range(5):
            ax[i, j].imshow(aevb.sample().reshape((8, 8)), cmap='gray_r', interpolation='nearest')


def showFeatureSpace(aevb, points, target):
    features = aevb.transform(points)

    plt.figure()
    plt.title('Feature Space')
    for i in range(10):
        plt.plot(features[:, 0][target == i], features[:, 1][target == i], '.')
    plt.grid(True)


def aevbExample():
    digits = datasets.load_digits(10)
    nSamples = len(digits.images)
    data = normalize(digits.images.reshape((nSamples, -1)))
    aevb = AEVB(n_components=2, n_epochs=5000, learning_rate=1e-3)
    aevb.fit(data)

    showElbo(aevb)
    showSamples(aevb)
    showFeatureSpace(aevb, data, digits.target)

    plt.show()


if __name__ == '__main__':
    aevbExample()
