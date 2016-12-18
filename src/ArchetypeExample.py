'''
Created on 04.06.2016

@author: Artur
'''

import numpy as np
from sklearn import datasets
import matplotlib.pylab as plt
from matplotlib import offsetbox
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from ArchetypalAnalysis import ArchetypalAnalysis


def showTSNE(ax, pointsTSNE, target, archetypes, archetypesTSNE):
    ax.set_title('TSNE')

    ax.scatter(pointsTSNE[:, 0], pointsTSNE[:, 1], c=target, cmap=plt.cm.get_cmap("jet", 10))
    for i in range(np.size(archetypesTSNE, axis=0)):
        image = offsetbox.OffsetImage(archetypes[i, :].reshape((8, 8)), cmap='gray_r')
        imagebox = offsetbox.AnnotationBbox(image, archetypesTSNE[i, :]) 
        ax.add_artist(imagebox)


def showSample(ax, samples):
    ax.clear()
    ax.set_title('Sample')
    ax.imshow(samples, cmap='gray_r', interpolation='nearest')
    ax.get_figure().canvas.draw()


def showRepresentation(ax, index, pointsRepresentation, numberArchetypes):
    ax.clear()
    ax.set_title('Representation')
    ax.barh(range(numberArchetypes), pointsRepresentation[index, :], align='center')
    plt.xlabel('Performance')
    ax.get_figure().canvas.draw()


def showArchetypes(archetypes):
    f, ax = plt.subplots(2, 5, sharex='col', sharey='row')
    f.suptitle('Archetypes', fontsize=14)

    for i in range(2):
        for j in range(5):
            ax[i, j].imshow(archetypes[i * 5 + j, :].reshape((8, 8)), cmap='gray_r', interpolation='nearest')


def archetypalExample():
    digits = datasets.load_digits(10)
    nSamples = len(digits.images)
    data = digits.images.reshape((nSamples, -1))

    numberArchetypes = 10

    archetypal = ArchetypalAnalysis(n_components=numberArchetypes, tmax=50, iterations=20)
    archetypal.fit(data)

    dataArchetypal = archetypal.transform(data)
    dataArchetypal = np.vstack([np.eye(numberArchetypes), dataArchetypal])

    tsne = TSNE(n_components=2)
    dataArchetypalTSNE = tsne.fit_transform(dataArchetypal)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(dataArchetypalTSNE[numberArchetypes:])

    archetypes = archetypal.get_archetypes()

    f, ax = plt.subplots(1, 3)

    showTSNE(ax[0], dataArchetypalTSNE[numberArchetypes:, :], digits.target, archetypes, dataArchetypalTSNE[:numberArchetypes, :])
    showSample(ax[1], digits.images[0])
    showRepresentation(ax[2], 0, dataArchetypal[numberArchetypes:, :], numberArchetypes)
    showArchetypes(archetypes)

    def onclick(event):
        distances, indices = nbrs.kneighbors(np.array([event.xdata, event.ydata]).reshape((1, -1)))
        index = indices[0][0]

        showSample(ax[1], digits.images[index])
        showRepresentation(ax[2], index, dataArchetypal[numberArchetypes:, :], numberArchetypes)
    cid = f.canvas.mpl_connect('button_press_event', onclick)

    plt.show()

if __name__ == '__main__':
    archetypalExample()
