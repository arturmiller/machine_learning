import numpy as np
from sklearn import datasets
import matplotlib.pylab as plt
from matplotlib import offsetbox
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

class Archetypal(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=None, tmax=30, iterations=20):
        self.n_components = n_components
        self.tmax = tmax
        self.iterations = iterations

    def computeA(self, X, B, tmax):
        (n, k) = np.shape(B)
        
        archetypes = np.dot(X, B)
        
        A = np.zeros((k, n))
        A[0,:] = 1.0
    
        for t in range(tmax):
            G = 2.0 * (np.dot(archetypes.T, np.dot(archetypes, A)) - np.dot(archetypes.T, X))
            for i in range(n):
                j = np.argmin(G[:,i])
    
                A[:,i] += 2.0/(t+2.0) * (np.eye(1, k, j)[0,:] - A[:,i])
                
        return A
    
    def computeB(self, X, A, tmax):
        (k, n) = np.shape(A)
        
        B = np.zeros((n, k))
        B[0,:] = 1.0
        
        for t in range(tmax):
            G = 2.0 * (np.dot(X.T, np.dot(X, np.dot(B, np.dot(A, A.T)))) - np.dot(X.T, np.dot(X, A.T)))
            for j in range(k):
                i = np.argmin(G[:,j])
                B[:,j] += 2.0/(t+2.0) * (np.eye(1, n, i)[0,:] - B[:,j])
                
        return B

    def fit(self, X, y=None):
        self._fit(X)
        return self

    def fit_transform(self, X, y=None):
        self._fit(X)
        
        return self.transform(X)
        

    def _fit(self, X):
        k = self.n_components
        nSamples = np.size(X, axis=0)
        X = X.T
        
        B = np.eye(nSamples, k)
    
        self.Z = np.dot(X,B)

        for i in range(self.iterations):
            A = self.computeA(X, B, self.tmax)
            B = self.computeB(X, A, self.tmax)
            print 'score: ' + str(self._score(X, A, B))
      
            self.Z = np.dot(X,B)
            
        self.B = B
    
    def get_archetypes(self):
        return self.Z.T

    def transform(self, X):
        A = self.computeA(X.T, self.B, self.tmax)
        return A.T

    def _score(self, X, A, B):
        return np.linalg.norm(X - np.dot(X, np.dot(B, A)))
    
def showTSNE(ax, pointsTSNE, target, archetypes, archetypesTSNE):
    ax.set_title('TSNE')
    
    
    ax.scatter(pointsTSNE[:,0], pointsTSNE[:,1], c=target, cmap=plt.cm.get_cmap("jet", 10))
    for i in range(np.size(archetypesTSNE, axis=0)):
        image = offsetbox.OffsetImage(archetypes[i,:].reshape((8,8)), cmap='gray_r')
        imagebox = offsetbox.AnnotationBbox(image, archetypesTSNE[i,:]) 
        ax.add_artist(imagebox)
    
def showSample(ax, samples):
    ax.clear()
    ax.set_title('Sample')
    ax.imshow(samples, cmap='gray_r', interpolation='nearest')
    ax.get_figure().canvas.draw()
    
def showRepresentation(ax, index, pointsRepresentation, numberArchetypes):
    ax.clear()
    ax.set_title('Representation')
    ax.barh(range(numberArchetypes), pointsRepresentation[index,:], align='center')
    plt.xlabel('Performance')
    ax.get_figure().canvas.draw()
    
def showArchetypes(archetypes):
    
    f, ax = plt.subplots(2, 5, sharex='col', sharey='row')
    f.suptitle('Archetypes', fontsize=14)

    for i in range(2):
        for j in range(5):
            ax[i,j].imshow(archetypes[i*5+j,:].reshape((8,8)), cmap='gray', interpolation='nearest')


if __name__=='__main__':
  
    digits = datasets.load_digits(8)
    nSamples = len(digits.images)
    data = digits.images.reshape((nSamples, -1))
    
    numberArchetypes = 10

    archetypal = Archetypal(n_components=numberArchetypes, tmax=50, iterations=20)
    archetypal.fit(data)

    dataArchetypal = archetypal.transform(data)
    dataArchetypal = np.vstack([np.eye(numberArchetypes), dataArchetypal])

    tsne = TSNE(n_components=2)
    dataArchetypalTSNE = tsne.fit_transform(dataArchetypal)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(dataArchetypalTSNE[numberArchetypes:])
    

    archetypes = archetypal.get_archetypes()
    

    f, ax = plt.subplots(1, 3)

    showTSNE(ax[0], dataArchetypalTSNE[numberArchetypes:,:], digits.target, archetypes, dataArchetypalTSNE[:numberArchetypes,:])
    showSample(ax[1], digits.images[0])
    showRepresentation(ax[2], 0, dataArchetypal[numberArchetypes:,:], numberArchetypes)
    showArchetypes(archetypes)
    
    def onclick(event):      
        distances, indices = nbrs.kneighbors(np.array([event.xdata, event.ydata]).reshape((1,-1)))
        index = indices[0][0]

        showSample(ax[1], digits.images[index])
        showRepresentation(ax[2], index, dataArchetypal[numberArchetypes:,:], numberArchetypes)
    cid = f.canvas.mpl_connect('button_press_event', onclick)
    
    plt.show()
    