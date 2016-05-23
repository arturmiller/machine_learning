import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pylab as plt
from sklearn import preprocessing

def plot2DPoints(X, Z):
    plt.clf()
    plt.plot(X[0,:], X[1,:], '.', color='b')
    plt.plot(Z[0,:], Z[1,:], 'ro', color='r')
    
    
def score(X, A, B):
    return np.linalg.norm(X - np.dot(X, np.dot(B, A)))

def computeA(X, B, tmax):
    (n, k) = np.shape(B)
    
    Z = np.dot(X, B)
    
    A = np.zeros((k, n))
    A[0,:] = 1.0
    #print np.linalg.norm(X - np.dot(Z, A))
    
    for t in range(tmax):
        G = 2.0 * (np.dot(Z.T, np.dot(Z, A)) - np.dot(Z.T, X))
        for i in range(n):
            j = np.argmin(G[:,i])

            A[:,i] += 2.0/(t+2.0) * (np.eye(1, k, j)[0,:] - A[:,i])
            
    #print np.linalg.norm(X - np.dot(Z, A))
    return A

def computeB(X, A, tmax):
    (k, n) = np.shape(A)
    
    B = np.zeros((n, k))
    B[0,:] = 1.0
    
    for t in range(tmax):
        G = 2.0 * (np.dot(X.T, np.dot(X, np.dot(B, np.dot(A, A.T)))) - np.dot(X.T, np.dot(X, A.T)))
        for j in range(k):
            
            #    import pdb
            #    pdb.set_trace()
            if np.linalg.norm(G[:,j], 1) < 1e-6:
                E = X - np.dot(X, np.dot(B, A)) 
                i = np.argmax(np.linalg.norm(E, axis=0))
                #import pdb
                #pdb.set_trace()
                #i = np.random.randint(n)
            else:
                i = np.argmin(G[:,j])
            #if j==2 or j==6:
            #    print 'woot: ' +  str((i, j))
            #    print G[:,j]
            B[:,j] += 2.0/(t+2.0) * (np.eye(1, n, i)[0,:] - B[:,j])
            
    return B

if __name__=='__main__':
    #plt.ion()
  
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)
    data_pca = preprocessing.scale(data_pca)

    #n_samples = 50
    #data_pca = data_pca[:n_samples,:]
    tmax = 100
    k = 10
    X = data_pca.T
    B = np.eye(n_samples, k)
    #B = np.random.rand(n_samples, k)
    #B = B/np.mean(B,axis=0)
    Z = np.dot(X,B)
    plot2DPoints(X, Z)
    
    #plt.plot(range(np.size(Z, axis=1)), Z[0,:])
    #plt.plot(range(np.size(Z, axis=1)), Z[1,:], color='r')
    #plt.show()
    for i in range(100):
        #print Z

        A = computeA(X, B, tmax)
        B = computeB(X, A, tmax)
        print score(X, A, B)
        #plt.show()
        plt.pause(0.5)
        Z_old = Z[:]
        Z = np.dot(X,B)
        print np.mean(Z-Z_old, axis=0)
     
        plot2DPoints(X, Z)
#         plt.plot(range(np.size(Z, axis=1)), Z[0,:])
#         plt.plot(range(np.size(Z, axis=1)), Z[1,:], color='r')
#         plt.show()
#         import pdb
#         pdb.set_trace()
        
    
    
    
    
    
    
    