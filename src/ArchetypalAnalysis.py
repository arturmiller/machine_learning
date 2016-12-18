import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ArchetypalAnalysis(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=None, tmax=30, iterations=20):
        self.n_components = n_components
        self.tmax = tmax
        self.iterations = iterations

    def computeA(self, X, Z, tmax):
        n = np.size(X, axis=1)
        k = np.size(Z, axis=1)

        A = np.zeros((k, n))
        A[0, :] = 1.0

        for t in range(tmax):
            G = 2.0 * (np.dot(Z.T, np.dot(Z, A)) -
                       np.dot(Z.T, X))
            for i in range(n):
                j = np.argmin(G[:, i])

                A[:, i] += 2.0 / (t + 2.0) * (np.eye(1, k, j)[0, :] - A[:, i])

        return A

    def computeB(self, X, A, tmax):
        (k, n) = np.shape(A)

        B = np.zeros((n, k))
        B[0, :] = 1.0

        for t in range(tmax):
            G = 2.0 * (np.dot(X.T, np.dot(X, np.dot(B, np.dot(A, A.T)))) -
                       np.dot(X.T, np.dot(X, A.T)))
            for j in range(k):
                i = np.argmin(G[:, j])
                B[:, j] += 2.0 / (t + 2.0) * (np.eye(1, n, i)[0, :] - B[:, j])

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

        Z = np.dot(X, B)

        for i in range(self.iterations):
            A = self.computeA(X, Z, self.tmax)
            B = self.computeB(X, A, self.tmax)
            Z = np.dot(X, B)
            print('RSS: ' + str(self._rss(X, A, Z)))

        self.Z_ = Z
        self.A_ = A

    def inverse_transform(self, X):
        return np.dot(self.Z_, X.T).T

    def get_archetypes(self):
        return self.Z_.T

    def transform(self, X):
        A = self.computeA(X.T, self.Z_, self.tmax)
        return A.T

    def _rss(self, X, A, Z):
        return np.linalg.norm(X - np.dot(Z, A))

    def score(self, X):
        X = X.T
        return -self._rss(X, self.A_, self.Z_)
