# PCA: Principal Component Analysis
import numpy as np
from Base.packages import *
from sklearn.decomposition import PCA


# Model
class PcaModel(BaseEstimator, TransformerMixin):

    # Initialization
    def __init__(self):
        super(PcaModel, self).__init__()

    # Fit
    def fit(self, X, y=None):
        w, v = np.linalg.eig(np.matmul(X.T, X))
        self.load = v
        self.eigen_value = w

        return self

    # Transform
    def transform(self, X):
        return np.matmul(X, self.load)

    # Fit & Transform
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
