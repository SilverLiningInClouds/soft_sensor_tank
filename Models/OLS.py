# OLS: Ordinary Least Square
import numpy as np
from Base.packages import *
from sklearn.linear_model import LinearRegression


# Model
class OlsModel(BaseEstimator, RegressorMixin):

    # Initialization
    def __init__(self):
        super(OlsModel, self).__init__()
        self.scaler = StandardScaler()

    # Train
    def fit(self, X, y):
        m = X.shape[0]
        self.X = np.concatenate((X, np.ones((m, 1))), axis=1)
        self.y = self.scaler.fit_transform(y)
        self.w = np.matmul(np.matmul(np.linalg.inv(np.matmul(self.X.T, self.X)), self.X.T), self.y)

        return self

    # Test
    def predict(self, X):
        m = X.shape[0]
        X = np.concatenate((X, np.ones((m, 1))), axis=1)
        y_pred = self.scaler.inverse_transform(np.matmul(X, self.w))

        return y_pred
