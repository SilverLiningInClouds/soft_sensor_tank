# RR: Ridge Regression
from Base.packages import *
from sklearn.linear_model import Ridge


# Model
class RrModel(BaseEstimator, RegressorMixin):

    # Initialization
    def __init__(self, alpha=1.0):
        super(RrModel, self).__init__()
        self.alpha = alpha
        self.scaler = StandardScaler()

    # Train
    def fit(self, X, y):
        m, self.n = X.shape
        self.X = np.concatenate((X, np.ones((m, 1))), axis=1)
        self.y = self.scaler.fit_transform(y)
        self.w = np.matmul(
            np.matmul(np.linalg.inv(np.matmul(self.X.T, self.X) + self.alpha * np.eye(self.n + 1)), self.X.T), self.y)

        return self

    # Test
    def predict(self, X):
        m = X.shape[0]
        X = np.concatenate((X, np.ones((m, 1))), axis=1)
        y_pred = self.scaler.inverse_transform(np.matmul(X, self.w))

        return y_pred
