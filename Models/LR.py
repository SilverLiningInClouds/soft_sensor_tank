# LR: Logistic Regression
from Base.packages import *
from sklearn.linear_model import LogisticRegression


# Model
class LrModel(BaseEstimator, ClassifierMixin):

    # Initialization
    def __init__(self, C=1000, lr=0.001, max_iter=200, seed=123):
        super(LrModel, self).__init__()
        self.C = C
        self.lr = lr
        self.max_iter = max_iter
        np.random.seed(seed)

    # Train
    def fit(self, X, y):
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        y = y.squeeze()
        self.classes = np.unique(y)
        self.n_classes = self.classes.shape[0]
        if self.n_classes == 2:
            self.w = np.random.uniform(-1, 1, (X.shape[1], 1))
        else:
            self.w = np.random.uniform(-1, 1, (X.shape[1], self.n_classes))
        for i in range(self.w.shape[1]):
            y_temp = np.zeros(y.shape)
            y_temp[y == self.classes[i]] = 1
            for j in range(self.max_iter):
                grad = np.matmul(X.T, 1 / (1 + np.exp(-np.matmul(X, self.w[:, i]))) - y_temp)
                self.w[:, i] = self.w[:, i] - self.lr * (self.C * grad / X.shape[0] + 2 * self.w[:, i])

        return self

    # Test
    def predict(self, X):
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        y_pred = 1 / (1 + np.exp(-np.matmul(X, self.w)))
        y_pred = self.classes[np.argmax(y_pred, axis=1)]

        return y_pred
