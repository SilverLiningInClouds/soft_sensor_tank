# ELM: Extreme Learning Machine
from Base.packages import *


# Model
class ElmModel(BaseEstimator, RegressorMixin):

    # Initialization
    def __init__(self, dim_h=1024, alpha=1.0, direct_link=True, seed=123):
        super(ElmModel, self).__init__()

        # Set seed
        np.random.seed(seed)

        # Parameter assignment
        self.dim_h = dim_h
        self.alpha = alpha
        self.direct_link = direct_link
        self.seed = seed
        self.scaler = StandardScaler()

    # Add ones
    def add_ones(self, X):
        return np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

    # ReLU
    def relu(self, X):
        return np.maximum(0, X)

    # Train
    def fit(self, X, y):
        self.dim_X = X.shape[1]
        self.dim_y = y.shape[1]
        self.X = X
        self.y = self.scaler.fit_transform(y)

        self.w1 = []
        self.w2 = []
        self.std = 1. / math.sqrt(self.dim_X)
        for i in range(self.dim_y):
            self.w1.append(np.random.uniform(-self.std, self.std, (self.dim_X + 1, self.dim_h)))

        X = self.add_ones(X)
        for i in range(self.dim_y):
            h = self.relu(np.matmul(X, self.w1[i]))
            if self.direct_link:
                h = np.concatenate((h, X), axis=1)
            else:
                h = self.add_ones(h)
            self.w2.append(
                np.matmul(np.linalg.inv(np.matmul(h.T, h) + np.eye(h.shape[1]) / self.alpha),
                          np.matmul(h.T, self.y[:, i])))

        return self

    # Test
    def predict(self, X):
        res_list = []
        X = self.add_ones(X)

        for i in range(self.dim_y):
            h = self.relu(np.matmul(X, self.w1[i]))
            if self.direct_link:
                h = np.concatenate((h, X), axis=1)
            else:
                h = self.add_ones(h)
            res_list.append(np.matmul(h, self.w2[i]).squeeze())
        y = self.scaler.inverse_transform(np.stack(res_list, axis=-1))

        return y
