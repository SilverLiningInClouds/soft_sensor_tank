# LASSO: Least Absolute Shrinkage and Selection Operator
from Base.packages import *
from sklearn.linear_model import Lasso


# Model
class LassoModel(BaseEstimator, RegressorMixin):

    # Initialization
    def __init__(self, alpha=1.0):
        super(LassoModel, self).__init__()
        self.alpha = alpha
        self.scaler = StandardScaler()

    # Train
    def fit(self, X, y):
        m, self.n = X.shape
        w = cp.Variable(shape=(self.n + 1, 1))
        self.X = np.concatenate((X, np.ones((m, 1))), axis=1)
        self.y = self.scaler.fit_transform(y)
        term1 = cp.norm2(self.y - self.X * w) ** 2 / (2 * m)
        term2 = self.alpha * cp.norm1(w)
        problem = cp.Problem(cp.Minimize(term1 + term2))
        problem.solve(solver='ECOS')
        self.w = w.value

        return self

    # Test
    def predict(self, X):
        m = X.shape[0]
        X = np.concatenate((X, np.ones((m, 1))), axis=1)
        y_pred = self.scaler.inverse_transform(np.matmul(X, self.w))

        return y_pred
