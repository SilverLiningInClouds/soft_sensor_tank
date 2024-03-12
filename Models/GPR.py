# GPR: Gaussian Process Regression
from Base.packages import *
from sklearn.gaussian_process import GaussianProcessRegressor


# Model
class GprModel(BaseEstimator, RegressorMixin):

    # Initialization
    def __init__(self, l=0.5, sigma=0.2, optimize=True):
        super(GprModel, self).__init__()
        self.l = l
        self.sigma = sigma
        self.optimize = optimize

    # Train
    def fit(self, X, y):
        self.X = X
        self.y = y

        def negative_log_likelihood_loss(params):
            self.l = params[0]
            self.sigma = params[1]
            Kyy = self.kernel(self.X, self.X) + 1e-8 * np.eye(len(self.X))
            loss = 0.5 * self.y.T.dot(np.linalg.inv(Kyy)).dot(self.y) + 0.5 * np.linalg.slogdet(Kyy)[1] + 0.5 * len(
                self.X) * np.log(2 * np.pi)
            return loss.ravel()

        if self.optimize:
            res = minimize(negative_log_likelihood_loss, [self.l, self.sigma], bounds=((1e-4, 1e4), (1e-4, 1e4)),
                           method='BFGS')
            self.l, self.sigma = res.x[0], res.x[1]

        return self

    # Test
    def predict(self, X, sigma=False):
        Kff = self.kernel(self.X, self.X)
        Kyy = self.kernel(X, X)
        Kfy = self.kernel(self.X, X)
        Kff_inv = np.linalg.inv(Kff + 1e-8 * np.eye(len(self.X)))

        mu = Kfy.T.dot(Kff_inv).dot(self.y)
        cov = Kyy - Kfy.T.dot(Kff_inv).dot(Kfy)

        return (mu, cov) if sigma else mu

    # Kernel
    def kernel(self, x1, x2):
        dist_matrix = np.sum(x1 ** 2, 1).reshape(-1, 1) + np.sum(x2 ** 2, 1) - 2 * np.dot(x1, x2.T)
        return self.sigma ** 2 * np.exp(-0.5 / self.l ** 2 * dist_matrix)
