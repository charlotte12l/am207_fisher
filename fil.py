import numpy as np
import scipy
from sklearn.linear_model import LogisticRegression, Ridge
from utils import *

class FIL():
    X = None
    y = None
    w = None
    lam = None
    sigma = None
    n = None
    d = None
    all_fils = None
    total_fil = None
    clf = None

    def __init__(self, lam=1, sigma=1):
        self.lam = lam
        self.sigma = sigma
        
    def predict(self, X):
        """
        Return predictions on data X
        """
        return self.clf.predict(X)

    def grad_x_grad_w_loss(self, x, y):
        """
        Grad_x_Grad_w
        """
        
    def grad_y_grad_w_loss(self, x, y):
        """
        Grad_y_Grad_w
        """
        
    def jacobian(self, x, y):
        """
        Jacobian for single example
        """
        # print(self.hessian_dataset())
        return -self.inverse_hessian @ np.hstack((self.grad_x_grad_w_loss(x, y), self.grad_y_grad_w_loss(x, y)))
    
    def compute_all_weights_irfil(self):
        all_weights = np.zeros(self.n)
        for i in range(self.n):
            J = self.jacobian(self.X[i], self.y[i])
            all_weights[i] = np.sqrt(np.linalg.norm(J.T @ J / self.sigma**2))
        return all_weights

    def fil(self, x, y):
        """
        Fisher Information Loss for single example
        """
        return np.linalg.norm(self.jacobian(x, y), 2)/self.sigma

    def compute_all_fils(self):
        """
        Computes all Fisher Information for each data point
        """
        self.all_fils = np.zeros(self.n)
        for i in range(self.n):
            self.all_fils[i] = self.fil(self.X[i], self.y[i])
        return self.all_fils
    
    def compute_total_fil(self):
        """
        Computess Fisher Information for entire dataset
        """
        
    def print_fil(self):
        """
        Prints fisher information loss for all examples
        """
        for i in range(self.n):
            if self.all_fils is not None:
                print("Point: {}, Value: {}, FIL: {}".format(self.X[i], self.y[i], np.round(self.all_fils[i], 3)))
            else:
                print("Point: {}, Value: {}, FIL: {}".format(self.X[i], self.y[i], np.round(self.fil(self.X[i], self.y[i]), 3)))

    def highest_fils(self, n):
        """
        Returns n data points with highest FIL
        """
        ind = np.argpartition(self.all_fils, -n)[-n:]
#         sorted_ind = ind[np.argsort(self.all_fils[ind])]
        return ind

    def lowest_fils(self, n):
        """
        Returns n data points with lowest FIL
        """
        ind = np.argpartition(-self.all_fils, -n)[-n:]
#         sorted_ind = ind[np.argsort((-self.all_fils)[ind])]
        return ind

    def irfil(self, X, y, T, noise):
        weights = np.ones(len(X))
        for t in range(T):
            self.train(X, y, weights=weights)
            self.w += np.random.normal(loc=0, scale=noise, size=self.d)
            self.compute_inverse_hessian()
            etas = self.compute_all_weights_irfil()
            num = np.divide(weights, etas)
            weights = self.n * num / np.sum(num)
        return weights


class FIL_Logistic(FIL):
    def __init__(self, lam=1, sigma=1):
        super().__init__(lam, sigma)
        
    def train(self, X, y, weights=None):
        self.X = X
        self.y = y
        assert len(self.X.shape) == 2, "Invalid data shape"
        self.n, self.d = X.shape[0], X.shape[1]
        
        # train model
        self.clf = LogisticRegression(fit_intercept=False, C=1/max(self.lam * self.n, 1e-5), solver='lbfgs').fit(X, y, sample_weight=weights)
        
        # find weights
        self.w = np.squeeze(self.clf.coef_)
        
        self.compute_inverse_hessian()
    
    def compute_inverse_hessian(self):
        # find inverse hessian
        self.inverse_hessian = np.linalg.inv((sigmoid(self.X @ self.w) * (1 - sigmoid(self.X @ self.w)) * self.X.T) @ self.X)
    
    def compute_accuracy(self, X, y):
        return self.clf.score(X, y)
        
    def grad_x_grad_w_loss(self, x, y):
        wtx = np.dot(self.w,x)
        return sigmoid(wtx) * (1 - sigmoid(wtx)) * np.outer(x, self.w) + (sigmoid(wtx) - y)
    
    def grad_y_grad_w_loss(self, x, y):
        return -x.reshape(-1, 1)
    
    def decision_boundary(self, xmin, xmax):
        b = self.w[0]
        w1, w2 = self.w[1:]
        c = -b/w2
        m = -w1/w2
        xd = np.array([xmin, xmax])
        yd = m*xd + c
        return xd, yd

class FIL_Linear(FIL):
    def __init__(self, lam=1, sigma=1):
        super().__init__(lam, sigma)
        
    def train(self, X, y, weights=None):
        self.X = X
        self.y = y
        assert len(self.X.shape) == 2, "Invalid data shape"
        self.n, self.d = X.shape[0], X.shape[1]
        
        # train model
        self.clf = Ridge(alpha=self.n * self.lam / 2, fit_intercept=False).fit(X, y, sample_weight=weights)
        
        # find weights
        self.w = np.squeeze(self.clf.coef_)
        
        # find inverse hessian
        self.compute_inverse_hessian()
        
    def compute_inverse_hessian(self):
        self.inverse_hessian = np.linalg.inv(np.transpose(self.X) @ self.X + self.n * self.lam * np.eye(self.d))

    def grad_x_grad_w_loss(self, x, y):
        wtx = np.dot(self.w,x)
        return np.outer(x, self.w) + (np.dot(self.w, x) - y)
    
    def grad_y_grad_w_loss(self, x, y):
        return -x.reshape(-1, 1)