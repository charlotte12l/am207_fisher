import abc
import numpy as np
import scipy
from utils import *

class FIL(abc.ABC):
    X = None
    y = None
    w = None
    lam = None
    sigma = None
    n = None
    all_fils = None

    def __init__(self, w, X, y, lam=1, sigma=1):
        self.w = w
        self.X = X
        self.y = y
        self.lam = lam
        self.sigma = sigma
        self.n = len(self.X)

    @abc.abstractmethod
    def grad_x_grad_w_loss(self,x, y):
        """
        Grad_x_Grad_w
        """
        
    @abc.abstractmethod
    def grad_y_grad_w_loss(self, x, y):
        """
        Grad_y_Grad_w
        """

    def hessian(self, x, y):
        """
        Hessian for single example
        """

    @abc.abstractmethod
    def hessian_dataset(self):
        """
        Hessian for entire dataset
        """

    def jacobian(self, x, y):
        """
        Jacobian for single example
        """
        # print(self.hessian_dataset())
        return -np.linalg.inv(self.hessian_dataset()) @ np.hstack((self.grad_x_grad_w_loss(x, y), self.grad_y_grad_w_loss(x, y)))

    def fil(self, x, y):
        """
        Fisher Information Loss for single example
        """
        return np.linalg.norm(self.jacobian(x, y), 2)/self.sigma

    def compute_all_fils(self):
        """
        Computes all Fisher Information for all data points
        """
        self.all_fils = np.zeros(self.n)
        for i in range(self.n):
            self.all_fils[i] = self.fil(self.X[i], self.y[i])
        return self.all_fils
        
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
        sorted_ind = ind[np.argsort(self.all_fils[ind])]
        return self.X[sorted_ind], self.all_fils[sorted_ind]

    def lowest_fils(self, n):
        """
        Returns n data points with lowest FIL
        """
        ind = np.argpartition(-self.all_fils, -n)[-n:]
        sorted_ind = ind[np.argsort((-self.all_fils)[ind])]
        return self.X[sorted_ind], self.all_fils[sorted_ind]


class FIL_Logistic(FIL):
    def __init__(self, w, X, y, lam=1, sigma=1):
        super().__init__(w, X, y, lam, sigma)

    def grad_x_grad_w_loss(self, x, y):
        wtx = np.dot(self.w,x)
        return sigmoid(wtx) * (1 - sigmoid(wtx)) * np.outer(x, self.w) + (sigmoid(wtx) - y)
    
    def grad_y_grad_w_loss(self, x, y):
        return -x.reshape(-1, 1)

    def hessian(self, x, y):
        wtx = np.dot(self.w,x)
        return sigmoid(wtx) * (1 - sigmoid(wtx)) * np.outer(x, x)
    
    def hessian_dataset(self):
        # how to vectorize this??
        return np.sum(np.array([self.hessian(x, y) for x, y in zip(self.X, self.y)]), axis=0) + self.lam * self.n


class FIL_Linear(FIL):
    def __init__(self, w, X, y, lam=1, sigma=1):
        super().__init__(w, X, y, lam, sigma)

    def grad_x_grad_w_loss(self, x, y):
        wtx = np.dot(self.w,x)
        return np.outer(x, self.w) + (np.dot(self.w, x) - y)
    
    def grad_y_grad_w_loss(self, x, y):
        return -x.reshape(-1, 1)
    
    # note: no need for hessian per example

    def hessian_dataset(self):
        return np.transpose(self.X) @ self.X + self.n * self.lam

# class FIL_Logistic_Reweighted(FIL_Logistic):
#     def __init__(self, w, X, y, lam=1, sigma=1):
#         super().__init__(w, X, y, lam, sigma)

#     def iterative_reweighted(self, X, y, loss_func, num_iters, noise_std, lam):
#         n = len(y)
#         sample_weights = np.ones(n) # number of training examples

#         def f_to_minimize(w, omega):
#             return np.dot(omega, loss_func(X @ w.reshape(-1, 1), y)) + n * lam * np.linalg.norm(w)/ 2

#         for t in range(num_iters):
#             f = lambda w : f_to_minimize(w, sample_weights)
#             w_opt = scipy.optimize.minimize(f, np.zeros(n)).x
#             print("w_opt: ", w_opt)
#             w_prime = w_opt + np.random.normal(0, noise_std ** 2, size=len(w_opt))
#             self.w = w_prime
#             fils = self.compute_all_fils() # need to be able to adjust w for this
#             sample_weights = n * np.divide(sample_weights / fils) / sum(np.divide(sample_weights / fils))
        
#         return w_prime

class FIL_Linear_Reweighted(FIL_Linear):
    def __init__(self, w, X, y, lam=1, sigma=1):
        super().__init__(w, X, y, lam, sigma)

    def iterative_reweighted(self, X, y, loss_func, num_iters, noise_std, lam):
        n = len(y)
        sample_weights = np.ones(n) # number of training examples

        def f_to_minimize(w, omega):
            # print(w.shape)
            # print(X.shape)
            # print(X @ w)
            return np.dot(omega, loss_func(np.dot(X, w), y)) + n * lam * np.linalg.norm(w)/ 2

        for t in range(num_iters):
            f = lambda w : f_to_minimize(w, sample_weights)
            w_opt = scipy.optimize.minimize(f, np.zeros(len(self.w))).x
            # print("w_opt: ", w_opt)
            w_prime = w_opt + np.random.normal(0, noise_std ** 2, size=len(w_opt))
            self.w = w_prime
            fils = self.compute_all_fils() # need to be able to adjust w for this
            sample_weights = n * np.divide(sample_weights, fils) / sum(np.divide(sample_weights, fils))
        
        return w_prime