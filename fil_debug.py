import abc
import numpy as np
from utils import *
import torch


class FIL(abc.ABC):
    X = None
    y = None
    w = None
    n = None
    all_fils = None

    def __init__(self):
        self.X = None
        self.y = None

    # @abc.abstractmethod
    # def grad_x_grad_w_loss(self,x, y):
    #     """
    #     Grad_x_Grad_w
    #     """
        
    # @abc.abstractmethod
    # def grad_y_grad_w_loss(self, x, y):
    #     """
    #     Grad_y_Grad_w
    #     """

    # def hessian(self, x, y):
    #     """
    #     Hessian for single example
    #     """

    # @abc.abstractmethod
    # def hessian_dataset(self):
    #     """
    #     Hessian for entire dataset
    #     """

    def compute_all_fils(self):
        self.all_fils = torch.linalg.norm(self.jacobian_dataset(), ord=2, dim=(1, 2))
        return self.all_fils


    def fil(self, x, y):
        """
        Fisher Information Loss for single example
        """
        return torch.linalg.norm(self.jacobian(x, y), ord=2, dim=(1, 2))
    
    
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


class FIL_Linear_lxy(FIL):
    def __init__(self):
        super().__init__()

    def train(self, data, lam=0):
        n = len(data["targets"])

        self.X = data["features"]
        self.y = data["targets"]
        XTX = (self.X).T @ self.X

        # However never use XTXdiag
        XTXdiag = torch.diagonal(XTX)
        XTXdiag += (n * lam)

        b = self.X.T @ self.y
        theta = torch.solve(b[:, None], XTX)[0].squeeze(1)

        print("THETA: {}".format(theta))
        # Need A to compute the Jacobian.
        # hessian_dataset
        A = torch.inverse(XTX)
        self.A = A
        self.theta = theta

    def get_params(self):
        return self.theta

    def set_params(self, theta):
        self.theta = theta

    def predict(self, X, regression=False):
        if regression:
            return X @ self.theta
        else:
            return (X @ self.theta) > 0

    def jacobian_dataset(self):
        return self.jacobian(self.X, self.y)
    
    def jacobian(self, X, y):
        r = (X @ self.theta - y)[:, None, None]
        XA = X @ self.A # self.A is Hessian Dataset
        # -((wx-y)*hessian+X@hessian*w)
        JX = -(r * self.A.unsqueeze(0) + XA.unsqueeze(2) * self.theta[None, None, :])
        # -hessian*(-x)
        JY = XA.unsqueeze(2)

        return torch.cat([JX, JY], dim=2)

    # note: no need for hessian per example
    def hessian_dataset(self):
        return self.A


class FIL_Logistic_lxy(FIL):
    def __init__(self):
        super().__init__()

    def train(self, data, l2=0, init=None):
        n = len(data["targets"])

        self.X = data["features"]
        self.y = data["targets"].float()

        # Save for the jacobian:
        self.l2 = n * l2

        theta = torch.randn(self.X.shape[1], requires_grad=True)
        if init is not None:
            theta.data[:] = init[:]


        crit = torch.nn.BCEWithLogitsLoss(reduction="none")
        optimizer = torch.optim.LBFGS([theta], line_search_fn="strong_wolfe")
        def closure():
            optimizer.zero_grad()
            loss = (crit(self.X @ theta, self.y)).sum()
            loss += (self.l2 / 2.0) * (theta**2).sum()
            loss.backward()
            return loss
        for _ in range(100):
            loss = optimizer.step(closure)
        self.theta = theta

    def get_params(self):
        return self.theta.detach().numpy()

    def set_params(self, theta):
        self.theta = theta

    def predict(self, X, regression=False):
        return (X @ self.theta) > 0

    def jacobian_dataset(self):
        return self.jacobian(self.X, self.y)
    
    def jacobian(self, X, y):

        # Compute the Hessian at theta for all X:
        s = (X @ self.theta).sigmoid().unsqueeze(1)
        H = (s * (1-s) * X).T @ X
        Hdiag = torch.diagonal(H)
        Hdiag += self.l2
        Hinv = H.inverse()

        # Compute the Jacobian of the gradient w.r.t. theta at each (x, y) pair
        XHinv = X @ Hinv
        JX = -(s * (1-s) * XHinv).unsqueeze(2) * self.theta[None, None, :]
        JX -= (s - y.unsqueeze(1)).unsqueeze(2) * Hinv.unsqueeze(0)
 
        JY =  (XHinv).unsqueeze(2)
        return torch.cat([JX, JY], dim=2)