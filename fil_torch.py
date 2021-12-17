# Some codes in this file were borrowed from the paper's original implementation
# Reference: https://github.com/facebookresearch/fisher_information_loss

# We implemented another version *purely based on our understanding* of the paper on our own in fil.py
# This file was used for initial verification of paper's results, 
# and because of parallel working, exploration on OOD probe also uses this implementation.

import abc
import numpy as np
from utils import *
import torch
import logging


class FIL(abc.ABC):
    X = None
    y = None
    w = None
    n = None
    all_fils = None

    def __init__(self):
        self.X = None
        self.y = None


    def compute_all_fils(self):
        """
        Example-level Fisher Information Loss for the whole dataset
        """
        self.all_fils = torch.linalg.norm(self.jacobian_dataset(), ord=2, dim=(1, 2))

        return self.all_fils


    def fil(self, x, y):
        """
        Example-level Fisher Information Loss for single example
        """
        return torch.linalg.norm(self.jacobian(x, y), ord=2, dim=(1, 2))



class FIL_Linear_torch(FIL):
    def __init__(self):
        super().__init__()

    def train(self, data, lam=0, weights=None):
        """
        Training
        """
        n = len(data["targets"])

        self.X = data["features"]
        self.y = data["targets"]

        if weights is None:
            weights = torch.ones(n)
        assert len(weights) == n, "Invalid number of weights"
        self.weights = weights

        XTX = (weights[:, None] * self.X).T @ self.X
        XTXdiag = torch.diagonal(XTX)
        XTXdiag += (n * lam)

        b = self.X.T @ (weights * self.y)
        theta = torch.solve(b[:, None], XTX)[0].squeeze(1)

        A = torch.inverse(XTX)
        self.A = A
        self.theta = theta

    def jacobian_total(self):
        """
        Full Fisher Information Loss for the whole dataset
        """
        cated = None
        for i in range(len(self.X)):
            x = self.X[i].clone()
            x = x[None, :]
            y = self.y[i].clone()
            y = torch.reshape(y, (1,))
            jac = self.jacobian(x, y) 
            #print('x:', x.shape,'y:', y.shape, 'jac:', jac.shape)
            if cated == None:
                cated = jac
            else:
                cated = torch.cat((cated, jac), dim=2)
        
        #print('cated:', cated.shape)
        return torch.linalg.norm(cated, ord=2,  dim=(1, 2))
        
    def get_params(self):
        return self.theta

    def set_params(self, theta):
        self.theta = theta

    def predict(self, X, regression=False):
        if regression:
            return X @ self.theta
        else:
            return (X @ self.theta) > 0

    def loss(self, data):
        """
        Evaluate the loss for each example in a given dataset.
        """
        X = data["features"]
        y = data["targets"].float()
        return (X @ self.theta - y)**2 / 2

    def jacobian_dataset(self):
        """
        Jacobian for the whole dataset
        """
        return self.jacobian(self.X, self.y, weighted=True)
    
    def jacobian(self, X, y, weighted=False):
        """
        Jacobian for a single example
        """
        r = (X @ self.theta - y)[:, None, None]
        XA = X @ self.A # self.A is inverse Hessian Dataset
        # -((wx-y)*inv_hessian+X@inv_hessian*w)
        JX = -(r * self.A.unsqueeze(0) + XA.unsqueeze(2) * self.theta[None, None, :])

        if weighted:
            JX = self.weights[:, None, None] * JX
            JY = (self.weights[:, None] * XA).unsqueeze(2)
        else:
            JY = XA.unsqueeze(2)
        
        return torch.cat([JX, JY], dim=2)



class FIL_Logistic_torch(FIL):
    def __init__(self):
        super().__init__()

    def train(self, data, l2=0, weights=None):
        n = len(data["targets"])

        self.X = data["features"]
        self.y = data["targets"].float()

        if weights is None:
            weights = torch.ones(n)
        assert len(weights) == n, "Invalid number of weights"

        self.weights = weights
        self.l2 = n * l2

        theta = torch.randn(self.X.shape[1], requires_grad=True)

        crit = torch.nn.BCEWithLogitsLoss(reduction="none")
        optimizer = torch.optim.LBFGS([theta], line_search_fn="strong_wolfe")
        def closure():
            optimizer.zero_grad()
            loss = (crit(self.X @ theta, self.y) * weights).sum()
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

    def loss(self, data):
        X = data["features"]
        y = data["targets"].float()
        return torch.nn.BCEWithLogitsLoss(reduction="none")(X @ self.theta, y)
    
    def jacobian_dataset(self):
        return self.jacobian(self.X, self.y, weighted=True)

    def jacobian_total(self):
        cated = None
        for i in range(len(self.X)):
            x = self.X[i].clone()
            x = x[None, :]
            y = self.y[i].clone()
            # y = torch.reshape(y, (-1,1))
            y = torch.reshape(y, (1,))
            jac = self.jacobian(x, y) 
            #print('x:', x.shape,'y:', y.shape, 'jac:', jac.shape)
            if cated == None:
                cated = jac
            else:
                cated = torch.cat((cated, jac), dim=2)
        
        #print('cated:', cated.shape)
        return torch.linalg.norm(cated, ord=2,  dim=(1, 2))
            
        # return torch.linalg.norm(torch.cat([self.jacobian(x, y) for i in range(len(self.X))], dim=2), ord=2)
    
    def jacobian(self, X, y, weighted=False):
        if weighted:
            s = (X @ self.theta).sigmoid().unsqueeze(1)
            H = (self.weights.unsqueeze(1) * s * (1-s) * X).T @ X
            Hdiag = torch.diagonal(H)
            Hdiag += self.l2
            Hinv = H.inverse()

            # Compute the Jacobian of the gradient w.r.t. theta at each (x, y) pair
            XHinv = X @ Hinv
            JX = -(s * (1-s) * XHinv).unsqueeze(2) * self.theta[None, None, :]
            JX -= (s - y.unsqueeze(1)).unsqueeze(2) * Hinv.unsqueeze(0)
            JX = self.weights[:, None, None] * JX
            JY =  (self.weights[:, None] * XHinv).unsqueeze(2)
        else:
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


def compute_accuracy(model, data, regression=False):
    X, y = data["features"], data["targets"].clone()
    if regression:
        acc = model.loss(data).mean().item()
    else:
        predictions = model.predict(X)

        # Linear Regression for MNIST, we preprocessed y[y==0]=-1 before training, 
        # however the prediction is 0/1 not -1/1
        y[y==-1] = 0 
        
        acc = ((predictions == y).float()).mean().item()
    return acc


def iterative_reweighted_fil(model, train_data, test_data, iters, l2=0, regression=False):
    model.train(train_data, l2)

    train_accuracy = compute_accuracy(model, train_data, regression=regression)
    test_accuracy = compute_accuracy(model, test_data, regression=regression)

    if regression:
        logging.info(f"Weighted model MSE train {train_accuracy:.3f},"
            f" test: {test_accuracy:.3f}.")
    else:
        logging.info(f"Weighted model accuracy train {train_accuracy:.3f},"
            f" test: {test_accuracy:.3f}.")

    # Compute the Fisher information loss, eta, for each example in the
    # training set:
    logging.info("Computing unweighted etas on training set...")
    etas = model.compute_all_fils()
    logging.info(f"etas max: {etas.max().item():.4f},"
        f" mean: {etas.mean().item():.4f}, std: {etas.std().item():.4f}.")
    
    # Reweight using the fisher information loss:
    updated_fi = etas.reciprocal().detach()
    maxs = [etas.max().item()]
    means = [etas.mean().item()]
    stds = [etas.std().item()]
    train_accs = [train_accuracy]
    test_accs = [test_accuracy]
    all_weights = [torch.ones(len(updated_fi))]
    for i in range(iters):
        logging.info(f"Iter {i}: Training weighted model...")
        updated_fi *= (len(updated_fi) / updated_fi.sum())

        weights = torch.ones(len(train_data["targets"]))
        weights[:] = updated_fi.data
        model.train(train_data, l2, weights=weights.detach())

        # Check predictions of weighted model:
        train_accuracy = compute_accuracy(model, train_data, regression=regression)
        test_accuracy = compute_accuracy(model, test_data, regression=regression)
        if regression:
            logging.info(f"Weighted model MSE train {train_accuracy:.3f},"
                f" test: {test_accuracy:.3f}.")
        else:
            logging.info(f"Weighted model accuracy train {train_accuracy:.3f},"
                f" test: {test_accuracy:.3f}.")

        # model.set_weights(weights)
        weighted_etas = model.compute_all_fils()
        updated_fi /= weighted_etas
        maxs.append(weighted_etas.max().item())
        means.append(weighted_etas.mean().item())
        stds.append(weighted_etas.std().item())
        train_accs.append(train_accuracy)
        test_accs.append(test_accuracy)
        all_weights.append(weights)
        logging.info(f"Weighted etas max: {maxs[-1]:.4f},"
            f" mean: {means[-1]:.4f},"
            f" std: {stds[-1]:.4f}.")

    results = {
        "weights" : weights.tolist(),
        "etas" : etas.tolist(),
        "weighted_etas" : weighted_etas.tolist(),
        "eta_maxes" : maxs,
        "eta_means" : means,
        "eta_stds" : stds,
        "train_accs" : train_accs,
        "test_accs" : test_accs,
    }

    return results