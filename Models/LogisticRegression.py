import numpy as np
import scipy.optimize as so


def logreg_obj_wrap(DTR, LTR, l, prior):
    def logreg_obj(v):
        w, b = v[0:-1], v[-1]
        e1 = (l / 2) * (np.dot(w.T, w))
        z = 2 * LTR - 1
        mask_T = z == 1
        mask_F = z == -1
        nT = z[mask_T].shape[0]
        nF = z[mask_F].shape[0]
        e2 = prior * np.logaddexp(0, -z[mask_T] * (np.dot(w.T, DTR[:, mask_T]) + b)).sum() / nT
        e3 = (1 - prior) * np.logaddexp(0, -z[mask_F] * (np.dot(w.T, DTR[:, mask_F]) + b)).sum() / nF
        return e1 + e2 + e3
    return logreg_obj



def score(DTE, w, b):
    return np.dot(w.T, DTE) + b


def accuracy(DTE, w, b, LTE):
    scr = score(DTE, w, b)
    return (((scr > 0).astype(int) - LTE) == 0).sum() * 100 / LTE.shape[0]


def train(D, L, lmd, prior):
    logreg_obj = logreg_obj_wrap(D, L, lmd, prior)
    x0 = np.zeros(D.shape[0] + 1)
    x, f, d = so.fmin_l_bfgs_b(logreg_obj, x0, approx_grad=True, factr=1.0)
    w, b = x[0:-1], x[-1]
    return w, b


def binarylogreg(DTR, LTR, DTE, lmd=0, prior=0.5):
    w, b = train(DTR, LTR, lmd, prior)
    return score(DTE, w, b)


def quadratic_expansion(X):
    x_T = np.repeat(X, repeats=X.shape[0], axis=0)
    x_stacked = X
    for _ in range(X.shape[0] - 1):
        x_stacked = np.vstack((x_stacked, X))
    mapped = np.vstack(((x_stacked * x_T), X))
    return mapped


def quadratic_binarylogreg(DTR, LTR, DTE, lmd=0, prior=0.5):
    DTR = quadratic_expansion(DTR)
    w, b = train(DTR, LTR, lmd, prior)
    DTE = quadratic_expansion(DTE)
    return score(DTE, w, b)


