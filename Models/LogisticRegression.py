import numpy as np
import scipy.optimize as so


class LogisticRegression:

    def __init__(self, lmd, prior):
        self.b = None
        self.w = None
        self.lmd = lmd
        self.prior = prior

    @staticmethod
    def logreg_obj_wrap(DTR, LTR, l, prior):
        z = 2 * LTR - 1
        mask_T = z == 1
        mask_F = z == -1
        nT = z[mask_T].shape[0]
        nF = z[mask_F].shape[0]

        def logreg_obj(v):
            w, b = v[0:-1], v[-1]
            e1 = (l / 2) * (np.dot(w.T, w))
            e2 = prior * np.logaddexp(0, -z[mask_T] * (np.dot(w.T, DTR[:, mask_T]) + b)).sum() / nT
            e3 = (1 - prior) * np.logaddexp(0, -z[mask_F] * (np.dot(w.T, DTR[:, mask_F]) + b)).sum() / nF
            return e1 + e2 + e3

        return logreg_obj

    def transform(self, DTE):
        return np.dot(self.w.T, DTE) + self.b

    def fit(self, D, L):
        logreg_obj = LogisticRegression.logreg_obj_wrap(D, L, self.lmd, self.prior)
        x0 = np.zeros(D.shape[0] + 1)
        x, f, d = so.fmin_l_bfgs_b(logreg_obj, x0, approx_grad=True, factr=1.0)
        self.w, self.b = x[0:-1], x[-1]


class QuadraticLogisticRegression(LogisticRegression):
    @staticmethod
    def quadratic_expansion(X):
        x_T = np.repeat(X, repeats=X.shape[0], axis=0)
        x_stacked = X
        for _ in range(X.shape[0] - 1):
            x_stacked = np.vstack((x_stacked, X))
        mapped = np.vstack(((x_stacked * x_T), X))
        return mapped

    def fit(self, D, L):
        DTR = QuadraticLogisticRegression.quadratic_expansion(D)
        super().fit(DTR, L)

    def transform(self, DTE):
        DTE = QuadraticLogisticRegression.quadratic_expansion(DTE)
        return np.dot(self.w.T, DTE) + self.b

