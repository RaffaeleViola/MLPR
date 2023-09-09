import sklearn.datasets
import numpy as np
import scipy.optimize as so


def linear_svm_obj_wrap(DTR, LTR, K=1):
    ones = np.ones(DTR.shape[1]).reshape((1, DTR.shape[1]))
    D = np.vstack((DTR, K * ones))
    G = np.dot(D.T, D)
    z = (2 * LTR - 1).reshape((1, DTR.shape[1]))
    H = np.dot(z.T, z) * G

    def linear_svm_obj(_alpha):
        alpha = _alpha.reshape((DTR.shape[1], 1))
        dual = 0.5 * np.dot(alpha.T, H).dot(alpha) - np.dot(alpha.T, ones.reshape((DTR.shape[1], 1)))
        gradient = (np.dot(H, alpha) - ones.reshape((DTR.shape[1], 1))).reshape((DTR.shape[1],))
        return dual, gradient

    return linear_svm_obj


def w_star(DTR, LTR, _alpha, K=1):
    ones = K * np.ones(DTR.shape[1]).reshape((1, DTR.shape[1]))
    D = np.vstack((DTR, ones))
    a_z = (_alpha * (2 * LTR - 1)).reshape((1, DTR.shape[1]))
    return (a_z * D).sum(axis=1)


def scores(DTE, w_s, K):
    ones = K * np.ones(DTE.shape[1]).reshape((1, DTE.shape[1]))
    D_mapped = np.vstack((DTE, ones))
    return np.dot(w_s.reshape((w_s.shape[0]), 1).T, D_mapped).reshape((DTE.shape[1],))


def accuracy(Pred, Labels):
    return (Pred - Labels == 0).astype(int).sum() / Labels.shape[0]


def primal(DTR, LTR, w_s, C, K=1):
    zeros = np.zeros(DTR.shape[1])
    ones = K * np.ones(DTR.shape[1]).reshape((1, DTR.shape[1]))
    D = np.vstack((DTR, ones))
    z = (2 * LTR - 1).reshape((1, LTR.shape[0]))
    return 0.5 * (w_s * w_s).sum() + (C * np.maximum(zeros, 1 - z * np.dot(w_s.T, D)).sum())


def polynomial_kernel(x1, x2, kwargs):
    c = kwargs['c']
    d = kwargs['d']
    eps = kwargs['eps']
    return np.power(np.dot(x1.T, x2) + c, d) + eps


def func1D(a, *args, **kwargs):
    x2 = kwargs['x2']
    return np.apply_along_axis(lambda x: ((a - x) * (a - x)).sum(), axis=0, arr=x2)


def radial_basis_kernel(x1, x2, kwargs):
    gamma = kwargs['gamma']
    eps = kwargs['eps']
    norm = np.apply_along_axis(func1d=func1D, axis=0, arr=x1, x2=x2).T
    return np.exp(-gamma * norm) + eps


def kernel_svm_obj_wrap(DTR, LTR, kernel, **kwargs):
    ones = np.ones(DTR.shape[1]).reshape((1, DTR.shape[1]))
    z = (2 * LTR - 1).reshape((1, DTR.shape[1]))
    H = np.dot(z.T, z) * kernel(DTR, DTR, kwargs)

    def kernel_svm_obj(_alpha):
        alpha = _alpha.reshape((DTR.shape[1], 1))
        dual = 0.5 * np.dot(alpha.T, H).dot(alpha) - np.dot(alpha.T, ones.reshape((DTR.shape[1], 1)))
        gradient = (np.dot(H, alpha) - ones.reshape((DTR.shape[1], 1))).reshape((DTR.shape[1],))
        return dual, gradient

    return kernel_svm_obj


def kernel_scores(DTR, DTE, LTR, alpha, kernel, **kwargs):
    z = (2 * LTR - 1)
    az = (alpha * z).reshape((DTR.shape[1], 1))
    return (az * kernel(DTR, DTE, kwargs)).sum(axis=0)


def weighted_bounds(LTR, p_T, C):
    p_Temp = (LTR == 1).sum() / LTR.shape[0]
    C_T = C * p_T / p_Temp
    C_F = C * (1 - p_T) / (1 - p_Temp)
    bounds = [(0, C_T) if LTR[i] == 1 else (0, C_F) for i in range(LTR.shape[0])]
    return bounds


def linear_svm(DTR, LTR, DTE, p_T=0.5, C=1.0, k=1):
    dual_obj = linear_svm_obj_wrap(DTR, LTR, K=k)
    bounds = weighted_bounds(LTR, p_T, C)
    x0 = np.zeros(DTR.shape[1])
    x, f, d = so.fmin_l_bfgs_b(dual_obj, x0, bounds=bounds, factr=1.0)
    w_s = w_star(DTR, LTR, x, K=k)
    return scores(DTE, w_s, k)


def polynomial_svm(DTR, LTR, DTE, p_T=0.5, C=1.0, k=1.0, d=2, c=1):
    bounds = weighted_bounds(LTR, p_T, C)
    eps = k ** 2
    k_svm_obj = kernel_svm_obj_wrap(DTR, LTR, polynomial_kernel, c=c, d=d, eps=eps)
    x0 = np.zeros(DTR.shape[1])
    x, f, d_ = so.fmin_l_bfgs_b(k_svm_obj, x0, bounds=bounds, factr=1.0)
    scores = kernel_scores(DTR, DTE, LTR, x, polynomial_kernel, c=c, d=d, eps=eps)
    return scores


def RBF_svm(DTR, LTR, DTE, p_T=0.5, C=1.0, k=1.0, gamma=1.0):
    bounds = weighted_bounds(LTR, p_T, C)
    eps = k ** 2
    k_svm_obj = kernel_svm_obj_wrap(DTR, LTR, radial_basis_kernel, gamma=gamma, eps=eps)
    x0 = np.zeros(DTR.shape[1])
    x, f, d_ = so.fmin_l_bfgs_b(k_svm_obj, x0, bounds=bounds, factr=1.0)
    scores = kernel_scores(DTR, DTE, LTR, x, radial_basis_kernel, gamma=gamma, eps=eps)
    return scores



