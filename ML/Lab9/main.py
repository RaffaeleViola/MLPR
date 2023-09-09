import sklearn.datasets
import numpy as np
import scipy.optimize as so


def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1] * 2.0 / 3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)


def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0]  # We remove setosa from D
    L = L[L != 0]  # We remove setosa from L
    L[L == 2] = 0  # We assign label 0 to virginica (was label 2)
    return D, L


def linear_svm_obj_wrap(DTR, LTR, K=1):
    def linear_svm_obj(_alpha):
        alpha = _alpha.reshape((DTR.shape[1], 1))
        ones = np.ones(DTR.shape[1]).reshape((1, DTR.shape[1]))
        D = np.vstack((DTR, K * ones))
        G = np.dot(D.T, D)
        z = (2 * LTR - 1).reshape((1, DTR.shape[1]))
        H = np.dot(z.T, z) * G
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
    return np.dot(w_s.reshape((w_s.shape[0]), 1).T, D_mapped)


def accuracy(Pred, Labels):
    return (Pred - Labels == 0).astype(int).sum() / Labels.shape[0]


def primal(DTR, LTR, w_s, C, K=1):
    zeros = np.zeros(DTR.shape[1])
    ones = K * np.ones(DTR.shape[1]).reshape((1, DTR.shape[1]))
    D = np.vstack((DTR, ones))
    z = (2 * LTR - 1).reshape((1, LTR.shape[0]))
    return 0.5 * (w_s * w_s).sum() + (C * np.maximum(zeros, 1 - z * np.dot(w_s.T, D)).sum())


def linear_svm():
    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    C_list = [0.1, 1.0, 10.0]
    K_list = [1, 10]
    for k in K_list:
        dual_obj = linear_svm_obj_wrap(DTR, LTR, K=k)
        for C in C_list:
            bounds = [(0, C) for _ in range(DTR.shape[1])]
            x0 = np.zeros(DTR.shape[1])
            x, f, d = so.fmin_l_bfgs_b(dual_obj, x0, bounds=bounds, factr=1.0)
            w_s = w_star(DTR, LTR, x, K=k)
            pred = (scores(DTE, w_s , k) > 0).astype(int)
            err = (1 - accuracy(pred, LTE)) * 100
            j_w = primal(DTR, LTR, w_s, C, K=k)
            gap = j_w + f
            print(f'K={k} ---- C={C} --- prima={j_w} ----- dual={-f} ---- gap={gap} ---- err={err}%\n')


def polynomial_kernel(x1, x2, kwargs):
    c = kwargs['c']
    d = kwargs['d']
    eps = kwargs['eps']
    return np.power(np.dot(x1.T, x2) + c, d) + eps


def func1D(a, *args, **kwargs):
    x2 = kwargs['x2']
    return np.apply_along_axis(lambda x: ((a - x)*(a - x)).sum(), axis=0, arr=x2)


def radial_basis_kernel(x1, x2, kwargs):
    gamma = kwargs['gamma']
    eps = kwargs['eps']
    norm = np.apply_along_axis(func1d=func1D, axis=0, arr=x1, x2=x2).T
    return np.exp(-gamma * norm) + eps


def kernel_svm_obj_wrap(DTR, LTR, kernel, **kwargs):
    def kernel_svm_obj(_alpha):
        alpha = _alpha.reshape((DTR.shape[1], 1))
        ones = np.ones(DTR.shape[1]).reshape((1, DTR.shape[1]))
        z = (2 * LTR - 1).reshape((1, DTR.shape[1]))
        H = np.dot(z.T, z) * kernel(DTR, DTR, kwargs)
        dual = 0.5 * np.dot(alpha.T, H).dot(alpha) - np.dot(alpha.T, ones.reshape((DTR.shape[1], 1)))
        gradient = (np.dot(H, alpha) - ones.reshape((DTR.shape[1], 1))).reshape((DTR.shape[1],))
        return dual, gradient
    return kernel_svm_obj


def kernel_scores(DTR, DTE, LTR, alpha, kernel, **kwargs):
    z = (2 * LTR - 1)
    az = (alpha * z).reshape((DTR.shape[1], 1))
    return (az * kernel(DTR, DTE, kwargs)).sum(axis=0)


def kernel_svm():
    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    K_list = [0.0, 1.0]
    C_list = [1.0]
    d = 2
    c_list = [0, 1]
    gamma_list = [1.0, 10.0]

    # polynomial kernel
    for c in c_list:
        for K in K_list:
            for C in C_list:
                bounds = [(0, C) for _ in range(DTR.shape[1])]
                eps = K**2
                k_svm_obj = kernel_svm_obj_wrap(DTR, LTR, polynomial_kernel, c=c, d=d, eps=eps)
                x0 = np.zeros(DTR.shape[1])
                x, f, d_ = so.fmin_l_bfgs_b(k_svm_obj, x0, bounds=bounds, factr=1.0)
                scores = kernel_scores(DTR, DTE, LTR, x, polynomial_kernel, c=c, d=d, eps=eps)
                pred = (scores > 0).astype(int)
                err = (1 - accuracy(pred, LTE)) * 100
                print(f'K={K} --- C={C} --- Poly(d = {d}, c = {c}) --- Dual Loss = {-f} --- err={err}%\n')

    print("\n-------------------------------------------\n")

    #radial basis kernel
    for gamma in gamma_list:
        for K in K_list:
            for C in C_list:
                bounds = [(0, C) for _ in range(DTR.shape[1])]
                eps = K**2
                k_svm_obj = kernel_svm_obj_wrap(DTR, LTR, radial_basis_kernel, gamma=gamma, eps=eps)
                x0 = np.zeros(DTR.shape[1])
                x, f, d_ = so.fmin_l_bfgs_b(k_svm_obj, x0, bounds=bounds, factr=1.0)
                scores = kernel_scores(DTR, DTE, LTR, x, radial_basis_kernel, gamma=gamma, eps=eps)
                pred = (scores > 0).astype(int)
                err = (1 - accuracy(pred, LTE)) * 100
                print(f'K={K} --- C={C} --- RBF(gamma = {gamma}) --- Dual Loss = {-f} --- err={err}%\n')


if __name__ == '__main__':
    print("-------Kernel------------")
    kernel_svm()
    print("-------Linear------------")
    linear_svm()

