import numpy as np
import scipy.optimize as so
import sklearn.datasets
import scipy


def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L


def f(x):
    func = np.square((x[0] + 3)) + np.sin(x[0]) + np.square((x[1] + 1))
    grad = np.array([2 * (x[0] + 3) + np.cos(x[0]), 2 * (x[1] + 1)])
    return func, grad


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


def logreg_obj_wrap(DTR, LTR, l):
    def logreg_obj(v):
        w, b = v[0:-1], v[-1]
        e1 = (l / 2) * (np.dot(w.T, w))
        z = 2 * LTR - 1
        e2 = np.logaddexp(0, -z * (np.dot(w.T, DTR) + b)).sum() / DTR.shape[1]
        return e1 + e2

    return logreg_obj


def score(DTE, w, b):
    return np.dot(w.T, DTE) + b


def accuracy(DTE, w, b, LTE):
    scr = score(DTE, w, b)
    return (((scr > 0).astype(int) - LTE) == 0).sum() * 100 / LTE.shape[0]


def binary_logreg():
    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    lambda_map = [10 ** -6, 10 ** -3, 10 ** -1, 1.0]
    for lmd in lambda_map:
        logreg_obj = logreg_obj_wrap(DTR, LTR, lmd)
        x0 = np.zeros(DTR.shape[0] + 1)
        x, f, d = so.fmin_l_bfgs_b(logreg_obj, x0, approx_grad=True)
        w, b = x[0:-1], x[-1]
        err = 100 - accuracy(DTE, w, b, LTE)
        print(f'lambda= {lmd} -- J(w,b)= {f} -- err={err}%')


def logreg_obj_wrap_multiclass(DTR, LTR, l, k):
    def logreg_obj(v):
        _w, _b = v[0:(DTR.shape[0]*k)], v[(DTR.shape[0]*k):]
        w = _w.reshape((DTR.shape[0], k))
        b = _b.reshape((k, 1))
        e1 = (l / 2) * (w * w).sum()
        z = np.zeros((k, DTR.shape[1]))
        for i in range(DTR.shape[1]):
            z[LTR[i], i] = 1
        ski = np.dot(w.T, DTR) + b
        lse = scipy.special.logsumexp(ski, axis=0).reshape(1, DTR.shape[1])
        for i in range(k):
            ski[i, :] = ski[i, :] - lse
        e2 = (z * ski).sum() / DTR.shape[1]
        return e1 - e2

    return logreg_obj


def accuracy_multi(Pred, LTE):
    res_vec = Pred - LTE
    corr_pred = 0
    for i in range(res_vec.shape[0]):
        if res_vec[i] == 0:
            corr_pred += 1
    acc = corr_pred / Pred.shape[0] * 100
    err = 100 - acc
    return acc, err


def multiclass_logreg():
    D, L = load_iris()
    n_class = 3
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    lambda_map = [10 ** -6, 10 ** -3, 10 ** -1, 1.0]
    for lmd in lambda_map:
        logreg_obj = logreg_obj_wrap_multiclass(DTR, LTR, lmd, n_class)
        x0 = np.zeros(((DTR.shape[0] * n_class) + n_class))
        x, f, d = so.fmin_l_bfgs_b(logreg_obj, x0, approx_grad=True)
        _w, _b = x[0:DTR.shape[0]*n_class], x[DTR.shape[0]*n_class:]
        w = _w.reshape((DTR.shape[0], n_class))
        b = _b.reshape((n_class, 1))
        score = np.dot(w.T, DTE) + b
        pred = score.argmax(axis=0)
        acc, err = accuracy_multi(pred, LTE)
        print(f'lambda= {lmd} -- J(w,b)= {f} -- err={err}%')


if __name__ == "__main__":
    # binary_logreg()
    multiclass_logreg()
