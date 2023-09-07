from sklearn import datasets
import numpy as np
from measures import *


def load_iris():
    D, L = datasets.load_iris()['data'].T, datasets.load_iris()['target']
    return D, L


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


def center_data(dataset):
    mu = vcol(dataset.mean(1))
    centered_dataset = dataset - mu
    return centered_dataset, mu


def zscore(dataset):
    mu = vcol(dataset.mean(1))
    return (dataset - mu) / np.std(dataset)


def covariance(X):
    mu = vcol(X.mean(1))
    return np.dot(X - mu, (X - mu).T) / X.shape[1]


def vrow(v):
    return v.reshape((1, v.shape[0]))


def vcol(v):
    return v.reshape((v.shape[0], 1))


def PCA(m, D):
    DC, _ = center_data(D)
    C = np.dot(DC, DC.T) / DC.shape[1]
    s, U = np.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]
    return P


def KFold_CV(D, L, K, Classifier, wpoint, pca_m=0, seed=0, pre_process=None, **kwargs):
    nTest = int(D.shape[1] / K)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    scores = np.array([])
    labels = np.array([])
    for i in range(K):
        start = nTest * i
        idxTrain = np.concatenate((idx[0:start], idx[(start + nTest):]))
        idxTest = idx[start: (start + nTest)]
        DTR = D[:, idxTrain]
        DTE = D[:, idxTest]
        LTR = L[idxTrain]
        LTE = L[idxTest]
        if pre_process is not None:
            DTR = pre_process(DTR)
        if pca_m != 0:
            P = PCA(pca_m, DTR)
            DTR = np.dot(P.T, DTR)
            DTE = np.dot(P.T, DTE)
        llr = Classifier(DTR, LTR, DTE, **kwargs)
        scores = np.hastck((scores, llr))
        labels = np.hstack((labels, LTE))
    return min_DCF(scores, labels, wpoint[0], wpoint[1], wpoint[2])


def num_corrects(Pred, LTE):
    res_vec = Pred - LTE
    corr_pred = 0
    for i in range(res_vec.shape[0]):
        if res_vec[i] == 0:
            corr_pred += 1
    return corr_pred


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    x = np.array([[1, 3, 2, 5, 3, 1, 4, 9]]).reshape((4, 2))
    print(x)
    x_T = np.repeat(x, repeats=x.shape[0], axis=0)
    print(x_T)
    x_stacked = x
    for _ in range(x.shape[0] - 1):
        x_stacked = np.vstack((x_stacked, x))
    mapped = np.vstack(((x_stacked * x_T), x))
    print(mapped)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
