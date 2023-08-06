import math
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets


def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris() ['target']
    return D, L


def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)


def vrow(D):
    return D.reshape((1, D.shape[0]))


def logpdf_GAU_ND(x, mu, C):
    log_vect = []
    for i in range(x.shape[1]):
        e1 = (C.shape[0]/2)*math.log(2*math.pi)
        e2 = np.linalg.slogdet(C)[1] / 2
        t = np.dot((x[:, i:i+1] - mu).T, np.linalg.inv(C))
        e3 = 0.5 * np.dot(t, (x[:, i:i+1] - mu))
        log_vect.append((-e1-e2-e3))
    return np.array(log_vect).ravel()


def loglikelihood(D, mu, C):
    return logpdf_GAU_ND(D, mu, C).sum()


def main():
    XND = np.load('X1D.npy')
    m_ML = XND.mean(1).reshape((XND.shape[0], 1))
    XND_C = XND - m_ML
    C_ML = np.dot(XND_C, XND_C.T) / XND_C.shape[1]
    plt.figure()
    # plt.hist(XND.ravel(), bins=50, density=True)
    XPlot = np.linspace(-8, 12, 1000)
    print(logpdf_GAU_ND(vrow(XPlot), m_ML, C_ML).shape)
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(vrow(XPlot), m_ML, C_ML)))
    plt.show()


main()
