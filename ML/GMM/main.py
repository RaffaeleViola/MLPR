import math
import numpy as np
import scipy
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import datasets


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


def save_gmm(gmm, filename):
    gmmJson = [(i, j.tolist(), k.tolist()) for i, j, k in gmm]
    with open(filename, 'w') as f:
        json.dump(gmmJson, f)


def load_gmm(filename):
    with open(filename, 'r') as f:
        gmm = json.load(f)
    return [(i, np.asarray(j), np.asarray(k)) for i, j, k in gmm]


def vcol(D):
    return D.reshape((D.shape[0], 1))


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


def logpdf_GMM(X, gmm):
    S = np.array([logpdf_GAU_ND(X, cluster[1], cluster[2]) for cluster in gmm])
    w = vcol(np.array([cluster[0] for cluster in gmm]))
    S += np.log(w)
    logdens = scipy.special.logsumexp(S, axis=0)
    return logdens, S


def expectation_maximization(X, gmm, tresh, psi, diag=False, tied=False):
    gmm_new = gmm
    logdens, S = logpdf_GMM(X, gmm_new)
    ll_avg_prev = logdens.sum() / logdens.shape[0]
    while True:
        # E-step
        S -= logdens
        Ygi = np.exp(S)
        # M-step
        Zg = Ygi.sum(axis=1)
        Fg = np.array([(Ygi[i, :] * X).sum(axis=1) for i in range(S.shape[0])]).T
        
        def compute_cov(a):
            tmp = vcol(a)
            return np.dot(tmp, tmp.T)

        tmp = np.apply_along_axis(compute_cov, 0, X)
        Sg = np.transpose(np.array([(Ygi[i, :] * tmp).sum(axis=2) for i in range(S.shape[0])]), (2, 1, 0))
        if diag:
            Sg = np.transpose(np.array([Sg[:, :, i] * np.eye(Sg[:, :, i].shape[0]) for i in range(Sg.shape[2])]), (2, 1, 0))
        elif tied:
            Sigma_tied = (Zg * Sg).sum(axis=2) / X.shape[1]
            Sg = np.transpose(np.array([Sigma_tied for _ in range(S.shape[0])]), (2, 1, 0))
        # constraining
        for i in range(Sg.shape[2]):
            U, s, _ = np.linalg.svd(Sg[:, :, i])
            s[s < psi] = psi
            Sg[:, :, i] = np.dot(U, vcol(s) * U.T)
        mu = Fg / vrow(Zg)
        cov = Sg / Zg - np.apply_along_axis(compute_cov, 0, mu)
        w = Zg / Zg.sum()
        gmm_new = [(w[i], vcol(mu[:, i]), cov[:, :, i]) for i in range(S.shape[0])]
        logdens, S = logpdf_GMM(X, gmm_new)
        ll_avg_new = logdens.sum() / logdens.shape[0]
        if ll_avg_new - ll_avg_prev < tresh:
            break
        ll_avg_prev = ll_avg_new

    return gmm_new


def LBG(X, G, alpha, tresh, psi, diag=False, tied=False):
    n_its = int(G / 2)
    mu = vcol(X.mean(1))
    C = np.dot(X, X.T) / X.shape[1]
    U, s, _ = np.linalg.svd(C)
    s[s < psi] = psi
    C = np.dot(U, vcol(s) * U.T)
    gmm = [(1.0, mu, C)]
    for g in range(n_its):
        gmm_new = []
        for cluster in gmm:
            U, s, Vh = np.linalg.svd(cluster[2])
            d = vcol(U[:, 0:1] * s[0] ** 0.5 * alpha)
            gmm_new.append((cluster[0] / 2, cluster[1] + d, cluster[2]))
            gmm_new.append((cluster[0] / 2, cluster[1] - d, cluster[2]))
        gmm = expectation_maximization(X, gmm_new, tresh, psi, diag=diag, tied=tied)
    return gmm


def fit(D, L, n_class, G, alpha, tresh, psi, diag=False, tied=False):
    params = [LBG(D[:, L == i], G, alpha, tresh, psi, diag, tied) for i in range(n_class)]
    return params


def compute_log_score_matrix(DTE, params, n_classes):
    score_matrix = []
    for i in range(n_classes):
        class_vec = []
        for j in range(DTE.shape[1]):
            logdens, _ = logpdf_GMM(vcol(DTE[:, j]), params[i])
            class_vec.append(logdens.sum())
        score_matrix.append(class_vec)
    return np.array(score_matrix)


def predict(DTE, params, n_classes):
    log_score_matrix = compute_log_score_matrix(DTE, params, n_classes)
    logSJoint = log_score_matrix + np.log(1 / 3)
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    SPost = np.exp(logSPost)
    Pred = np.argmax(SPost, axis=0)
    return Pred


def accuracy(Pred, LTE):
    res_vec = Pred - LTE
    corr_pred = 0
    for i in range(res_vec.shape[0]):
        if res_vec[i] == 0:
            corr_pred += 1
    acc = corr_pred / Pred.shape[0] * 100
    err = 100 - acc
    return acc, err


def main():
    D,  L = load_iris()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    params = fit(DTR, LTR, 3, 4, 0.1, 0.00001, 0.01, diag=False, tied=False)
    pred = predict(DTE, params, 3)
    acc, err = accuracy(pred, LTE)
    print(f'{acc=}%, {err=}%')


if __name__ == '__main__':
    main()






