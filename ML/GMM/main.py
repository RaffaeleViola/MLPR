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
        e1 = (C.shape[0] / 2) * math.log(2 * math.pi)
        e2 = np.linalg.slogdet(C)[1] / 2
        t = np.dot((x[:, i:i + 1] - mu).T, np.linalg.inv(C))
        e3 = 0.5 * np.dot(t, (x[:, i:i + 1] - mu))
        log_vect.append((-e1 - e2 - e3))
    return np.array(log_vect).ravel()


def logpdf_GMM(X, gmm):
    S = np.array([logpdf_GAU_ND(X, cluster[1], cluster[2]) for cluster in gmm])
    w = vcol(np.array([cluster[0] for cluster in gmm]))
    S += np.log(w)
    logdens = scipy.special.logsumexp(S, axis=0)
    return logdens, S


def constraining(sigma, psi):
    U, s, _ = np.linalg.svd(sigma)
    s[s < psi] = psi
    sigma = np.dot(U, vcol(s) * U.T)
    return sigma


def expectation_maximization(X, gmm, tresh, psi, diag=False, tied=False):
    logdens, S = logpdf_GMM(X, gmm)
    ll_avg_prev = logdens.sum() / logdens.shape[0]
    while True:
        # E-step
        S -= logdens
        Ygi = np.exp(S)
        # M-step
        zg_vec = []
        mu_vec = []
        sigma_vec = []
        for g in range(len(gmm)):
            Yg = Ygi[g, :]
            Zg = Yg.sum()
            zg_vec.append(Zg)
            tmp = vrow(Yg) * X
            Fg = tmp.sum(axis=1)
            Sg = np.dot(X, tmp.T)
            mu = vcol(Fg / Zg)
            sigma = Sg / Zg - np.dot(mu, mu.T)
            if diag:
                sigma = sigma * np.eye(sigma.shape[0])
            # constraining
            if not tied:
                sigma = constraining(sigma, psi)
            mu_vec.append(mu)
            sigma_vec.append(sigma)
        zg_sum = sum(zg_vec)
        if tied:
            tied_sum = 0
            for i in range(len(zg_vec)):
                tied_sum += zg_vec[i] * sigma_vec[i]
            sigma_tied = tied_sum / X.shape[1]
            sigma_tied = constraining(sigma_tied, psi)
            gmm_new = [(zg_vec[i] / X.shape[1], mu_vec[i], sigma_tied) for i in range(len(gmm))]
        else:
            gmm_new = [(zg_vec[i] / X.shape[1], mu_vec[i], sigma_vec[i]) for i in range(len(gmm))]
        logdens, S = logpdf_GMM(X, gmm_new)
        ll_avg_new = logdens.sum() / logdens.shape[0]
        if ll_avg_new - ll_avg_prev < tresh:
            break
        ll_avg_prev = ll_avg_new

    return gmm_new


def LBG(X, G, alpha, tresh, psi, diag=False, tied=False):
    n_its = int(math.log2(G))
    mu = vcol(X.mean(1))
    C = np.dot((X - mu), (X - mu).T) / X.shape[1]
    if diag:
        C = C * np.eye(C.shape[0])
    # constraining
    C = constraining(C, psi)
    gmm = [(1.0, mu, C)]
    for g in range(n_its):
        gmm_new = []
        for cluster in gmm:
            U, s, Vh = np.linalg.svd(cluster[2])
            d = vcol(U[:, 0:1] * s[0]**0.5 * alpha)
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


def center_data(dataset):
    mu = vcol(dataset.mean(1))
    centered_dataset = dataset - mu
    return centered_dataset, mu


def main():
    D, L = load_iris()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    for G in [1, 2, 4, 8, 16]:
        params = fit(DTR, LTR, 3, G, 0.1, 1e-6, 0.01, diag=False, tied=True)
        pred = predict(DTE, params, 3)
        acc, err = accuracy(pred, LTE)
        print(f'{len(params[0])} -  {acc=}%, {err=}%')


if __name__ == '__main__':
    main()
