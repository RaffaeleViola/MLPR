import math
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy

N_ATTR = 10
N_CLASS = 2


def load(path):
    matrix = list()
    labels = list()

    with open(path, "r") as f:
        for line in f:
            words = line.split(",")
            label = int(words[N_ATTR].rstrip())
            matrix.append(np.array([float(words[i]) for i in range(N_ATTR)],
                                   dtype=np.float32).reshape(N_ATTR, 1))
            labels.append(label)
    return np.concatenate(matrix, axis=1, dtype=np.float32), np.array(labels)


def plot_hist(dataset, labels, path, label="original"):
    mask0 = (labels == 0.0)
    d0 = dataset[:, mask0]
    mask1 = (labels == 1.0)
    d1 = dataset[:, mask1]
    plt.figure()
    for attr in range(N_ATTR):
        plt.hist(d0[attr, :], bins=30, density=True, ec='black', color='Blue', alpha=0.5, label="Spoofed")
        plt.hist(d1[attr, :], bins=30, density=True, ec='black', color='Red', alpha=0.5, label="Authentic")
        plt.legend()
        plt.title(f'feature: {attr+1}_{label}')
        file_name = f'/feature_{attr+1}_{label}.png'
        plt.savefig(path + file_name)
        plt.show()


def plot_scatter(dataset, labels):
    mask0 = (labels == 0)
    d0 = dataset[:, mask0]
    mask1 = (labels == 1)
    d1 = dataset[:, mask1]
    plt.figure()
    for i in range(N_ATTR):
        f, ax = plt.subplots(10, 1)
        n = 0
        for j in range(N_ATTR):
            if i == j:
                continue
            ax[n].scatter(d0[i, :], d0[j, :])
            ax[n].scatter(d1[i, :], d1[j, :])
            n += 1
        plt.show()


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


def gaussianization(dataset):
    mu = vcol(dataset.mean(1))
    std = vcol(np.std(dataset, axis=1))
    return (dataset - mu) / std


def covariance(dataset):
    return dataset.dot(dataset.T) / dataset.shape[1]


def vrow(v):
    return v.reshape((1, v.shape[0]))


def vcol(v):
    return v.reshape((v.shape[0], 1))


def compute_MVG_parameters(D, L):
    param_list = []
    for i in range(N_CLASS):
        DC, mu = center_data(D[:, L == i])
        C = covariance(DC)
        param_list.append((mu, C))
    return param_list


def compute_MVG_tied_parameters(D, L):
    mean_list = []
    C = 0
    for i in range(N_CLASS):
        DC, mu = center_data(D[:, L == i])
        C += np.dot(DC, DC.T)
        mean_list.append(mu)
    C = C / D.shape[1]
    param_list = [(mean_list[i], C) for i in range(N_CLASS)]
    return param_list


def logpdf_GAU_ND(x, mu, C):
    log_vect = []
    for i in range(x.shape[1]):
        e1 = (C.shape[0] / 2) * math.log(2 * math.pi)
        e2 = np.linalg.slogdet(C)[1] / 2
        t = np.dot((x[:, i:i + 1] - mu).T, np.linalg.inv(C))
        e3 = 0.5 * np.dot(t, (x[:, i:i + 1] - mu))
        log_vect.append((-e1 - e2 - e3))
    return np.array(log_vect).ravel()


def loglikelihood(D, mu, C):
    return logpdf_GAU_ND(D, mu, C).sum()


def compute_log_score_matrix(TRD, params):
    score_matrix = []
    for i in range(N_CLASS):
        class_vec = []
        for j in range(TRD.shape[1]):
            ##class_vec.append(np.exp(loglikelihood(vcol(TRD[:, j]), params[i][0], params[i][1])))
            class_vec.append(loglikelihood(vcol(TRD[:, j]), params[i][0], params[i][1]))
        score_matrix.append(class_vec)
    return np.array(score_matrix)


def compute_SJoint(pdf_vec, Pc):
    return pdf_vec * Pc


def MVG_classifier(DTR, LTR, DTE, tied=False):
    if tied:
        params = compute_MVG_tied_parameters(DTR, LTR)
    else:
        params = compute_MVG_parameters(DTR, LTR)
    return transform(DTE, params)


def Naive_Bayes_Classifier(DTR, LTR, DTE, tied=False):
    if tied:
        params = compute_MVG_tied_parameters(DTR, LTR)
    else:
        params = compute_MVG_parameters(DTR, LTR)
    new_params = []
    for i in range(len(params)):
        mu, C = params[i]
        Id = np.eye(C.shape[0])
        new_C = C * Id
        new_params.append((mu, new_C))
    return transform(DTE, new_params)


def transform(DTE, params):
    log_score_matrix = compute_log_score_matrix(DTE, params)
    # normal computation
    #logSJoint = compute_SJoint(log_score_matrix, 1 / 3)
    # SMarginal = vrow(SJoint.sum(0))
    # SPost = SJoint / SMarginal
    # SPost = SJoint / SMarginal
    #logsumexp computation
    logSJoint = log_score_matrix + np.log(1/2) #posterior probability
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    SPost = np.exp(logSPost)
    Pred = np.argmax(SPost, axis=0)
    return Pred


def num_corrects(Pred, LTE):
    res_vec = Pred - LTE
    corr_pred = 0
    for i in range(res_vec.shape[0]):
        if res_vec[i] == 0:
            corr_pred += 1
    return corr_pred


def KFold_CV(D, L, K, Classifier, seed=0):
    nTest = int(D.shape[1] / K)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    corr_pred = 0
    for i in range(K):
        start = nTest * i
        idxTrain = np.concatenate((idx[0:start], idx[(start + nTest):]))
        idxTest = idx[start: (start + nTest)]
        DTR = D[:, idxTrain]
        DTE = D[:, idxTest]
        LTR = L[idxTrain]
        LTE = L[idxTest]
        pred = Classifier(DTR, LTR, DTE, tied=False)
        corr_pred += num_corrects(pred, LTE)
    return corr_pred / D.shape[1] * 100


def accuracy(Pred, LTE):
    res_vec = Pred - LTE
    corr_pred = 0
    for i in range(res_vec.shape[0]):
        if res_vec[i] == 0:
            corr_pred += 1
    acc = corr_pred / Pred.shape[0] * 100
    err = 100 - acc
    return acc, err


def plot_heatmap(dataset, path, label="all"):
    ds = pd.DataFrame(dataset.T)
    rounded_corr_matrix = ds.corr().round(2)
    heatmap = sns.heatmap(rounded_corr_matrix, annot=True)
    heatmap.set_title(f'Correlation Heatmap {label}', fontdict={'fontsize': 12}, pad=12)
    plt.savefig(f'{path}/heat_{label}.png')
    plt.show()



if __name__ == '__main__':
    current_path = os.getcwd()
    DTR, LTR = load("Train.txt")
    DTE, LTE = load("Test.txt")
    # path = f'{current_path}/figures/hists_no_centered'
    # plot_hist(DTR, LTR, path, label="original")

    # DC, mu = center_data(DTR)
    # path = f'{current_path}/figures/hists_centered'
    # plot_hist(DC, LTR, path, label="centered")
    # path = f'{current_path}/figures/hists_gaussianized'
    # GD = gaussianization(DTR)
    # plot_hist(GD, LTR, path, label="gauss")

    path = f'{current_path}/figures/heatmaps'
    plot_heatmap(DTR, path)
    D0 = DTR[:, LTR == 0]
    plot_heatmap(D0, path, label="Spoofed")
    D1 = DTR[:, LTR == 1]
    plot_heatmap(D1, path, label="Authentic")















