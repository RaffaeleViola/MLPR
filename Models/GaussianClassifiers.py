from utils import *
import numpy as np
import math
import scipy

N_CLASS = 2


def compute_MVG_parameters(D, L):
    param_list = []
    for i in range(N_CLASS):
        X = D[:, L == i]
        mu = vcol(X.mean(1))
        C = np.dot((X - mu), (X - mu).T) / X.shape[1]
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
    llr = log_score_matrix[1, :] - log_score_matrix[0, :]
    return llr


