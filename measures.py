import numpy as np


def fnr(c_matrix):
    return c_matrix[0, 1] / (c_matrix[0, 1] + c_matrix[1, 1])


def fpr(c_matrix):
    return c_matrix[1, 0] / (c_matrix[1, 0] + c_matrix[0, 0])


def optimal_bayes_decision(log_ratios: np.ndarray, p_T, Cfn, Cfp, tresh=0, t_flag=False):
    if not t_flag:
        tresh = -(np.log(p_T * Cfn) - np.log((1 - p_T) * Cfp))
    pred = (log_ratios > tresh).astype(int)
    return pred


def bayes_empirical_risk(c_matrix, p_T, Cfn, Cfp):
    FNR = fnr(c_matrix)
    FPR = fpr(c_matrix)
    DCF = p_T * Cfn * FNR + (1 - p_T) * Cfp * FPR
    return DCF


def normalized_bayes_empirical_risk(c_matrix, p_T, Cfn, Cfp):
    return bayes_empirical_risk(c_matrix, p_T, Cfn, Cfp) / min(p_T * Cfn, (1 - p_T) * Cfp)


def confusion_matrix(pred, labels):
    n_class = int(np.amax(labels) + 1)
    pred = pred.reshape((1, pred.shape[0]))
    labels = labels.reshape((1, labels.shape[0]))
    c_matrix = np.zeros((n_class, n_class)).astype(int)
    for i in range(labels.shape[1]):
        c_matrix[int(pred[0, i]), int(labels[0, i])] += 1
    return c_matrix


def min_DCF(llr, labels, p_T, Cfn, Cfp):
    scores = [-np.inf, *sorted(llr), np.inf]
    dcf_min = np.inf
    for score in scores:
        pred = optimal_bayes_decision(llr, p_T, Cfn, Cfp, tresh=score, t_flag=True)
        c_mat = confusion_matrix(pred, labels)
        dcf_norm = normalized_bayes_empirical_risk(c_mat, p_T, Cfn, Cfp)
        if dcf_norm < dcf_min:
            dcf_min = dcf_norm
    return dcf_min

