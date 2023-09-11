import numpy as np
from utils import *
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


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


def act_DCF(llr, labels, p_T, Cfn, Cfp):
    pred = optimal_bayes_decision(llr, p_T, Cfn, Cfp)
    c_mat = confusion_matrix(pred, labels)
    dcf_norm = normalized_bayes_empirical_risk(c_mat, p_T, Cfn, Cfp)
    return dcf_norm


def bayes_error_plot(logRatios, labels, name):
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    make_dir("Images")
    make_dir("BayesErrorPlots")
    effPriorLogOdds = np.linspace(-4, 4, 100)
    effPrior = 1 / (1 + np.exp(-effPriorLogOdds))
    dcf = []
    min_dcf = []
    for prior in tqdm(effPrior):
        pred = optimal_bayes_decision(logRatios, prior, 1, 1)
        c_mat = confusion_matrix(pred, labels)
        dcf.append(normalized_bayes_empirical_risk(c_mat, prior, 1, 1))
        min_dcf.append(min_DCF(logRatios, labels, prior, 1, 1))
    dcf = np.array(dcf)
    mindcf = np.array(min_dcf)
    fig = plt.figure()
    plt.plot(effPriorLogOdds, dcf, label='DCF', color='r')
    plt.plot(effPriorLogOdds, mindcf, label='minDCF', color='b')
    plt.ylim([0, 1.3])
    plt.xlim([-4, 4])
    plt.savefig(f'{absolute_path}/Images/BayesErrorPlots/{name}.png')
    plt.close(fig)

