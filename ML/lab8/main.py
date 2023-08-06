import numpy as np
import scipy
from matplotlib import pyplot as plt


def load_data():
    dataset = np.load('commedia_llr_infpar.npy')
    labels = np.load('commedia_labels_infpar.npy')
    return dataset, labels


def mcol(v):
    return v.reshape((v.size, 1))


def compute_classPosteriors(S, logPrior=None):
    '''
    Compute class posterior probabilities

    S: Matrix of class-conditional log-likelihoods
    logPrior: array with class prior probability (shape (#cls, ) or (#cls, 1)). If None, uniform priors will be used

    Returns: matrix of class posterior probabilities
    '''

    if logPrior is None:
        logPrior = np.log(np.ones(S.shape[0]) / float(S.shape[0]))
    J = S + mcol(logPrior)  # Compute joint probability
    ll = scipy.special.logsumexp(J, axis=0)  # Compute marginal likelihood log f(x)
    P = J - ll  # Compute posterior log-probabilities P = log ( f(x, c) / f(x)) = log f(x, c) - log f(x)
    return np.exp(P)


def confusion_matrix(pred, labels):
    n_class = np.amax(labels) + 1
    pred = pred.reshape((1, pred.shape[0]))
    labels = labels.reshape((1, labels.shape[0]))
    c_matrix = np.zeros((n_class, n_class)).astype(int)
    for i in range(labels.shape[1]):
        c_matrix[pred[0, i], labels[0, i]] += 1
    return c_matrix


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


def fnr(c_matrix):
    return c_matrix[0, 1] / (c_matrix[0, 1] + c_matrix[1, 1])


def fpr(c_matrix):
    return c_matrix[1, 0] / (c_matrix[1, 0] + c_matrix[0, 0])


def ROC_curve(logRatios, label, tresh_list):
    FPR = np.array([fpr(confusion_matrix(optimal_bayes_decision(logRatios, 0, 0, 0, tresh=tresh, t_flag=True), label))
                    for tresh in tresh_list])
    TPR = np.array([1 - fnr(confusion_matrix(optimal_bayes_decision(logRatios, 0, 0, 0, tresh=tresh, t_flag=True), label))
                    for tresh in tresh_list])
    plt.figure()
    plt.plot(FPR, TPR)
    plt.grid()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.show()


def min_DCF(logRatios, p_T, Cfn, Cfp, scores):
    dcf_min = np.inf
    for score in scores:
        pred = optimal_bayes_decision(logRatios, p_T, Cfn, Cfp, tresh=score, t_flag=True)
        c_mat = confusion_matrix(pred, labels)
        dcf_norm = normalized_bayes_empirical_risk(c_mat, p_T, Cfn, Cfp)
        if dcf_norm < dcf_min:
            dcf_min = dcf_norm
    return dcf_min


def bayes_error_plot(logRatios, labels):
    effPriorLogOdds = np.linspace(-3, 3, 21)
    effPrior = 1 / (1 + np.exp(-effPriorLogOdds))
    score_list = [-np.inf, *sorted(logRatios), np.inf]
    dcf = []
    min_dcf = []
    for prior in effPrior:
        pred = optimal_bayes_decision(logRatios, prior, 1, 1)
        c_mat = confusion_matrix(pred, labels)
        dcf.append(normalized_bayes_empirical_risk(c_mat, prior, 1, 1))
        min_dcf.append(min_DCF(logRatios, prior, 1, 1, score_list))
    dcf = np.array(dcf)
    mindcf = np.array(min_dcf)
    plt.plot(effPriorLogOdds, dcf, label='DCF', color='r')
    plt.plot(effPriorLogOdds, mindcf, label='minDCF', color='b')
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.show()


if __name__ == '__main__':
    llr, labels = load_data()
    score_list = [-np.inf, *sorted(llr), np.inf]
    """ params = [(0.5, 1, 1), (0.8, 1, 1), (0.5, 10, 1), (0.8, 1, 10)]
    for param in params:
        dcf_min = np.inf
        for score in score_list:
            pred = optimal_bayes_decision(llr, *param, tresh=score)
            c_mat = confusion_matrix(pred, labels)
            dcf_norm = normalized_bayes_empirical_risk(c_mat, *param)
            if dcf_norm < dcf_min:
                dcf_min = dcf_norm
        print(f'{param} --- {dcf_min}')"""
    # ROC_curve(llr, labels, score_list)
    bayes_error_plot(llr, labels)
