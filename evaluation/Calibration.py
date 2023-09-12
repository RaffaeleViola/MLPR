from Models import SVM, LogisticRegression
from measures import *


def calibrate(scores, labels, name, pT):
    scores, labels = KFold_CV(vrow(scores), labels, K, LogisticRegression.LogisticRegression,
                              wpoint=wpoint, pca_m=0, seed=13, pre_process=None, lmd=0, prior=0.5)
    scores = scores - np.log(pT / (1 - pT))
    print(f'\n{name}_min_DCF: {min_DCF(scores, labels, pT, Cfn, Cfp)}')
    print(f'\n{name}_actDCF: {act_DCF(scores, labels, pT, Cfn, Cfp)}')
    bayes_error_plot(scores, labels, name)
    return scores


# define K for K-FOld Cross Validation
K = 5

# define working point
Cfn = 1
Cfp = 1
p_T = 0.5

wpoint = (p_T, Cfn, Cfp)

# import training data
D, L = load_dataset("Train.txt")



# define data_preprocessing strategies
pre_processing = zscore  # None is RAW data

# define lambda range list
C = 10


# Logistic Regression
# s, lab = KFold_CV(D, L, K, LogisticRegression.LogisticRegression,
#                    wpoint=wpoint, pca_m=0, pre_process=zscore, lmd=0, prior=0.9)
# bayes_error_plot(s, lab, "LR_non_calibrated_zscore")
# calibrate(s, lab, "LR_calibrated_zscore", p_T)


# SVM
# s, lab = KFold_CV(D, L, K, SVM.SVM, wpoint=wpoint, pca_m=0, pre_process=zscore, p_T=0.9, C=10, k=1)
# bayes_error_plot(s, lab, "SVM_non_calibrated_zscore")
# calibrate(s, lab, "SVM_calibrated_zscore", p_T)



# RBSVM

s, lab = KFold_CV(D, L, K, SVM.RBFSVM,
                   wpoint=wpoint, pca_m=0, pre_process=zscore, p_T=0.9, C=5, k=1, gamma=0.1)
bayes_error_plot(s, lab, "RBSVM_non_calibrated_zscore")
calibrate(s, lab, "RBSVM_calibrated_zscore", p_T)



