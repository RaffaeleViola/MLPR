from Models import SVM, LogisticRegression
from utils import *
from measures import *
import matplotlib.pyplot as plt
from tqdm import tqdm

# define K for K-FOld Cross Validation
K = 5

# define working point
Cfn = 1
Cfp = 1
p_T = 0.5

wpoint = (p_T, Cfn, Cfp)

# import training data
D, L = load_dataset()



# define data_preprocessing strategies
pre_processing = zscore  # None is RAW data

# define lambda range list
C = 10


# Logistic Regression
# scores, labels = KFold_CV(D, L, K, LogisticRegression.binarylogreg,
#                    wpoint=wpoint, pca_m=0, pre_process=zscore, lmd=0, prior=0.9)
# print(f'\nmin_DCF_non_calibrated: {min_DCF(scores, labels, p_T, Cfn, Cfp)}')
# print(f'\nactDCF_non_calibrated: {act_DCF(scores, labels, p_T, Cfn, Cfp)}')
# bayes_error_plot(scores, labels, "LR_non_calibrated_zscore")
# scores, labels = KFold_CV(vrow(scores), labels, K, LogisticRegression.binarylogreg,
#                    wpoint=wpoint, pca_m=0, seed=13, pre_process=None, lmd=0, prior=0.5)
# scores = scores - np.log(p_T / (1 - p_T))
# bayes_error_plot(scores, labels, "LR_calibrated_zscore")

# SVM
scores, labels = KFold_CV(D, L, K, SVM.linear_svm, wpoint=wpoint, pca_m=0, pre_process=zscore, p_T=0.9, C=10, k=1)
bayes_error_plot(scores, labels, "SVM_non_calibrated_zscore")
scores, labels = KFold_CV(vrow(scores), labels, K, LogisticRegression.binarylogreg,
                   wpoint=wpoint, pca_m=0, seed=13, pre_process=None, lmd=0, prior=0.5)
scores = scores - np.log(p_T / (1 - p_T))
print(f'\nmin_DCF_calibrated: {min_DCF(scores, labels, p_T, Cfn, Cfp)}')
print(f'\nactDCF_calibrated: {act_DCF(scores, labels, p_T, Cfn, Cfp)}')
bayes_error_plot(scores, labels, "SVM_calibrated_zscore")



# RBSVM

# scores, labels = KFold_CV(D, L, K, SVM.RBF_svm,
#                    wpoint=wpoint, pca_m=0, pre_process=zscore, p_T=0.9, C=5, k=1, gamma=0.1)
# print(f'min_DCF_non_calibrated: {min_DCF(scores, labels, p_T, Cfn, Cfp)}')
# print(f'actDCF_non_calibrated: {act_DCF(scores, labels, p_T, Cfn, Cfp)}')
# bayes_error_plot(scores, labels, "RBSVM_non_calibrated_zscore")
# scores, labels = KFold_CV(vrow(scores), labels, K, LogisticRegression.binarylogreg,
#                    wpoint=wpoint, pca_m=0, seed=13, pre_process=None, lmd=0, prior=0.5)
# scores = scores - np.log(p_T / (1 - p_T))
# bayes_error_plot(scores, labels, "RBSVM_calibrated_zscore")


