import numpy as np
from Models import LogisticRegression, SVM
from utils import *
from measures import *


# define K for K-FOld Cross Validation
K = 5

# define working point
Cfn = 1
Cfp = 1
p_T = 0.5

wpoint = (p_T, Cfn, Cfp)

D, L = load_dataset("Train.txt")

# choose model to retrieve
data1, label1 = KFold_CV(D, L, K, SVM.SVM,
                   wpoint=wpoint, pca_m=0, pre_process=zscore, p_T=0.9, C=10, k=1)

data2, label2 = KFold_CV(D, L, K, LogisticRegression.LogisticRegression,
                   wpoint=wpoint, pca_m=0, pre_process=zscore, lmd=0, prior=0.9)
scores = np.array(np.vstack([data1, data2]))
labels = label1

scores, labels = KFold_CV(scores, labels, K, LogisticRegression.LogisticRegression,
                          wpoint=wpoint, pca_m=0, seed=13, pre_process=None, lmd=0, prior=0.5)

print(min_DCF(scores, labels, p_T, Cfn, Cfp))
bayes_error_plot(scores, labels, "FusionSVM_LR")

