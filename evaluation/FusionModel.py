import numpy as np
from Models import LogisticRegression, SVM, GMMClassifier, GaussianClassifiers
from itertools import combinations
from measures import *
from prettytable import PrettyTable, MARKDOWN

absolute_path = os.path.dirname(os.path.abspath(__file__))
score_path = f'{absolute_path}/../scores/train'
# define K for K-FOld Cross Validation
K = 5

# define working point
Cfn = 1
Cfp = 1
p_T = 0.5

wpoints = [(0.5, 1, 1), (0.2, 1, 1), (0.8, 1, 1)]

# import training data
DTR, LTR = load_dataset("Train.txt")
DTE, LTE = load_dataset("Test.txt")

alpha, tresh, psi = 0.1, 1e-6, 0.01

# calibrator obj
calibrator = LogisticRegression.LogisticRegression(lmd=0, prior=0.3)

# Load scores of best models from validation
# LR Linear lambda=10-3p_T=0.7
lr_scores, labels = np.load(f'{score_path}/LR_m{0}_preNone_prior{0.7}_lmd{1e-3}.npy')
lr = LogisticRegression.LogisticRegression(lmd=1e-3, prior=0.7)
lr.fit(DTR, LTR)
# RBFSVM p_T=0.7 gamma 0.001 C=10
rbfsvm_scores, _ = np.load(f'{score_path}/RBFSVM_m{0}_preNone_prior{0.7}_C{10}_k{1}_gamma{0.001}.npy')
rbfsvm = SVM.RBFSVM(p_T=0.7, C=10, k=1, gamma=0.001)
rbfsvm.fit(DTR, LTR)
# GMMTied K=4
gmm_scores, _ = np.load(f'{score_path}/GMMTied_G{4}_m{0}_preNone_a{alpha}_t{tresh}_psi{psi}.npy')
gmm = GMMClassifier.GMM(4, alpha, tresh, psi, diag=False, tied=True)
gmm.fit(DTR, LTR)

fuser = LogisticRegression.LogisticRegression(lmd=0, prior=0.3)
table = PrettyTable()
table.set_style(MARKDOWN)
table.field_names = ['Model', 'validation_minDCF', 'validation_actDCF', 'evaluation_minDCF', 'evaluation_actDCF']

for (name1, (scores1, clf1)), (name2, (scores2, clf2)) in combinations({"RBFSVM": (rbfsvm_scores, rbfsvm),
                                                                        "LR": (lr_scores, lr),
                                                                        "GMMTied": (gmm_scores, gmm)}.items(), 2):
    DCF = []
    # Validation set
    scores_fused = calibrate(np.vstack([scores1, scores2]), labels, 0.3)
    DCF.append(min_DCF(scores_fused, labels, 0.5, 1, 1))
    DCF.append(act_DCF(scores_fused, labels, 0.5, 1, 1))
    # Evaluation set
    fuser.fit(np.vstack([scores1, scores2]), labels)  # train on validation scores
    data1 = clf1.transform(DTE)  # compute scores for first model
    data2 = clf2.transform(DTE)  # compute scores for second model
    scores_fused_eval = fuser.transform(np.vstack([data1, data2]))  # compute scores on eval polled scores
    scores_fused_eval -= np.log(0.3 / (1 - 0.3))
    DCF.append(min_DCF(scores_fused_eval, LTE, 0.5, 1, 1))
    DCF.append(act_DCF(scores_fused_eval, LTE, 0.5, 1, 1))
    bayes_error_plot(scores_fused_eval, LTE, f'Eval {name1} + {name2}')
    table.add_row([f'{name1} + {name2}', *DCF])

# ALL three models
DCF = []
# Validation set
scores_fused = calibrate(np.vstack([rbfsvm_scores, gmm_scores, lr_scores]), labels, 0.3)
DCF.append(min_DCF(scores_fused, labels, 0.5, 1, 1))
DCF.append(act_DCF(scores_fused, labels, 0.5, 1, 1))
# Evaluation set
fuser.fit(np.vstack([rbfsvm_scores, gmm_scores, lr_scores]), labels)  # train on validation scores noKFOLD whole dataset
data1 = rbfsvm.transform(DTE)  # compute scores for first model
data2 = gmm.transform(DTE)  # compute scores for second model
data3 = lr.transform(DTE)  # compute scores for third model
scores_fused_eval = fuser.transform(np.vstack([data1, data2, data3]))  # compute scores on eval pooled scores
scores_fused_eval -= np.log(0.3 / (1 - 0.3))
DCF.append(min_DCF(scores_fused_eval, LTE, 0.5, 1, 1))
DCF.append(act_DCF(scores_fused_eval, LTE, 0.5, 1, 1))
bayes_error_plot(scores_fused_eval, LTE, f'Eval RBFSVM + GMMTied + LR')
table.add_row([f'RBFSVM + GMMTied + LR', *DCF])

print(table.get_string())
