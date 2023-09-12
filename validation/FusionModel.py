from Models import LogisticRegression, SVM, FusionModel
from measures import *


absolute_path = os.path.dirname(os.path.abspath(__file__))
score_path = f'{absolute_path}/../scores/train'
if not os.path.exists(score_path):
    os.mkdir(score_path)

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

clf = FusionModel.Fusion()
clf.fit(5, 0, 0.5)
scores = clf.transform([data1, data2], label1)

print(min_DCF(scores, label1, p_T, Cfn, Cfp))
bayes_error_plot(scores, label1, "FusionSVM_LR")

