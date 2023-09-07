from utils import *
from Models import GaussianClassifiers

# define K for K-FOld Cross Validation
K = 5

# define working point
Cfn = 1
Cfp = 1
p_T = 0.5

wpoint = (p_T, Cfn, Cfp)

# import training data
D = []
L = []

# define PCA m values list
m_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # example values - 0 mandatory for no PCA training

# define data_preprocessing strategies
pre_processing = {"None": None, "zscore": zscore}  # None is RAW data

# define classifiers
classifiers = [GaussianClassifiers.MVG_classifier, GaussianClassifiers.Naive_Bayes_Classifier]
classifier_map = ["MVG", "NaiveBayes"]

print(f'Classifier\t\t-\t\ttied\t\t-\t\tPCA(m)\t\t-\t\tPreprocess\t\t-\t\tminDCF\t\t')
for i, classfier in enumerate(classifiers):
    for tied in [False, True]:
        for m in m_list:
            for name_pre, pre_process in pre_processing.items():
                minDCF = KFold_CV(D, L, 5, classfier, wpoint=wpoint, pca_m=m, pre_process=pre_process, tied=tied)
                print(f'{classifier_map[i]}\t\t-\t\t{tied}\t\t-\t\tPCA({m})\t\t-\t\t{name_pre}\t\t-\t\t{minDCF}\t\t')

