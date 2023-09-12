import numpy as np

from utils import *
from Models import GaussianClassifiers
from measures import *

absolute_path = os.path.dirname(os.path.abspath(__file__))
score_path = f'{absolute_path}/../scores/test'
os.mkdir(score_path)
# define K for K-FOld Cross Validation
K = 5

# define working point
Cfn = 1
Cfp = 1
p_T = 0.5

wpoint = (p_T, Cfn, Cfp)

# import training data
DTR, LTR = load_dataset("Train.txt")
DTE, LTE = load_dataset("Test.txt")

# define PCA m values list
m_list = [0, 10, 11]  # example values - 0 mandatory for no PCA training

# define data_preprocessing strategies
pre_processing = {"None": None, "zscore": zscore}  # None is RAW data

# define classifiers
classifiers = [GaussianClassifiers.MVGClassifier, GaussianClassifiers.NaiveBayesClassifier]
classifier_map = ["MVG", "NaiveBayes"]

print(f'Evaluation: Classifier\t\t-\t\ttied\t\t-\t\tPCA(m)\t\t-\t\tPreprocess\t\t-\t\tminDCF\t\t')
PCA_reducer = PCA()
for i, classfier in enumerate(classifiers):
    for tied in [False, True]:
        for m in m_list:
            for name_pre, pre_process in pre_processing.items():
                DTR_r = DTR
                DTE_r = DTE
                if m != 0:
                    PCA_reducer.fit(DTR)
                    DTR_r = PCA_reducer.transform(m, DTR)
                    DTE_r = PCA_reducer.transform(m, DTE)
                clf = classfier(tied=tied)
                clf.fit(DTR_r, LTR)
                scores = clf.transform(DTE_r)
                minDCF = min_DCF(scores, LTE, p_T, Cfn, Cfp)
                print(f'\nEvaluation: {classifier_map[i]}\t\t-\t\t{tied}\t\t-\t\tPCA({m})\t\t-\t\t{name_pre}\t\t-\t\t{minDCF}\t\t')

