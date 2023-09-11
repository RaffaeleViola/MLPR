import numpy as np

from utils import *
from Models import GaussianClassifiers
from measures import *

absolute_path = os.path.dirname(os.path.abspath(__file__))
score_path = f'{absolute_path}/../scores/train'
# define K for K-FOld Cross Validation
K = 5

# define working point
Cfn = 1
Cfp = 1
p_T = 0.5

wpoint = (p_T, Cfn, Cfp)

# import training data
D, L = load_dataset("Train.txt")

# define PCA m values list
m_list = [0, 10, 11]  # example values - 0 mandatory for no PCA training

# define data_preprocessing strategies
pre_processing = {"None": None, "zscore": zscore}  # None is RAW data

# define classifiers
classifiers = [GaussianClassifiers.MVGClassifier, GaussianClassifiers.NaiveBayesClassifier]
classifier_map = ["MVG", "NaiveBayes"]

print(f'Classifier\t\t-\t\ttied\t\t-\t\tPCA(m)\t\t-\t\tPreprocess\t\t-\t\tminDCF\t\t')
for i, classfier in enumerate(classifiers):
    for tied in [False, True]:
        for m in m_list:
            for name_pre, pre_process in pre_processing.items():
                scores, labels = KFold_CV(D, L, 5, classfier, wpoint=wpoint, pca_m=m, pre_process=pre_process, tied=tied)
                np.save(f'{score_path}/{classifier_map[i]}_tied{tied}_m{m}_pre{name_pre}.npz', np.array([scores, labels]))
                minDCF = min_DCF(scores, labels, p_T, Cfn, Cfp)
                print(f'\n{classifier_map[i]}\t\t-\t\t{tied}\t\t-\t\tPCA({m})\t\t-\t\t{name_pre}\t\t-\t\t{minDCF}\t\t')

