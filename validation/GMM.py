from utils import *
from Models import GMMClassifier
from measures import *

# define K for K-FOld Cross Validation
K = 5

# define working point
Cfn = 1
Cfp = 1
p_T = 0.5

wpoint = (p_T, Cfn, Cfp)

# import training data
D, L = load_dataset()

# define PCA m values list
m_list = [0, 10, 11]  # example values - 0 mandatory for no PCA training

# define data_preprocessing strategies
pre_processing = {"None": None, "zscore": zscore}  # None is RAW data

# define G list
G_list = [4, 8]

# you can try other parameters setting alpha, tresh, psi prova ad esplorare
alpha, tresh, psi = 0.1, 1e-6, 0.01


print(f'tied\t\t-\t\tPCA(m)\t\t-\t\tPreprocess\t\t-\t\tG(g)\t\t-\t\tminDCF\t\t')

for tied in [False, True]:
    for m in m_list:
        for name_pre, pre_process in pre_processing.items():
            for G in G_list:
                scores, labels = KFold_CV(D, L, 5, GMMClassifier.GMM, wpoint=wpoint, pca_m=m, pre_process=pre_process,
                                  G=G, alpha=alpha, tresh=tresh, psi=psi, diag=False, tied=tied)
                minDCF = min_DCF(scores, labels, p_T, Cfn, Cfp)
                classifier = "StandardGMM" if tied is False else "TiedGMM"
                print(f'{classifier}\t\t-\t\tPCA({m})\t\t-\t\t{name_pre}\t\t-\t\tG({G})\t\t-\t\t{minDCF}\t\t')

for m in m_list:
    for name_pre, pre_process in pre_processing.items():
        for G in G_list:
            scores, labels = KFold_CV(D, L, 5, GMMClassifier.GMM, wpoint=wpoint, pca_m=m, pre_process=pre_process,
                              G=G, alpha=alpha, tresh=tresh, psi=psi, diag=True, tied=False)
            minDCF = min_DCF(scores, labels, p_T, Cfn, Cfp)
            print(f'DiagonalGMM\t\t-\t\tPCA({m})\t\t-\t\t{name_pre}\t\t-\t\t{minDCF}\t\t')

