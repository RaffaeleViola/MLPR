from utils import *
from Models import GMMClassifier
from measures import *


absolute_path = os.path.dirname(os.path.abspath(__file__))
score_path = f'{absolute_path}/../scores/test'
if not os.path.exists(score_path):
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
m_list = [0]  # example values - 0 mandatory for no PCA training

# define data_preprocessing strategies
pre_processing = {"zscore": zscore}  # None is RAW data

# define G list
G_list = [4]

# you can try other parameters setting alpha, tresh, psi prova ad esplorare
alpha, tresh, psi = 0.1, 1e-6, 0.01

scores = evaluation(DTR, LTR, DTE, GMMClassifier.GMM, m=11, pre_process=zscore, G=[4, 4], alpha=alpha,
                                    tresh=tresh, psi=psi, diag=False, tied=True)
minDCF = min_DCF(scores, LTE, p_T, Cfn, Cfp)
print(f'GMMTied zscore 4, 4: {minDCF}')


scores = evaluation(DTR, LTR, DTE, GMMClassifier.GMM, m=11, pre_process=zscore, G=[4, 4], alpha=alpha,
                                    tresh=tresh, psi=psi, diag=False, tied=True)
actDCF = act_DCF(scores, LTE, p_T, Cfn, Cfp)
print(f'GMMstd zscore 4, 4: {actDCF}')


# print(f'tied\t\t-\t\tPCA(m)\t\t-\t\tPreprocess\t\t-\t\tG(g)\t\t-\t\tminDCF\t\t')
# for tied in [False, True]:
#     for m in m_list:
#         for name_pre, pre_process in pre_processing.items():
#             for G in G_list:
#                 scores = evaluation(DTR, LTR, DTE, GMMClassifier.GMM, m=m, pre_process=pre_process, G=G, alpha=alpha,
#                                     tresh=tresh, psi=psi, diag=False, tied=tied)
#                 minDCF = min_DCF(scores, LTE, p_T, Cfn, Cfp)
#                 classifier = "StandardGMM" if tied is False else "TiedGMM"
#                 print(f'{classifier}\t\t-\t\tPCA({m})\t\t-\t\t{name_pre}\t\t-\t\tG({G})\t\t-\t\t{minDCF}\t\t')
#
#
# for m in m_list:
#     for name_pre, pre_process in pre_processing.items():
#         for G in G_list:
#             scores = evaluation(DTR, LTR, DTE, GMMClassifier.GMM, m=m, pre_process=pre_process,
#                                 G=G, alpha=alpha, tresh=tresh, psi=psi, diag=True, tied=False)
#             minDCF = min_DCF(scores, LTE, p_T, Cfn, Cfp)
#             print(f'DiagonalGMM\t\t-\t\tPCA({m})\t\t-\t\t{name_pre}\t\t-\t\t{minDCF}\t\t')

