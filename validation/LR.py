from Models import LogisticRegression
from utils import *
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

# define PCA m values list
m_list = [0, 10, 11]  # example values - 0 mandatory for no PCA training

# define data_preprocessing strategies
pre_processing = {"None": None, "zscore": zscore}  # None is RAW data

# define lambda range list
lambdas = np.logspace(-5, 5, num=40)  # per la quadratic metti num=13/15 perchè ci mette più tempo e ti esplode il pc


# Training and Validation
minDCF = np.array([KFold_CV(D, L, K, LogisticRegression.binarylogreg,
                   wpoint=wpoint, pca_m=0, pre_process=None, lmd=lmd, prior=p_T) for lmd in tqdm(lambdas)])

plt.plot(lambdas, minDCF)
plt.xscale('log')
plt.xlim(lambdas[0], lambdas[-1])
plt.show()

exit(0)

