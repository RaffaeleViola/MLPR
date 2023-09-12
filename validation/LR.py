from Models import LogisticRegression
from utils import *
import matplotlib.pyplot as plt
from tqdm import tqdm
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

# import training data
D, L = load_dataset("Train.txt")

# define PCA m values list
m_list = [0, 10, 11]  # example values - 0 mandatory for no PCA training

# define data_preprocessing strategies
pre_processing = {"None": None, "zscore": zscore}  # None is RAW data

# define lambda range list
lambdas = np.logspace(-5, 5, num=10)  # per la quadratic metti num=13/15 perchè ci mette più tempo e ti esplode il pc

make_dir("LRPlot")
# Training and Validation LR
minDCF = []
for lmd in tqdm(lambdas):
    scores, labels = KFold_CV(D, L, K, LogisticRegression.LogisticRegression,
                   wpoint=wpoint, pca_m=0, pre_process=None, lmd=lmd, prior=p_T)
    minDCF.append(min_DCF(scores, labels, p_T, Cfn, Cfp))

fig = plt.figure()
plt.plot(lambdas, minDCF)
plt.xscale('log')
plt.xlim(lambdas[0], lambdas[-1])
plt.savefig(f'{absolute_path}/../Images/LRPlot/LR_prior{p_T}.png')
plt.close(fig)

# Training and Validation QLR
minDCF = []
for lmd in tqdm(lambdas):
    scores, labels = KFold_CV(D, L, K, LogisticRegression.QuadraticLogisticRegression,
                   wpoint=wpoint, pca_m=0, pre_process=None, lmd=lmd, prior=p_T)
    minDCF.append(min_DCF(scores, labels, p_T, Cfn, Cfp))

fig = plt.figure()
plt.plot(lambdas, minDCF)
plt.xscale('log')
plt.xlim(lambdas[0], lambdas[-1])
plt.savefig(f'{absolute_path}/../Images/LRPlot/QLR_prior{p_T}.png')
plt.close(fig)

exit(0)  # cancel when you want to train the chosen models
# After you choose the model parameter train and save what you think is useful
#STD LOG REG
# lmd = ..
# m = ..
# preprocess = ..
# p_T = ..
# scores, labels = KFold_CV(D, L, K, LogisticRegression.LogisticRegression,
#                    wpoint=wpoint, pca_m=0, pre_process=None, lmd=lmd, prior=p_T)
# np.save(f'{score_path}/LR_lmd{lmd}_m{m}_{preprocess}_prior{p_T}.npz',
#                         np.array([scores, labels]))


#Quadratic LOG REG
# lmd = ..
# m = ..
# preprocess = ..
# p_T = ..
# scores, labels = KFold_CV(D, L, K, LogisticRegression.QuadraticLogisticRegression,
#                    wpoint=wpoint, pca_m=0, pre_process=None, lmd=lmd, prior=p_T)
# np.save(f'{score_path}/LR_lmd{lmd}_m{m}_{preprocess}_prior{p_T}.npz',
#                         np.array([scores, labels]))

