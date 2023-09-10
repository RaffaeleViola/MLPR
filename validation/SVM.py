from Models import SVM
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
C_list = np.logspace(-5, 5, num=15)


print(KFold_CV(D, L, K, SVM.linear_svm, wpoint=wpoint, pca_m=0, pre_process=None, p_T=p_T, C=10, k=1))
exit(0)
# Training and Validation
minDCF = np.array([KFold_CV(D, L, K, SVM.linear_svm,
                   wpoint=wpoint, pca_m=0, pre_process=None, p_T=p_T, C=C, k=1) for C in tqdm(C_list)])

plt.plot(C_list, minDCF)
plt.xscale('log')
plt.xlim(C_list[0], C_list[-1])
plt.show()

exit(0)
