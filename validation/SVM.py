from Models import SVM
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
C_list = np.logspace(-5, 5, num=15)

make_dir("SVMPlot")
# Training and Validation LinearSVM
minDCF = []
for C in tqdm(C_list):
    scores, labels = KFold_CV(D, L, K, SVM.SVM,
                   wpoint=wpoint, pca_m=0, pre_process=None, p_T=p_T, C=C, k=1)
    minDCF.append(min_DCF(scores, labels, p_T, Cfn, Cfp))

fig = plt.figure()
plt.plot(C_list, minDCF)
plt.xscale('log')
plt.xlim(C_list[0], C_list[-1])
plt.savefig(f'{absolute_path}/../Images/SVMPlot/SVMLinear_prior{p_T}.png')
plt.close(fig)

# polynomial SVM
# minDCF = []
# for C in tqdm(C_list):
#    scores, labels = KFold_CV(D, L, K, SVM.PolynomialSVM,
#                    wpoint=wpoint, pca_m=0, pre_process=None, p_T=p_T, C=C, k=1)  # change parameters
#     minDCF.append(min_DCF(scores, labels, p_T, Cfn, Cfp))
#
# fig = plt.figure()
# plt.plot(C_list, minDCF)
# plt.xscale('log')
# plt.xlim(C_list[0], C_list[-1])
# plt.savefig(f'{absolute_path}/../Images/SVMPlot/SVMPolynomial_prior{p_T}.png')
# plt.close(fig)


# Training and Validation RBFSVM
# minDCF = []
# for C in tqdm(C_list):
#    scores, labels = KFold_CV(D, L, K, SVM.RBFSVM,
#                    wpoint=wpoint, pca_m=0, pre_process=None, p_T=p_T, C=C, k=1)  # change parameters
#     minDCF.append(min_DCF(scores, labels, p_T, Cfn, Cfp))
#
# fig = plt.figure()
# plt.plot(C_list, minDCF)
# plt.xscale('log')
# plt.xlim(C_list[0], C_list[-1])
# plt.savefig(f'{absolute_path}/../Images/SVMPlot/RBFSVM_prior{p_T}.png')
# plt.close(fig)


# After you choose the model parameter train and save what you think is useful
# STD SVM
# # change parameters as needed
# C = ..
# k = ..
# m = ..
# preprocess = ..
# p_T = ..
# scores, labels = KFold_CV(D, L, K, SVM.SVM,
#                    wpoint=wpoint, pca_m=0, pre_process=None, p_T=p_T, C=C, k=1)
# np.save(f'{score_path}/SVMLinear_m{m}_pre{preprocess}_prior{p_T}_C{C}_k{k}.npz',
#         np.array([scores, labels]))


# After you choose the model parameter train and save what you think is useful
# Polynomial SVM
# # change parameters as needed
# C = ..
# k = ..
# m = ..
# preprocess = ..
# p_T = ..
# scores, labels = KFold_CV(D, L, K, SVM.PolynomialSVM,
#                    wpoint=wpoint, pca_m=0, pre_process=None, p_T=p_T, C=C, k=1)
# np.save(f'{score_path}/SVMPol_m{m}_pre{preprocess}_prior{p_T}_C{C}_k{k}.npz', # change parameters name please
#         np.array([scores, labels]))


# After you choose the model parameter train and save what you think is useful
#  RBFSVM
# # change parameters as needed
# C = ..
# k = ..
# m = ..
# preprocess = ..
# p_T = ..
# scores, labels = KFold_CV(D, L, K, SVM.RBFSVM,
#                    wpoint=wpoint, pca_m=0, pre_process=None, p_T=p_T, C=C, k=1)
# np.save(f'{score_path}/RBFSVM_m{m}_pre{preprocess}_prior{p_T}_C{C}_k{k}.npz', # change parameters name please
#         np.array([scores, labels]))

exit(0)
