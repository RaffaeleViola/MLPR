from Models import SVM
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
m_list = [0, 10, 11]  # example values - 0 mandatory for no PCA training
m = 0
# define data_preprocessing strategies
pre_processing = {"None": None, "zscore": zscore}  # None is RAW data

# define lambda range list
C_list = np.logspace(-5, 5, num=15)

make_dir("SVMPlot")

# Training and Validation LinearSVM
minDCF = []
for C in tqdm(C_list):
    scores = evaluation(DTR, LTR, DTE, SVM.SVM, p_T=p_T, C=C, k=1)
    minDCF.append(min_DCF(scores, LTE, p_T, Cfn, Cfp))

fig = plt.figure()
plt.plot(C_list, minDCF)
plt.xscale('log')
plt.xlim(C_list[0], C_list[-1])
plt.savefig(f'{absolute_path}/../Images/SVMPlot/SVMLinear_prior{p_T}.png')
plt.close(fig)

# # polynomial SVM
# minDCF = []
# for C in tqdm(C_list):
#     scores = evaluation(DTR, LTR, DTE, SVM.PolynomialSVM, p_T=p_T, C=C, k=1, d=2, c=1)
#     minDCF.append(min_DCF(scores, LTE, p_T, Cfn, Cfp))
#
# fig = plt.figure()
# plt.plot(C_list, minDCF)
# plt.xscale('log')
# plt.xlim(C_list[0], C_list[-1])
# plt.savefig(f'{absolute_path}/../Images/SVMPlot/SVMPolynomial_prior{p_T}.png')
# plt.close(fig)
#
#
# # Training and Validation RBFSVM
# minDCF = []
# for C in tqdm(C_list):
#     scores = evaluation(DTR, LTR, DTE, SVM.RBFSVM, p_T=p_T, C=C, k=1, gamma=1.0)
#     minDCF.append(min_DCF(scores, LTE, p_T, Cfn, Cfp))
#
# fig = plt.figure()
# plt.plot(C_list, minDCF)
# plt.xscale('log')
# plt.xlim(C_list[0], C_list[-1])
# plt.savefig(f'{absolute_path}/../Images/SVMPlot/RBFSVM_prior{p_T}.png')
# plt.close(fig)
#
