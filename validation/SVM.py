from Models import SVM
from measures import *
from prettytable import PrettyTable, MARKDOWN

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

wpoints = [(0.5, 1, 1), (0.2, 1, 1), (0.8, 1, 1)]

# import training data
D, L = load_dataset("Train.txt")

# define PCA m values list
m_list = [0, 10, 11]  # example values - 0 mandatory for no PCA training

# define data_preprocessing strategies
pre_processing = {"None": None, "zscore": zscore}  # None is RAW data

# define lambda range list
C_list = np.logspace(-5, -1, num=10)

make_dir("SVMPlot")


def plot_SVMs(D, L, K, classifier, wpoints, m, p_T, k, title, **kwargs):
    minDCF_raw = [[], [], []]
    minDCF_zscore = [[], [], []]
    for C in tqdm(C_list):
        # RAW
        scores, labels = KFold_CV(D, L, K, classifier,
                                  wpoint=wpoints, pca_m=m, pre_process=None, p_T=p_T, C=C, k=k, **kwargs)
        np.save(f'{score_path}/{title}_m{m}_preNone_prior{p_T}_C{C}_k{k}',
                np.array([scores, labels]))
        for i, wpoint in enumerate(wpoints):
            minDCF_raw[i].append(min_DCF(scores, labels, wpoint[0], wpoint[1], wpoint[2]))
        # ZSCORE
        scores, labels = KFold_CV(D, L, K, classifier,
                                  wpoint=wpoints, pca_m=m, pre_process=zscore, p_T=p_T, C=C, k=k, **kwargs)
        np.save(f'{score_path}/{title}_m{m}_prezscore_prior{p_T}_C{C}_k{k}',
                np.array([scores, labels]))
        for i, wpoint in enumerate(wpoints):
            minDCF_zscore[i].append(min_DCF(scores, labels, wpoint[0], wpoint[1], wpoint[2]))

    fig = plt.figure()
    plt.plot(C_list, minDCF_raw[0], color="red", label="p_T = 0.5")
    plt.plot(C_list, minDCF_zscore[0], linestyle='dashed', color="red", label="zscore p_T = 0.5")
    plt.plot(C_list, minDCF_raw[1], color="blue", label="p_T = 0.2")
    plt.plot(C_list, minDCF_zscore[1], linestyle='dashed', color="blue", label="zscore p_T = 0.2")
    plt.plot(C_list, minDCF_raw[2], color="green", label="p_T = 0.8")
    plt.plot(C_list, minDCF_zscore[2], linestyle='dashed', color="green", label="zscore p_T = 0.8")
    plt.suptitle(title)
    plt.xscale('log')
    plt.xlim(C_list[0], C_list[-1])
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.legend()
    plt.savefig(f'{absolute_path}/../Images/SVMPlot/{title}_prior{p_T}.png')
    plt.close(fig)


def plot_RBFs(D, L, K, C_list, gamma_list, wpoints, m, p_T, k, title):
    minDCF_raw = [[] for _ in range(len(gamma_list))]
    minDCF_zscore = [[] for _ in range(len(gamma_list))]
    for i, gamma in enumerate(gamma_list):
        for C in tqdm(C_list):
            # RAW
            scores, labels = KFold_CV(D, L, K, SVM.RBFSVM,
                                      wpoint=wpoints, pca_m=m, pre_process=None, p_T=p_T, C=C, k=k, gamma=gamma)
            np.save(f'{score_path}/{title}_m{m}_preNone_prior{p_T}_C{C}_k{k}_gamma{gamma}',
                    np.array([scores, labels]))
            minDCF_raw[i].append(min_DCF(scores, labels, 0.5, 1, 1))
            # ZSCORE
            scores, labels = KFold_CV(D, L, K, SVM.RBFSVM,
                                      wpoint=wpoints, pca_m=m, pre_process=zscore, p_T=p_T, C=C, k=k, gamma=gamma)
            np.save(f'{score_path}/{title}_m{m}_prezscore_prior{p_T}_C{C}_k{k}_gamma{gamma}',
                    np.array([scores, labels]))
            minDCF_zscore[i].append(min_DCF(scores, labels, 0.5, 1, 1))
    print(minDCF_raw)
    colors = ["blue", "green", "red", "cyan", "magenta", "yellow", "black", "white"]
    fig = plt.figure()
    for i in range(len(gamma_list)):
        plt.plot(C_list, minDCF_raw[i], color=colors[i], label=f"gamma = {gamma_list[i]}")
        plt.plot(C_list, minDCF_zscore[i], linestyle='dashed', color=colors[i],
                 label=f"gamma = {gamma_list[i]} - zscore")
    plt.suptitle(title)
    plt.xscale('log')
    plt.xlim(C_list[0], C_list[-1])
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.legend()
    plt.savefig(f'{absolute_path}/../Images/SVMPlot/{title}_prior{p_T}.png')
    plt.close(fig)


# # Training and Validation LinearSVM
# plot_SVMs(D, L, K, SVM.SVM, wpoints, 0, 0.5, 1, "SVMLinear")

# polynomial SVM
# plot_SVMs(D, L, K, SVM.PolynomialSVM, wpoints, 0, 0.5, 1, "SVMPolynomial_Degree2", d=2, c=1)
# plot_SVMs(D, L, K, SVM.PolynomialSVM, wpoints, 0, 0.5, 1, "SVMPolynomial_Degree3", d=3, c=1)

# Training and Validation RBFSVM
# plot_RBFs(D, L, K, np.logspace(-4, 4, num=6), [0.1, 0.01, 0.001, 0.0001], wpoints, 0, 0.5, 1, "RBFSVM")

# # Table for SVM Linear
# table = PrettyTable()
# table.set_style(MARKDOWN)
# table.field_names = ['Model', 'prior', 'pi = 0.5', 'pi = 0.2', "pi = 0.8", 'zs_pi = 0.5', 'zs_pi = 0.2', "zs_pi = 0.8"]
#
# for prior in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
#     minDcf_vec = []
#     scores, labels = KFold_CV(D, L, K, SVM.SVM,
#                               wpoint=wpoints, pca_m=0, pre_process=None, p_T=prior, C=0.1, k=1)
#     for p_t, cfn, cfp in wpoints:
#         minDcf_vec.append(min_DCF(scores, labels, p_t, cfn, cfp))
#     scores, labels = KFold_CV(D, L, K, SVM.SVM,
#                               wpoint=wpoints, pca_m=0, pre_process=zscore, p_T=prior, C=10, k=1)
#     for p_t, cfn, cfp in wpoints:
#         minDcf_vec.append(min_DCF(scores, labels, p_t, cfn, cfp))
#     table.add_row(["SVMLinear", f'{prior}', *minDcf_vec])
#
# print(table.get_string())
#
# # Table for PolySVM
# table = PrettyTable()
# table.set_style(MARKDOWN)
# table.field_names = ['Model', 'prior', 'pi = 0.5', 'pi = 0.2', "pi = 0.8", 'zs_pi = 0.5', 'zs_pi = 0.2', "zs_pi = 0.8"]
#
# for prior in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
#     minDcf_vec = []
#     scores, labels = KFold_CV(D, L, K, SVM.PolynomialSVM,
#                               wpoint=wpoints, pca_m=0, pre_process=None, p_T=prior, C=1e-4, k=1, d=2, c=1)
#     for p_t, cfn, cfp in wpoints:
#         minDcf_vec.append(min_DCF(scores, labels, p_t, cfn, cfp))
#     scores, labels = KFold_CV(D, L, K, SVM.PolynomialSVM,
#                               wpoint=wpoints, pca_m=0, pre_process=zscore,  p_T=prior, C=0.1, k=1, d=2, c=1)
#     for p_t, cfn, cfp in wpoints:
#         minDcf_vec.append(min_DCF(scores, labels, p_t, cfn, cfp))
#     table.add_row(["SVMPoly_d=2", f'{prior}', *minDcf_vec])
#
# print(table.get_string())

# # Table for RBFSVM
# table = PrettyTable()
# table.set_style(MARKDOWN)
# table.field_names = ['Model', 'prior', 'pi = 0.5', 'pi = 0.2', "pi = 0.8", 'zs_pi = 0.5', 'zs_pi = 0.2', "zs_pi = 0.8"]
#
# for prior in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
#     minDcf_vec = []
#     scores, labels = KFold_CV(D, L, K, SVM.RBFSVM,
#                               wpoint=wpoints, pca_m=0, pre_process=None, p_T=prior, C=10, k=1, gamma=0.001)
#     for p_t, cfn, cfp in wpoints:
#         minDcf_vec.append(min_DCF(scores, labels, p_t, cfn, cfp))
#     scores, labels = KFold_CV(D, L, K, SVM.RBFSVM,
#                               wpoint=wpoints, pca_m=0, pre_process=zscore,  p_T=prior, C=10, k=1, gamma=0.1)
#     for p_t, cfn, cfp in wpoints:
#         minDcf_vec.append(min_DCF(scores, labels, p_t, cfn, cfp))
#     table.add_row(["RBFSVM", f'{prior}', *minDcf_vec])
#
# print(table.get_string())

# # Table for RBFSVM p_T=0.7 best model PCA validation
# table = PrettyTable()
# table.set_style(MARKDOWN)
# table.field_names = ['Model', 'prior', 'PCA(m)', 'pi = 0.5', 'pi = 0.2', "pi = 0.8"]
#
# for m in [0, 12, 11, 9]:
#     minDcf_vec = []
#     scores, labels = KFold_CV(D, L, K, SVM.RBFSVM,
#                               wpoint=wpoints, pca_m=m, pre_process=None, p_T=0.7, C=10, k=1, gamma=0.001)
#     for p_t, cfn, cfp in wpoints:
#         minDcf_vec.append(min_DCF(scores, labels, p_t, cfn, cfp))
#     if m == 0:
#         m = '-'
#     table.add_row(["RBFSVM prior = 0.7 gamma=0.001 C=10", f'{0.7}', f'PCA({m})', *minDcf_vec])
#
# print(table.get_string())

# Save best model
scores, labels = KFold_CV(D, L, K, SVM.RBFSVM,
                          wpoint=wpoints, pca_m=0, pre_process=None, p_T=0.7, C=10, k=1, gamma=0.001)

np.save(f'{score_path}/RBFSVM_m{0}_preNone_prior{0.7}_C{10}_k{1}_gamma{0.001}',
        np.array([scores, labels]))
