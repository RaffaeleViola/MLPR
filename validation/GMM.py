from utils import *
from Models import GMMClassifier
from measures import *
from prettytable import PrettyTable, MARKDOWN

absolute_path = os.path.dirname(os.path.abspath(__file__))
score_path = f'{absolute_path}/../scores/train'
if not os.path.exists(score_path):
    os.mkdir(score_path)
make_dir("GMMPlot")
plot_path = f'{absolute_path}/../Images/GMMPlot'

class_names = {(False, False): "GMMStandard", (False, True): "GMMTied", (True, False): "GMMDiagonal"}


def train_GMMs(Gvalues, m, diag=False, tied=False):
    for i, G in enumerate(Gvalues):
        scores, labels = KFold_CV(D, L, 5, GMMClassifier.GMM, wpoint=wpoints, pca_m=m, pre_process=None,
                                  G=G, alpha=alpha, tresh=tresh, psi=psi, diag=diag, tied=tied)
        np.save(f'{score_path}/{class_names[(diag, tied)]}_G{G}_m{m}_preNone_a{alpha}_t{tresh}_psi{psi}',
                np.array([scores, labels]))
        scores, labels = KFold_CV(D, L, 5, GMMClassifier.GMM, wpoint=wpoints, pca_m=m, pre_process=zscore,
                                  G=G, alpha=alpha, tresh=tresh, psi=psi, diag=diag, tied=tied)
        np.save(f'{score_path}/{class_names[(diag, tied)]}_G{G}_m{m}_prezscore_a{alpha}_t{tresh}_psi{psi}',
                np.array([scores, labels]))


def bar_plots(G_list, m, diag=False, tied=False):
    dcf_map_raw = [0] * len(G_list)
    dcf_map_zscore = [0] * len(G_list)
    for i, G in tqdm(enumerate(G_list)):
        data = np.load(f'{score_path}/{class_names[(diag, tied)]}_G{G}_m{m}_preNone_a{alpha}_t{tresh}_psi{psi}.npy')
        dcf_map_raw[i] = min_DCF(data[0], data[1], 0.5, 1, 1)
        data = np.load(f'{score_path}/{class_names[(diag, tied)]}_G{G}_m{m}_prezscore_a{alpha}_t{tresh}_psi{psi}.npy')
        dcf_map_zscore[i] = min_DCF(data[0], data[1], 0.5, 1, 1)
    fig = plt.figure()
    plt.bar(np.array(range(len(G_list))) - 0.1, np.array(dcf_map_raw), width=0.2)
    plt.bar(np.array(range(len(G_list))) + 0.1, np.array(dcf_map_zscore), width=0.2)
    plt.xticks(np.array(range(len(G_list))), np.array(G_list))
    plt.legend(["Raw", "Zscore"])
    plt.xlabel("G")
    plt.ylabel("minDCF")
    plt.suptitle(f'{class_names[(diag, tied)]}')
    plt.savefig(f'{plot_path}/{class_names[(diag, tied)]}_m{m}_preNone_a{alpha}_t{tresh}_psi{psi}.png')
    plt.close(fig)


def validatePCA(D, L, table, wpoints, pre_process, G, m, alpha, tresh, psi, diag, tied):
    minDCF = []
    scores, labels = KFold_CV(D, L, 5, GMMClassifier.GMM, wpoint=wpoints, pca_m=m, pre_process=pre_process,
                              G=G, alpha=alpha, tresh=tresh, psi=psi, diag=diag, tied=tied)
    np.save(f'{score_path}/{class_names[(diag, tied)]}_G{G[0]}{G[1]}_m{m}_preNone_a{alpha}_t{tresh}_psi{psi}',
            np.array([scores, labels]))
    for wpoint in wpoints:
        minDCF.append(min_DCF(scores, labels, wpoint[0], wpoint[1], wpoint[2]))
    table.add_row([f'{class_names[(diag, tied)]}', f'{G[0]}', f'{G[1]}', f'PCA({m})', *minDCF])


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
m_list = [0, 12, 11, 10, 9]  # example values - 0 mandatory for no PCA training
# define data_preprocessing strategies
pre_processing = {"None": None, "zscore": zscore}  # None is RAW data
# define G list
G_list = [1, 2, 4, 8, 16, 32]
# you can try other parameters setting alpha, tresh, psi prova ad esplorare
alpha, tresh, psi = 0.1, 1e-6, 0.01

# GMMstandard
# train_GMMs(G_list, 0, False, False)
# bar_plots(G_list, 0, False, False)

# GMMDiag
# train_GMMs(G_list, 0, True, False)
# bar_plots(G_list, 0, True, False)

# GMMTied
# train_GMMs(G_list, 0, False, True)
# bar_plots(G_list, 0, False, True)

# GMMDiag PCA(12)
train_GMMs(G_list, 12, True, False)
bar_plots(G_list, 12, True, False)


# table = PrettyTable()
# table.set_style(MARKDOWN)
# table.field_names = ['Model', 'K_class0', 'K_class1', 'pi = 0.5', 'pi = 0.2', "pi = 0.8", 'zs_pi = 0.5',
#                      'zs_pi = 0.2',
#                      "zs_pi = 0.8"]

# K combinations
# for g0 in [2, 4]:
#     for g1 in [2, 4]:
# minDCF = []
# for name_pre, pre_process in [("None", None), ("zscore", zscore)]:
#     if os.path.exists(f'{score_path}/{class_names[(False, False)]}_G{g0}{g1}_m{0}_pre{name_pre}_a{alpha}_t{tresh}_psi{psi}.npz'):
#         continue
#     scores, labels = KFold_CV(D, L, 5, GMMClassifier.GMM, wpoint=wpoints, pca_m=0, pre_process=pre_process,
#                               G=[g0, g1], alpha=alpha, tresh=tresh, psi=psi, diag=False, tied=False)
#     np.save(f'{score_path}/{class_names[(False, False)]}_G{g0}{g1}_m{0}_pre{name_pre}_a{alpha}_t{tresh}_psi{psi}',
#             np.array([scores, labels]))
#     for wpoint in wpoints:
#         minDCF.append(min_DCF(scores, labels, wpoint[0], wpoint[1], wpoint[2]))
# table.add_row(["GMMStandard", f'{g0}', f'{g1}', *minDCF])


# K combinations
# for g0 in [4, 8]:
#     for g1 in [4, 8]:
# minDCF = []
# for name_pre, pre_process in [("None", None), ("zscore", zscore)]:
#     if os.path.exists(f'{score_path}/{class_names[(False, True)]}_G{g0}{g1}_m{0}_pre{name_pre}_a{alpha}_t{tresh}_psi{psi}.npy'):
#         continue
#     scores, labels = KFold_CV(D, L, 5, GMMClassifier.GMM, wpoint=wpoints, pca_m=0, pre_process=pre_process,
#                               G=[g0, g1], alpha=alpha, tresh=tresh, psi=psi, diag=False, tied=True)
#     np.save(f'{score_path}/{class_names[(False, True)]}_G{g0}{g1}_m{0}_pre{name_pre}_a{alpha}_t{tresh}_psi{psi}',
#             np.array([scores, labels]))
# for wpoint in wpoints:
#     minDCF.append(min_DCF(scores, labels, wpoint[0], wpoint[1], wpoint[2]))
# table.add_row(["GMMTied", f'{g0}', f'{g1}', *minDCF])

# print(table.get_string())


# # PCA validation
# table = PrettyTable()
# table.set_style(MARKDOWN)
# table.field_names = ['Model', 'K_class0', 'K_class1', 'PCA(m)', 'pi = 0.5', 'pi = 0.2', "pi = 0.8"]
#
# for m in [12, 11, 10, 9, 8]:
#     # GMM Standard K = 4
#     validatePCA(D, L, table, wpoints, None, [4, 4], m, alpha, tresh, psi, False, False)
#     # GMM Tied K0 = 8 K1 = 4
#     validatePCA(D, L, table, wpoints, None, [8, 4], m, alpha, tresh, psi, False, True)
#     # GMM Diag K = 4
#     validatePCA(D, L, table, wpoints, None, [4, 4], m, alpha, tresh, psi, True, False)
#
# print(table.get_string())
