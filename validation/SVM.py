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

wpoints = [(0.5, 1, 1), (0.2, 1, 1), (0.8, 1, 1)]

# import training data
D, L = load_dataset("Train.txt")

# define PCA m values list
m_list = [0, 10, 11]  # example values - 0 mandatory for no PCA training

# define data_preprocessing strategies
pre_processing = {"None": None, "zscore": zscore}  # None is RAW data

# define lambda range list
C_list = np.logspace(-5, 5, num=10)

make_dir("SVMPlot")


def plot_SVMs(D, L, K, classifier, wpoints, m, p_T, k, title, **kwargs):
    minDCF_raw = [[], [], []]
    minDCF_zscore = [[], [], []]
    for C in tqdm(C_list):
        # RAW
        scores, labels = KFold_CV(D, L, K, classifier,
                                  wpoint=wpoints, pca_m=m, pre_process=None, p_T=p_T, C=C, k=k, **kwargs)
        np.save(f'{score_path}/{title}_m{0}_preNone_prior{p_T}_C{C}_k{k}',
                np.array([scores, labels]))
        for i, wpoint in enumerate(wpoints):
            minDCF_raw[i].append(min_DCF(scores, labels, wpoint[0], wpoint[1], wpoint[2]))
        # ZSCORE
        scores, labels = KFold_CV(D, L, K, classifier,
                                  wpoint=wpoints, pca_m=m, pre_process=zscore, p_T=p_T, C=C, k=k, **kwargs)
        np.save(f'{score_path}/{title}_m{0}_prezscore_prior{p_T}_C{C}_k{k}',
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
    plt.suptitle("title")
    plt.xscale('log')
    plt.xlim(C_list[0], C_list[-1])
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.legend()
    plt.savefig(f'{absolute_path}/../Images/SVMPlot/{title}_prior{p_T}.png')
    plt.close(fig)


# Training and Validation LinearSVM
plot_SVMs(D, L, K, SVM.SVM, wpoints, 0, 0.5, 1, "SVMLinear")

# polynomial SVM
plot_SVMs(D, L, K, SVM.PolynomialSVM, wpoints, 0, 0.5, 1, "SVMPolynomial", d=2, c=1)

# Training and Validation RBFSVM
plot_SVMs(D, L, K, SVM.RBFSVM, wpoints, 0, 0.5, 1, "RBFSVM", gamma=1.0)

