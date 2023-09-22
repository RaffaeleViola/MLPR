from measures import *
from prettytable import PrettyTable, MARKDOWN

absolute_path = os.path.dirname(os.path.abspath(__file__))
score_path = f'{absolute_path}/../scores/train'
if not os.path.exists(score_path):
    os.mkdir(score_path)


def plot_LR(D, L, K, classifier, lambda_vec, wpoints, name, m, pre_process):
    minDCF = [[[], []], [[], []], [[], []]]
    for lmd in tqdm(lambda_vec):
        for i, (p_T, Cfn, Cfp) in enumerate(wpoints):
            for j, pre in enumerate(pre_process):
                scores, labels = KFold_CV(D, L, K, classifier,
                                          wpoint=wpoints, pca_m=m, pre_process=pre, lmd=lmd, prior=p_T)
                minDCF[i][j].append(min_DCF(scores, labels, p_T, Cfn, Cfp))
    fig = plt.figure()
    plt.plot(lambda_vec, minDCF[0][0], color="red", label="pi = 0.5")
    plt.plot(lambda_vec, minDCF[0][1], color="red", linestyle='dashed', label="pi = 0.5 - ZSCORE")
    plt.plot(lambda_vec, minDCF[1][0], color="blue", label="pi = 0.2 ")
    plt.plot(lambda_vec, minDCF[1][1], color="blue", linestyle='dashed', label="pi = 0.2 - ZSCORE")
    plt.plot(lambda_vec, minDCF[2][0], color="green", label="pi = 0.8")
    plt.plot(lambda_vec, minDCF[2][1], color="green", linestyle='dashed', label="pi = 0.8 - ZSCORE")
    plt.xlabel("Lambda")
    plt.ylabel("minDCF")
    sup = "RAW" if pre_process is None else "ZSCORE"
    plt.suptitle(sup)
    plt.xscale('log')
    plt.xlim(lambda_vec[0], lambda_vec[-1])
    plt.legend()
    plt.savefig(f'{absolute_path}/../Images/LRPlot/{name}.png')
    plt.close(fig)


def LRValidation():
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
    m_list = [0, 11, 9]  # example values - 0 mandatory for no PCA training
    # define data_preprocessing strategies
    pre_processing = {"None": None, "zscore": zscore}  # None is RAW data
    # define lambda range list
    lambdas = np.logspace(-5, 5, num=10)  # per la quadratic metti num=13/15 perchè ci mette più tempo e ti esplode il pc

    make_dir("LRPlot")
    # Training and Validation LR
    for m in m_list:
        plot_LR(D, L, K, LogisticRegression.LogisticRegression, lambdas, wpoints, f"LR_PCA({m})", m, [None, zscore])

    lambda_linear = 1e-3
    table = PrettyTable()
    table.set_style(MARKDOWN)
    table.field_names = ['Model', 'prior', 'pi = 0.5', 'pi = 0.2', "pi = 0.8", 'zs_pi = 0.5', 'zs_pi = 0.2', "zs_pi = 0.8"]

    for prior in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        minDcf_vec = []
        scores, labels = KFold_CV(D, L, K, LogisticRegression.LogisticRegression,
                                  wpoint=wpoints, pca_m=0, pre_process=None, lmd=lambda_linear, prior=prior)
        for p_t, cfn, cfp in wpoints:
            minDcf_vec.append(min_DCF(scores, labels, p_t, cfn, cfp))
        scores, labels = KFold_CV(D, L, K, LogisticRegression.LogisticRegression,
                                  wpoint=wpoints, pca_m=0, pre_process=zscore, lmd=lambda_linear, prior=prior)
        for p_t, cfn, cfp in wpoints:
            minDcf_vec.append(min_DCF(scores, labels, p_t, cfn, cfp))
        table.add_row(["LR", f'{prior}', *minDcf_vec])

    print(table.get_string())

    # Training and Validation QLR
    for m in [0]:
        plot_LR(D, L, K, LogisticRegression.QuadraticLogisticRegression, lambdas, wpoints, f"QLR_PCA({m})", m, [None, zscore])

    minDCF = [[], [], [], [], [], []]
    for lmd in lambdas:
        for i, m in enumerate([0, 11, 10, 9, 8, 7]):
            scores, labels = KFold_CV(D, L, K, LogisticRegression.QuadraticLogisticRegression,
                                      wpoint=wpoints, pca_m=m, pre_process=None, lmd=lmd, prior=0.5)
            minDCF[i].append(min_DCF(scores, labels, 0.5, 1, 1))

    fig = plt.figure()
    for i, m in enumerate([0, 11, 10, 9, 8, 7]):
        label = f'PCA({m})' if m != 0 else "NO PCA"
        plt.plot(lambdas, minDCF[i], label=f"pi = 0.5 {label}")
    plt.xlabel("Lambda")
    plt.ylabel("minDCF")
    sup = "RAW PCA"
    plt.suptitle(sup)
    plt.xscale('log')
    plt.xlim(lambdas[0], lambdas[-1])
    plt.legend()
    plt.savefig(f'{absolute_path}/../Images/LRPlot/QLR_PCA.png')
    plt.close(fig)

    table = PrettyTable()
    table.set_style(MARKDOWN)
    table.field_names = ['Model', 'prior', 'pi = 0.5', 'pi = 0.2', "pi = 0.8", 'zs_pi = 0.5', 'zs_pi = 0.2', "zs_pi = 0.8"]

    for prior in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        minDcf_vec = []
        scores, labels = KFold_CV(D, L, K, LogisticRegression.QuadraticLogisticRegression,
                                  wpoint=wpoints, pca_m=0, pre_process=None, lmd=100, prior=prior)
        for p_t, cfn, cfp in wpoints:
            minDcf_vec.append(min_DCF(scores, labels, p_t, cfn, cfp))
        scores, labels = KFold_CV(D, L, K, LogisticRegression.QuadraticLogisticRegression,
                                  wpoint=wpoints, pca_m=0, pre_process=zscore, lmd=1e-3, prior=prior)
        for p_t, cfn, cfp in wpoints:
            minDcf_vec.append(min_DCF(scores, labels, p_t, cfn, cfp))
        table.add_row(["QLR", f'{prior}', *minDcf_vec])

    print(table.get_string())

    # Save best model
    scores, labels = KFold_CV(D, L, K, LogisticRegression.LogisticRegression,
                              wpoint=wpoints, pca_m=0, pre_process=None, lmd=1e-3, prior=0.7)
    np.save(f'{score_path}/LR_m{0}_preNone_prior{0.7}_lmd{1e-3}',
            np.array([scores, labels]))
