from Models import LogisticRegression
from utils import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from measures import *
from prettytable import PrettyTable, MARKDOWN


def LogisticRegressionEvaluation():
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

    wpoints = [(0.5, 1, 1), (0.2, 1, 1), (0.8, 1, 1)]

    # import training data
    DTR, LTR = load_dataset("Train.txt")
    DTE, LTE = load_dataset("Test.txt")
    DTR_ZS, DTE_ZS = zscore(DTR, DTE)

    # define data_preprocessing strategies
    pre_processing = {"None": None, "zscore": zscore}  # None is RAW data

    # define lambda range list
    lambdas = np.logspace(-4, 4, num=8)  # per la quadratic metti num=13/15 perchè ci mette più tempo e ti esplode il pc

    make_dir("LRPlot")

    minDCF_validation = [[], [], []]
    minDCF_evaluation = [[], [], []]
    for lmd in tqdm(lambdas):
        scores, labels = KFold_CV(DTR, LTR, K, LogisticRegression.LogisticRegression,
                                  wpoint=wpoints, pca_m=0, pre_process=None, lmd=lmd, prior=0.7)
        for i, (p_T, Cfn, Cfp) in enumerate(wpoints):
            minDCF_validation[i].append(min_DCF(scores, labels, p_T, Cfn, Cfp))
        clf = LogisticRegression.LogisticRegression(lmd=lmd, prior=0.7)
        clf.fit(DTR, LTR)
        scores_DTE = clf.transform(DTE)
        for i, (p_T, Cfn, Cfp) in enumerate(wpoints):
            minDCF_evaluation[i].append(min_DCF(scores_DTE, LTE, p_T, Cfn, Cfp))

    fig = plt.figure()
    plt.plot(lambdas, minDCF_validation[0], color="red", label="pi = 0.5 [Dev Set]")
    plt.plot(lambdas, minDCF_evaluation[0], color="red", linestyle='dashed', label="pi = 0.5 [Eval Set]")
    plt.plot(lambdas, minDCF_validation[1], color="blue", label="pi = 0.2 [Dev Set]")
    plt.plot(lambdas, minDCF_evaluation[1], color="blue", linestyle='dashed', label="pi = 0.2 [Eval Set]")
    plt.plot(lambdas, minDCF_validation[2], color="green", label="pi = 0.8 [Dev Set]")
    plt.plot(lambdas, minDCF_evaluation[2], color="green", linestyle='dashed', label="pi = 0.8 [Eval Set]")
    plt.xlabel("Lambda")
    plt.ylabel("minDCF")
    plt.suptitle("LogReg Eval Valid Comparison - zscore")
    plt.xscale('log')
    plt.xlim(lambdas[0], lambdas[-1])
    plt.legend()
    plt.savefig(f'{absolute_path}/../Images/LRPlot/LogRegEvalValidZSCORE.png')
    plt.close(fig)

    # Priors evaluation

    table = PrettyTable()
    table.set_style(MARKDOWN)
    table.field_names = ['Model', 'prior', 'pi = 0.5', 'pi = 0.2', "pi = 0.8", 'zs_pi = 0.5', 'zs_pi = 0.2',
                         "zs_pi = 0.8"]

    for prior in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        minDcf_vec = []
        clf = LogisticRegression.LogisticRegression(lmd=0.2, prior=prior)
        clf.fit(DTR, LTR)
        scores = clf.transform(DTE)
        for p_t, cfn, cfp in wpoints:
            minDcf_vec.append(min_DCF(scores, LTE, p_t, cfn, cfp))
        clf = LogisticRegression.LogisticRegression(lmd=1e-3, prior=prior)
        clf.fit(DTR_ZS, LTR)
        scores = clf.transform(DTE_ZS)
        for p_t, cfn, cfp in wpoints:
            minDcf_vec.append(min_DCF(scores, LTE, p_t, cfn, cfp))
        table.add_row(["LR", f'{prior}', *minDcf_vec])

    print(table.get_string())

    # QLogReg No zscore prior=0.7 lambda validation
    minDCF_validation = [[], [], []]
    minDCF_evaluation = [[], [], []]
    for lmd in tqdm(lambdas):
        scores, labels = KFold_CV(DTR, LTR, K, LogisticRegression.QuadraticLogisticRegression,
                                  wpoint=wpoints, pca_m=0, pre_process=None, lmd=lmd, prior=0.7)
        for i, (p_T, Cfn, Cfp) in enumerate(wpoints):
            minDCF_validation[i].append(min_DCF(scores, labels, p_T, Cfn, Cfp))
        clf = LogisticRegression.QuadraticLogisticRegression(lmd=lmd, prior=0.7)
        clf.fit(DTR, LTR)
        scores_DTE = clf.transform(DTE)
        for i, (p_T, Cfn, Cfp) in enumerate(wpoints):
            minDCF_evaluation[i].append(min_DCF(scores_DTE, LTE, p_T, Cfn, Cfp))

    fig = plt.figure()
    plt.plot(lambdas, minDCF_validation[0], color="red", label="pi = 0.5 [Dev Set]")
    plt.plot(lambdas, minDCF_evaluation[0], color="red", linestyle='dashed', label="pi = 0.5 [Eval Set]")
    plt.plot(lambdas, minDCF_validation[1], color="blue", label="pi = 0.2 [Dev Set]")
    plt.plot(lambdas, minDCF_evaluation[1], color="blue", linestyle='dashed', label="pi = 0.2 [Eval Set]")
    plt.plot(lambdas, minDCF_validation[2], color="green", label="pi = 0.8 [Dev Set]")
    plt.plot(lambdas, minDCF_evaluation[2], color="green", linestyle='dashed', label="pi = 0.8 [Eval Set]")
    plt.xlabel("Lambda")
    plt.ylabel("minDCF")
    plt.suptitle("QLogReg Eval Valid Comparison")
    plt.xscale('log')
    plt.xlim(lambdas[0], lambdas[-1])
    plt.legend()
    plt.savefig(f'{absolute_path}/../Images/LRPlot/QLogRegEvalValid.png')
    plt.close(fig)

    # Best Qlog-Reg PCA validation

    table = PrettyTable()
    table.set_style(MARKDOWN)
    table.field_names = ['Model', 'PCA(m)', 'pi = 0.5', 'pi = 0.2', "pi = 0.8", 'zs_pi = 0.5', 'zs_pi = 0.2',
                         "zs_pi = 0.8"]

    pca_reducer = PCA()
    for m in [0, 12, 11, 9, 8]:
        DCF = []
        # RAW
        DTR_, DTE_ = DTR, DTE
        if m != 0:
            pca_reducer.fit(DTR)
            DTR_ = pca_reducer.transform(m, DTR)
            DTE_ = pca_reducer.transform(m, DTE)
        clf = LogisticRegression.QuadraticLogisticRegression(lmd=0.02, prior=0.7)
        clf.fit(DTR_, LTR)
        scores_DTE = clf.transform(DTE_)
        DCF.append(min_DCF(scores_DTE, LTE, 0.5, 1, 1))
        DCF.append(min_DCF(scores_DTE, LTE, 0.2, 1, 1))
        DCF.append(min_DCF(scores_DTE, LTE, 0.8, 1, 1))
        # Z Score
        DTR_ZS_, DTE_ZS_ = DTR_ZS, DTE_ZS
        if m != 0:
            pca_reducer.fit(DTR_ZS)
            DTR_ZS_ = pca_reducer.transform(m, DTR_ZS)
            DTE_ZS_ = pca_reducer.transform(m, DTE_ZS)
        clf = LogisticRegression.QuadraticLogisticRegression(lmd=1e-3, prior=0.7)
        clf.fit(DTR_ZS_, LTR)
        scores_DTE = clf.transform(DTE_ZS_)
        DCF.append(min_DCF(scores_DTE, LTE, 0.5, 1, 1))
        DCF.append(min_DCF(scores_DTE, LTE, 0.2, 1, 1))
        DCF.append(min_DCF(scores_DTE, LTE, 0.8, 1, 1))
        table.add_row(['QLR', f'{m}', *DCF])

    print(table.get_string())
