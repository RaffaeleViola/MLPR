from Models import SVM
from measures import *
from prettytable import PrettyTable, MARKDOWN


def SVmEvaluation():
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    score_path = f'{absolute_path}/../scores/test'
    if not os.path.exists(score_path):
        os.mkdir(score_path)

    # define K for K-FOld Cross Validation
    K = 5
    wpoints = [(0.5, 1, 1), (0.2, 1, 1), (0.8, 1, 1)]

    # import training data
    DTR, LTR = load_dataset("Train.txt")
    DTE, LTE = load_dataset("Test.txt")
    DTR_ZS, DTE_ZS = zscore(DTR, DTE)
    # define PCA m values list
    m = 0
    # define lambda range list
    C_list = np.logspace(-5, 5, num=10)
    make_dir("SVMPlot")

    # LinearSVM prior=0.8 C evaluation zscore
    minDCF_validation = [[], [], []]
    minDCF_evaluation = [[], [], []]
    for C in tqdm(C_list):
        # RAW
        scores, labels = KFold_CV(DTR, LTR, K, SVM.SVM,
                                  wpoint=wpoints, pca_m=0, pre_process=zscore, p_T=0.8, C=C, k=1)
        for i, wpoint in enumerate(wpoints):
            minDCF_validation[i].append(min_DCF(scores, labels, wpoint[0], wpoint[1], wpoint[2]))
        # ZSCORE
        clf = SVM.SVM(0.8, C, 1)
        clf.fit(DTR_ZS, LTR)
        scores = clf.transform(DTE_ZS)
        for i, wpoint in enumerate(wpoints):
            minDCF_evaluation[i].append(min_DCF(scores, LTE, wpoint[0], wpoint[1], wpoint[2]))

    fig = plt.figure()
    plt.plot(C_list, minDCF_validation[0], color="red", linestyle='dashed', label="pi = 0.5 [Dev Set]")
    plt.plot(C_list, minDCF_evaluation[0], color="red", label="pi = 0.5 [Eval Set]")
    plt.plot(C_list, minDCF_validation[1], color="blue", linestyle='dashed', label="pi = 0.2 [Dev Set]")
    plt.plot(C_list, minDCF_evaluation[1], color="blue", label="pi = 0.2 [Eval Set]")
    plt.plot(C_list, minDCF_validation[2], color="green", linestyle='dashed', label="pi = 0.8 [Dev Set]")
    plt.plot(C_list, minDCF_evaluation[2], color="green", label="pi = 0.8 [Eval Set]")
    plt.suptitle("SVMLinear Zscore")
    plt.xscale('log')
    plt.xlim(C_list[0], C_list[-1])
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.legend()
    plt.savefig(f'{absolute_path}/../Images/SVMPlot/SVMLinearZscoreEvalValid.png')
    plt.close(fig)

    # LinearSVM prior=0.8 C evaluation No zscore
    minDCF_validation = [[], [], []]
    minDCF_evaluation = [[], [], []]
    for C in tqdm(C_list):
        # RAW
        scores, labels = KFold_CV(DTR, LTR, K, SVM.SVM,
                                  wpoint=wpoints, pca_m=0, pre_process=None, p_T=0.8, C=C, k=1)
        for i, wpoint in enumerate(wpoints):
            minDCF_validation[i].append(min_DCF(scores, labels, wpoint[0], wpoint[1], wpoint[2]))
        # ZSCORE
        clf = SVM.SVM(0.8, C, 1)
        clf.fit(DTR, LTR)
        scores = clf.transform(DTE)
        for i, wpoint in enumerate(wpoints):
            minDCF_evaluation[i].append(min_DCF(scores, LTE, wpoint[0], wpoint[1], wpoint[2]))

    fig = plt.figure()
    plt.plot(C_list, minDCF_validation[0], color="red", linestyle='dashed', label="pi = 0.5 [Dev Set]")
    plt.plot(C_list, minDCF_evaluation[0], color="red", label="pi = 0.5 [Eval Set]")
    plt.plot(C_list, minDCF_validation[1], color="blue", linestyle='dashed', label="pi = 0.2 [Dev Set]")
    plt.plot(C_list, minDCF_evaluation[1], color="blue", label="pi = 0.2 [Eval Set]")
    plt.plot(C_list, minDCF_validation[2], color="green", linestyle='dashed', label="pi = 0.8 [Dev Set]")
    plt.plot(C_list, minDCF_evaluation[2], color="green", label="pi = 0.8 [Eval Set]")
    plt.suptitle("SVMLinear RAW")
    plt.xscale('log')
    plt.xlim(C_list[0], C_list[-1])
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.legend()
    plt.savefig(f'{absolute_path}/../Images/SVMPlot/SVMLinearRawEvalValid.png')
    plt.close(fig)

    C_list = np.logspace(-4, 4, num=6)
    gamma_list = [0.1, 0.01, 0.001, 0.0001]
    minDCF_validation = [[] for _ in range(len(gamma_list))]
    minDCF_evaluation = [[] for _ in range(len(gamma_list))]
    for i, gamma in enumerate(gamma_list):
        for C in tqdm(C_list):
            scores, labels = KFold_CV(DTR, LTR, K, SVM.RBFSVM,
                                      wpoint=wpoints, pca_m=m, pre_process=None, p_T=0.7, C=C, k=1, gamma=gamma)
            minDCF_validation[i].append(min_DCF(scores, labels, 0.5, 1, 1))
            clf = SVM.RBFSVM(0.7, C, 1, gamma)
            clf.fit(DTR, LTR)
            scores = clf.transform(DTE)
            minDCF_evaluation[i].append(min_DCF(scores, LTE, 0.5, 1, 1))

    colors = ["blue", "green", "red", "cyan", "magenta", "yellow", "black", "white"]
    fig = plt.figure()
    for i in range(len(gamma_list)):
        plt.plot(C_list, minDCF_evaluation[i], color=colors[i], label=f"gamma = {gamma_list[i]} [Eval]")
        plt.plot(C_list, minDCF_validation[i], linestyle='dashed', color=colors[i],
                 label=f"gamma = {gamma_list[i]} - [Dev]")
    plt.suptitle("RBFSVMEvalValid")
    plt.xscale('log')
    plt.xlim(C_list[0], C_list[-1])
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.legend()
    plt.savefig(f'{absolute_path}/../Images/SVMPlot/RBFSVMEvalValid.png')
    plt.close(fig)

    # ZScore Version
    C_list = np.logspace(0, 2, num=5)
    gamma_list = [0.1, 0.01]
    minDCF_validation = [[] for _ in range(len(gamma_list))]
    minDCF_evaluation = [[] for _ in range(len(gamma_list))]
    for i, gamma in enumerate(gamma_list):
        for C in tqdm(C_list):
            scores, labels = KFold_CV(DTR, LTR, K, SVM.RBFSVM,
                                      wpoint=wpoints, pca_m=m, pre_process=zscore, p_T=0.7, C=C, k=1, gamma=gamma)
            minDCF_validation[i].append(min_DCF(scores, labels, 0.5, 1, 1))
            clf = SVM.RBFSVM(0.7, C, 1, gamma)
            clf.fit(DTR_ZS, LTR)
            scores = clf.transform(DTE_ZS)
            minDCF_evaluation[i].append(min_DCF(scores, LTE, 0.5, 1, 1))

    colors = ["blue", "green", "red", "cyan", "magenta", "yellow", "black", "white"]
    fig = plt.figure()
    for i in range(len(gamma_list)):
        plt.plot(C_list, minDCF_evaluation[i], color=colors[i], label=f"gamma = {gamma_list[i]} [Eval]")
        plt.plot(C_list, minDCF_validation[i], linestyle='dashed', color=colors[i],
                 label=f"gamma = {gamma_list[i]} - [Dev]")
    plt.suptitle("RBFSVMEvalValid ZScore")
    plt.xscale('log')
    plt.xlim(C_list[0], C_list[-1])
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.legend()
    plt.savefig(f'{absolute_path}/../Images/SVMPlot/RBFSVMEvalValidZscore.png')
    plt.close(fig)

    # Table for RBFSVM evaluating prior effect
    table = PrettyTable()
    table.set_style(MARKDOWN)
    table.field_names = ['Model', 'prior', 'pi = 0.5', 'pi = 0.2', "pi = 0.8", 'zs_pi = 0.5', 'zs_pi = 0.2',
                         "zs_pi = 0.8"]

    for prior in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        minDcf_vec = []
        clf = SVM.RBFSVM(prior, 10, 1, 0.001)
        clf.fit(DTR, LTR)
        scores = clf.transform(DTE)
        for p_t, cfn, cfp in wpoints:
            minDcf_vec.append(min_DCF(scores, LTE, p_t, cfn, cfp))
        clf = SVM.RBFSVM(prior, 3, 1, 0.1)
        clf.fit(DTR_ZS, LTR)
        scores = clf.transform(DTE_ZS)
        for p_t, cfn, cfp in wpoints:
            minDcf_vec.append(min_DCF(scores, LTE, p_t, cfn, cfp))
        table.add_row(["RBFSVM", f'{prior}', *minDcf_vec])

    print(table.get_string())

    # Table for RBFSVM evaluating PCA effect given prior, c, gamma e PreProcess
    table = PrettyTable()
    table.set_style(MARKDOWN)
    table.field_names = ['Model', 'PCA(m)', 'pi = 0.5', 'pi = 0.2', "pi = 0.8"]

    pca_reducer = PCA()
    for m in [0, 11, 9, 8]:
        minDcf_vec = []
        DTR_, DTE_ = DTR, DTE
        if m != 0:
            pca_reducer.fit(DTR_)
            DTR_ = pca_reducer.transform(m, DTR_)
            DTE_ = pca_reducer.transform(m, DTE_)
        clf = SVM.RBFSVM(0.5, 10, 1, 0.001)
        clf.fit(DTR_, LTR)
        scores = clf.transform(DTE_)
        for p_t, cfn, cfp in wpoints:
            minDcf_vec.append(min_DCF(scores, LTE, p_t, cfn, cfp))
        tag = f'PCA({m})' if m != 0 else "NO PCA"
        table.add_row(["RBFSVM", f'{tag}', *minDcf_vec])

    print(table.get_string())
