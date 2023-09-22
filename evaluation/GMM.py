from Models import GMMClassifier
from measures import *
from prettytable import PrettyTable, MARKDOWN


def add_row(clf, DTR, LTR, DTE, LTE, DTR_ZS, DTE_ZS, wpoints, table, name, k0, k1):
    minDCF = []
    clf.fit(DTR, LTR)
    scores = clf.transform(DTE)
    for p_T, Cfn, Cfp in wpoints:
        minDCF.append(min_DCF(scores, LTE, p_T, Cfn, Cfp))
    clf.fit(DTR_ZS, LTR)
    scores = clf.transform(DTE_ZS)
    for p_T, Cfn, Cfp in wpoints:
        minDCF.append(min_DCF(scores, LTE, p_T, Cfn, Cfp))
    table.add_row([name, k0, k1, *minDCF])


def GMMEvaluation():
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    score_path = f'{absolute_path}/../scores/test'
    if not os.path.exists(score_path):
        os.mkdir(score_path)
    plot_path = f'{absolute_path}/../Images/GMMPlot'


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

    # define G list
    G_list = [1, 2, 4, 8, 16, 32]

    # you can try other parameters setting alpha, tresh, psi
    alpha, tresh, psi = 0.1, 1e-6, 0.01

    # Plot for GMMStandard
    dcf_map_raw = [0] * len(G_list)
    dcf_map_zscore = [0] * len(G_list)
    for i, G in tqdm(enumerate(G_list)):
        clf = GMMClassifier.GMM(G, alpha, tresh, psi, False, False)
        clf.fit(DTR, LTR)
        scores = clf.transform(DTE)
        dcf_map_raw[i] = min_DCF(scores, LTE, 0.5, 1, 1)
        clf = GMMClassifier.GMM(G, alpha, tresh, psi, False, False)
        clf.fit(DTR_ZS, LTR)
        scores = clf.transform(DTE_ZS)
        dcf_map_zscore[i] = min_DCF(scores, LTE, 0.5, 1, 1)
    fig = plt.figure()
    plt.bar(np.array(range(len(G_list))) - 0.1, np.array(dcf_map_raw), width=0.2)
    plt.bar(np.array(range(len(G_list))) + 0.1, np.array(dcf_map_zscore), width=0.2)
    plt.xticks(np.array(range(len(G_list))), np.array(G_list))
    plt.legend(["Raw", "Zscore"])
    plt.xlabel("G")
    plt.ylabel("minDCF")
    plt.suptitle(f'GMMStandard')
    plt.savefig(f'{plot_path}/GMMStandardEvaluation.png')
    plt.close(fig)

    # Plot for GMMTied
    dcf_map_raw = [0] * len(G_list)
    dcf_map_zscore = [0] * len(G_list)
    for i, G in tqdm(enumerate(G_list)):
        clf = GMMClassifier.GMM(G, alpha, tresh, psi, False, True)
        clf.fit(DTR, LTR)
        scores = clf.transform(DTE)
        dcf_map_raw[i] = min_DCF(scores, LTE, 0.5, 1, 1)
        clf = GMMClassifier.GMM(G, alpha, tresh, psi, False, True)
        clf.fit(DTR_ZS, LTR)
        scores = clf.transform(DTE_ZS)
        dcf_map_zscore[i] = min_DCF(scores, LTE, 0.5, 1, 1)
    fig = plt.figure()
    plt.bar(np.array(range(len(G_list))) - 0.1, np.array(dcf_map_raw), width=0.2)
    plt.bar(np.array(range(len(G_list))) + 0.1, np.array(dcf_map_zscore), width=0.2)
    plt.xticks(np.array(range(len(G_list))), np.array(G_list))
    plt.legend(["Raw", "Zscore"])
    plt.xlabel("G")
    plt.ylabel("minDCF")
    plt.suptitle(f'GMMTied')
    plt.savefig(f'{plot_path}/GMMTiedEvaluation.png')
    plt.close(fig)

    # Plot for GMMDiagonal PCA(12)
    pca_reducer = PCA()
    pca_reducer.fit(DTR)
    DTR_ = pca_reducer.transform(12, DTR)
    DTE_ = pca_reducer.transform(12, DTE)
    pca_reducer.fit(DTR_ZS)
    DTR_ZS_ = pca_reducer.transform(12, DTR_ZS)
    DTE_ZS_ = pca_reducer.transform(12, DTE_ZS)
    dcf_map_raw = [0] * len(G_list)
    dcf_map_zscore = [0] * len(G_list)
    for i, G in tqdm(enumerate(G_list)):
        clf = GMMClassifier.GMM(G, alpha, tresh, psi, True, False)
        clf.fit(DTR_, LTR)
        scores = clf.transform(DTE_)
        dcf_map_raw[i] = min_DCF(scores, LTE, 0.5, 1, 1)
        clf = GMMClassifier.GMM(G, alpha, tresh, psi, True, False)
        clf.fit(DTR_ZS_, LTR)
        scores = clf.transform(DTE_ZS_)
        dcf_map_zscore[i] = min_DCF(scores, LTE, 0.5, 1, 1)
    fig = plt.figure()
    plt.bar(np.array(range(len(G_list))) - 0.1, np.array(dcf_map_raw), width=0.2)
    plt.bar(np.array(range(len(G_list))) + 0.1, np.array(dcf_map_zscore), width=0.2)
    plt.xticks(np.array(range(len(G_list))), np.array(G_list))
    plt.legend(["Raw", "Zscore"])
    plt.xlabel("G")
    plt.ylabel("minDCF")
    plt.suptitle(f'GMMDiagonal')
    plt.savefig(f'{plot_path}/GMMDiagEvaluation.png')
    plt.close(fig)

    table = PrettyTable()
    table.set_style(MARKDOWN)
    table.field_names = ['Model', 'K_class0', 'K_class1', 'pi = 0.5', 'pi = 0.2', "pi = 0.8", 'zs_pi = 0.5',
                         'zs_pi = 0.2',
                         "zs_pi = 0.8"]

    # Best models comparison
    # GMMStd K4
    clf = GMMClassifier.GMM(4, alpha, tresh, psi, False, False)
    add_row(clf, DTR, LTR, DTE, LTE, DTR_ZS, DTE_ZS, wpoints, table, "GMMStandard", 4, 4)

    # GMMStd K0=2/K1=4
    clf = GMMClassifier.GMM([2, 4], alpha, tresh, psi, False, False)
    add_row(clf, DTR, LTR, DTE, LTE, DTR_ZS, DTE_ZS, wpoints, table, "GMMStandard", 2, 4)

    # GMMTied K4
    clf = GMMClassifier.GMM(4, alpha, tresh, psi, False, True)
    add_row(clf, DTR, LTR, DTE, LTE, DTR_ZS, DTE_ZS, wpoints, table, "GMMTied", 4, 4)

    # GMMTied K0=8/K1=4
    clf = GMMClassifier.GMM([8, 4], alpha, tresh, psi, False, True)
    add_row(clf, DTR, LTR, DTE, LTE, DTR_ZS, DTE_ZS, wpoints, table, "GMMTied", 8, 4)

    print(table.get_string())
