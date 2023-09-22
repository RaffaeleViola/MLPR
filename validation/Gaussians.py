from prettytable import PrettyTable, MARKDOWN
from Models import GaussianClassifiers
from measures import *


def GaussiansValidation():
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    score_path = f'{absolute_path}/../scores/train'
    if not os.path.exists(score_path):
        os.mkdir(score_path)

    # define K for K-FOld Cross Validation
    K = 5

    # define working point
    wpoints = [(0.5, 1, 1), (0.2, 1, 1), (0.8, 1, 1)]

    # import training data
    D, L = load_dataset("Train.txt")

    # define PCA m values list
    m_list = [0, 12, 11, 10, 9, 8]  # example values - 0 mandatory for no PCA training

    # define data_preprocessing strategies
    pre_processing = {"None": None, "zscore": zscore}  # None is RAW data

    # define classifiers
    classifiers = [GaussianClassifiers.MVGClassifier, GaussianClassifiers.NaiveBayesClassifier]
    classifier_map = [{False: "MVG", True: "TMVG"}, {False: "NB", True: "TNB"}]

    tables = [PrettyTable() for _ in range(len(m_list))]
    for table in tables:
        table.set_style(MARKDOWN)
        table.field_names = ['Model', 'pi = 0.5', 'pi = 0.2', "pi = 0.8", 'zs_pi = 0.5', 'zs_pi = 0.2', "zs_pi = 0.8"]

    for i, classfier in enumerate(classifiers):
        for tied in [False, True]:
            for k, m in enumerate(m_list):
                minDCF = [0] * (len(wpoints) * 2)
                for j, (p_T, Cfn, Cfp) in enumerate(wpoints):
                        scores, labels = KFold_CV(D, L, 5, classfier, wpoint=(p_T, Cfn, Cfp),
                                                  pca_m=m, pre_process=None, tied=tied)
                        minDCF[j] = min_DCF(scores, labels, p_T, Cfn, Cfp)
                        np.save(f'{score_path}/{classifier_map[i][tied]}_tied{tied}_m{m}_preNone',
                                np.array([scores, labels]))
                        scores, labels = KFold_CV(D, L, 5, classfier, wpoint=(p_T, Cfn, Cfp),
                                                  pca_m=m, pre_process=zscore, tied=tied)
                        np.save(f'{score_path}/{classifier_map[i][tied]}_tied{tied}_m{m}_prezscore', np.array([scores, labels]))
                        minDCF[(j + len(wpoints))] = min_DCF(scores, labels, p_T, Cfn, Cfp)
                name = classifier_map[i][tied]
                tables[k].add_row([name, *minDCF])

    for i, table in enumerate(tables):
        print(f"\n---------PCA({m_list[i]})-----------\n")
        print(table.get_string())
