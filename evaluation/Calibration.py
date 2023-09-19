from Models import SVM, LogisticRegression, GaussianClassifiers, GMMClassifier
from measures import *
from utils import *
from prettytable import PrettyTable, MARKDOWN

absolute_path = os.path.dirname(os.path.abspath(__file__))
score_path = f'{absolute_path}/../scores/train'
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

alpha, tresh, psi = 0.1, 1e-6, 0.01

# calibrator obj
calibrator = LogisticRegression.LogisticRegression(lmd=0, prior=0.3)

# Load scores of best models from validation
# TMVG(RAW + zscore)
tmvg_scores, labels = np.load(f'{score_path}/TMVG_tied{True}_m{0}_prezscore.npy')
tmvg = GaussianClassifiers.MVGClassifier(tied=True)
tmvg.fit(DTR_ZS, LTR)
# LR Linear lambda=10-3p_T=0.7
lr_scores, _ = np.load(f'{score_path}/LR_m{0}_preNone_prior{0.7}_lmd{1e-3}.npy')
lr = LogisticRegression.LogisticRegression(lmd=1e-3, prior=0.7)
lr.fit(DTR, LTR)
# RBFSVM p_T=0.7 gamma 0.001 C=10
rbfsvm_scores, _ = np.load(f'{score_path}/RBFSVM_m{0}_preNone_prior{0.7}_C{10}_k{1}_gamma{0.001}.npy')
rbfsvm = SVM.RBFSVM(p_T=0.7, C=10, k=1, gamma=0.001)
rbfsvm.fit(DTR, LTR)
# GMMTied K=4
gmm_scores, _ = np.load(f'{score_path}/GMMTied_G{4}_m{0}_preNone_a{alpha}_t{tresh}_psi{psi}.npy')
gmm = GMMClassifier.GMM(4, alpha, tresh, psi, diag=False, tied=True)
gmm.fit(DTR, LTR)

table = PrettyTable()
table.set_style(MARKDOWN)
table.field_names = ['Model', 'validation_minDCF', 'validation_actDCF', 'evaluation_minDCF', 'evaluation_actDCF']

det_list = []
for name, (scores, clf) in {"TMVG": (tmvg_scores, tmvg), "RBFSVM": (rbfsvm_scores, rbfsvm),
                            "LR": (lr_scores, lr), "GMMTied": (gmm_scores, gmm)}.items():
    DCF = []
    # Validation set
    scores_calibrated = calibrate(vrow(scores), labels, 0.3)
    DCF.append(min_DCF(scores_calibrated, labels, 0.5, 1, 1))
    DCF.append(act_DCF(scores_calibrated, labels, 0.5, 1, 1))
    # Evaluation set
    calibrator.fit(vrow(scores), labels)
    if name == "TMVG":
        scores_DTE = clf.transform(DTE_ZS)
    else:
        scores_DTE = clf.transform(DTE)
    scores_DTE_cal = calibrator.transform(vrow(scores_DTE))
    scores_DTE_cal -= np.log(0.3 / (1 - 0.3))
    det_list.append((scores_DTE_cal, name))
    DCF.append(min_DCF(scores_DTE_cal, LTE, 0.5, 1, 1))
    DCF.append(act_DCF(scores_DTE_cal, LTE, 0.5, 1, 1))
    bayes_error_plot(scores_DTE_cal, LTE, f'Eval_{name}')
    table.add_row([name, *DCF])

print(table.get_string())
plot_det([el[0] for el in det_list], LTE, [el[1] for el in det_list], "bestModelsDETNoncalibrated")
