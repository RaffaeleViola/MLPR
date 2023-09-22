from Models import SVM, LogisticRegression, GaussianClassifiers, GMMClassifier
from measures import *
from prettytable import PrettyTable, MARKDOWN

absolute_path = os.path.dirname(os.path.abspath(__file__))
score_path = f'{absolute_path}/../scores/train'
if not os.path.exists(score_path):
    os.mkdir(score_path)


def minactDCF(scores, labels, p_T, Cfn, Cfp):
    vDCF = [min_DCF(scores, labels, p_T, Cfn, Cfp), act_DCF(scores, labels, p_T, Cfn, Cfp)]
    return vDCF


def CalibrationValidation():
    wpoints = [(0.5, 1, 1), (0.2, 1, 1), (0.8, 1, 1)]
    alpha, tresh, psi = 0.1, 1e-6, 0.01

    # ------------------------------------------------- Best models
    table = PrettyTable()
    table.set_style(MARKDOWN)
    table.field_names = ['Model', 'pi = 0.5', 'pi = 0.2', "pi = 0.8"]
    # TMVG(RAW + zscore)
    minDCF = []
    scores, labels = np.load(f'{score_path}/TMVG_tied{True}_m{0}_prezscore.npy')
    for p_T, Cfn, Cfp in wpoints:
        minDCF.append(min_DCF(scores, labels, p_T, Cfn, Cfp))
    table.add_row(["TMVG(zscore)", *minDCF])
    # LR Linear lambda=10-3p_T=0.7
    minDCF = []
    scores, labels = np.load(f'{score_path}/LR_m{0}_preNone_prior{0.7}_lmd{1e-3}.npy')
    for p_T, Cfn, Cfp in wpoints:
        minDCF.append(min_DCF(scores, labels, p_T, Cfn, Cfp))
    table.add_row(["LinearLR (lmd=10^-3, p_T=0.7)", *minDCF])
    # RBFSVM p_T=0.7 gamma 0.001 C=10
    minDCF = []
    scores, label = np.load(f'{score_path}/RBFSVM_m{0}_preNone_prior{0.7}_C{10}_k{1}_gamma{0.001}.npy')
    for p_T, Cfn, Cfp in wpoints:
        minDCF.append(min_DCF(scores, labels, p_T, Cfn, Cfp))
    table.add_row(["RBFSVM (C=10, gamma=0.001, p_T=0.7)", *minDCF])
    # GMMTied K=4
    minDCF = []
    scores, labels = np.load(f'{score_path}/GMMTied_G{4}_m{0}_preNone_a{alpha}_t{tresh}_psi{psi}.npy')
    for p_T, Cfn, Cfp in wpoints:
        minDCF.append(min_DCF(scores, labels, p_T, Cfn, Cfp))
    table.add_row(["GMMTied K=4", *minDCF])

    print(table.get_string())

    # --------------------------------------------------------- minDCf to actDCf
    table = PrettyTable()
    table.set_style(MARKDOWN)
    table.field_names = ['Model', 'minDCF', 'actDCF']
    table.add_row(['', 'pi = 0.5', 'pi = 0.5'])
    # TMVG(RAW + zscore)
    tmvg_scores, labels = np.load(f'{score_path}/TMVG_tied{True}_m{0}_prezscore.npy')
    table.add_row(["TMVG(zscore)", *minactDCF(tmvg_scores, labels, 0.5, 1, 1)])
    # LR Linear lambda=10-3p_T=0.7
    lr_scores, labels = np.load(f'{score_path}/LR_m{0}_preNone_prior{0.7}_lmd{1e-3}.npy')
    table.add_row(["LinearLR (lmd=10^-3, p_T=0.7)", *minactDCF(lr_scores, labels, 0.5, 1, 1)])
    # RBFSVM p_T=0.7 gamma 0.001 C=10
    rbfsvm_scores, labels = np.load(f'{score_path}/RBFSVM_m{0}_preNone_prior{0.7}_C{10}_k{1}_gamma{0.001}.npy')
    table.add_row(["RBFSVM (C=10, gamma=0.001, p_T=0.7)", *minactDCF(rbfsvm_scores, labels, 0.5, 1, 1)])
    # GMMTied K=4
    gmm_scores, labels = np.load(f'{score_path}/GMMTied_G{4}_m{0}_preNone_a{alpha}_t{tresh}_psi{psi}.npy')
    table.add_row(["GMMTied K=4", *minactDCF(gmm_scores, labels, 0.5, 1, 1)])
    print(table.get_string())

    # -------------------------------------------- Bayes Error Plots
    # Logistic Regression
    bayes_error_plot(lr_scores, labels, "LR_non_calibrated")
    # RBFSVM
    bayes_error_plot(rbfsvm_scores, labels, "RBFSVM_non_calibrated")
    # TMVG(RAW + zscore)
    bayes_error_plot(tmvg_scores, labels, "TMVG_non_calibrated")
    # GMMTied K=4
    bayes_error_plot(gmm_scores, labels, "GMMTied_non_calibrated")

    # ------------------------------------------- Calibration prior validation
    table = PrettyTable()
    table.set_style(MARKDOWN)
    table.field_names = ['Model', 'prior', 'minDCF(Calibrated scores)', 'actDCF']
    table.add_row(['', '', 'pi = 0.5', 'pi = 0.5'])
    for name, scores in {"TMVG": tmvg_scores, "RBFSVM": rbfsvm_scores, "LR": lr_scores, "GMMTied": gmm_scores}.items():
        for train_prior in [0.2, 0.3, 0.5, 0.7, 0.8]:
            calibrated_scores = calibrate(vrow(scores), labels, train_prior)
            minDCF = min_DCF(calibrated_scores, labels, 0.5, 1, 1)
            actDCf = act_DCF(calibrated_scores, labels, 0.5, 1, 1)
            table.add_row([name, train_prior, minDCF, actDCf])
        table.add_row(['', '', '', ''])
    print(table.get_string())
    final_prior = 0.3

    # ------------------------------------------- Calibration and Bayes Error Plots
    # Logistic Regression
    scores = calibrate(vrow(lr_scores), labels, final_prior)
    name = "LR_calibrated"
    bayes_error_plot(scores, labels, name)

    # RBFSVM
    scores = calibrate(vrow(rbfsvm_scores), labels, final_prior)
    name = "RBFSVM_calibrated_zscore"
    bayes_error_plot(scores, labels, name)

    # TMVG(RAW + zscore)
    scores = calibrate(vrow(tmvg_scores), labels, final_prior)
    name = "TMVG_calibrated_zscore"
    bayes_error_plot(scores, labels, name)

    # GMMTied K=4
    scores = calibrate(vrow(gmm_scores), labels, final_prior)
    name = "GMMTied_calibrated_zscore"
    bayes_error_plot(scores, labels, name)

    # ------------------------------------------- Calibration overview on all wpoints
    table = PrettyTable()
    table.set_style(MARKDOWN)
    table.field_names = ['Model', 'prior', 'minDCF p = 0.5', 'actDCF p = 0.5',
                         'minDCF p = 0.2', 'actDCF p = 0.2',
                         'minDCF p = 0.8', 'actDCF p = 0.8']
    for name, scores in {"TMVG": tmvg_scores, "RBFSVM": rbfsvm_scores, "LR": lr_scores, "GMMTied": gmm_scores}.items():
        calibrated_scores = calibrate(vrow(scores), labels, final_prior)
        minDCF_5 = min_DCF(calibrated_scores, labels, 0.5, 1, 1)
        actDCf_5 = act_DCF(calibrated_scores, labels, 0.5, 1, 1)
        minDCF_2 = min_DCF(calibrated_scores, labels, 0.2, 1, 1)
        actDCf_2 = act_DCF(calibrated_scores, labels, 0.2, 1, 1)
        minDCF_8 = min_DCF(calibrated_scores, labels, 0.8, 1, 1)
        actDCf_8 = act_DCF(calibrated_scores, labels, 0.8, 1, 1)
        table.add_row([name, final_prior, minDCF_5, actDCf_5, minDCF_2, actDCf_2, minDCF_8, actDCf_8])

    print(table.get_string())
