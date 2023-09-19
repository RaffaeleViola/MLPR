from Models import LogisticRegression, SVM, FusionModel
from measures import *
from itertools import combinations
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

alpha, tresh, psi = 0.1, 1e-6, 0.01

D, L = load_dataset("Train.txt")

final_prior = 0.3

# Loadind scores
lr_scores, labels = np.load(f'{score_path}/LR_m{0}_preNone_prior{0.7}_lmd{1e-3}.npy')
rbfsvm_scores, _ = np.load(f'{score_path}/RBFSVM_m{0}_preNone_prior{0.7}_C{10}_k{1}_gamma{0.001}.npy')
gmm_scores, _ = np.load(f'{score_path}/GMMTied_G{4}_m{0}_preNone_a{alpha}_t{tresh}_psi{psi}.npy')

table = PrettyTable()
table.set_style(MARKDOWN)
table.field_names = ['Model', 'prior', 'minDCF p = 0.5', 'actDCF p = 0.5',
                     'minDCF p = 0.2', 'actDCF p = 0.2',
                     'minDCF p = 0.8', 'actDCF p = 0.8']
# Fusion of all combinations
for (name1, scores1), (name2, scores2) in combinations({"RBFSVM": rbfsvm_scores,
                                                        "LR": lr_scores, "GMMTied": gmm_scores}.items(), 2):
    scores = calibrate(np.vstack([scores1, scores2]), labels, final_prior)
    bayes_error_plot(scores, labels, f'Fusion {name1} + {name2}')
    minDCF_5 = min_DCF(scores, labels, 0.5, 1, 1)
    actDCf_5 = act_DCF(scores, labels, 0.5, 1, 1)
    minDCF_2 = min_DCF(scores, labels, 0.2, 1, 1)
    actDCf_2 = act_DCF(scores, labels, 0.2, 1, 1)
    minDCF_8 = min_DCF(scores, labels, 0.8, 1, 1)
    actDCf_8 = act_DCF(scores, labels, 0.8, 1, 1)
    table.add_row([f'{name1} + {name2}', final_prior, minDCF_5, actDCf_5, minDCF_2, actDCf_2, minDCF_8, actDCf_8])

# Fusion with all three models
scores = calibrate(np.vstack([lr_scores, gmm_scores, rbfsvm_scores]), labels, final_prior)
bayes_error_plot(scores, labels, f'Fusion RBFSVM + GMMTied + LR')
minDCF_5 = min_DCF(scores, labels, 0.5, 1, 1)
actDCf_5 = act_DCF(scores, labels, 0.5, 1, 1)
minDCF_2 = min_DCF(scores, labels, 0.2, 1, 1)
actDCf_2 = act_DCF(scores, labels, 0.2, 1, 1)
minDCF_8 = min_DCF(scores, labels, 0.8, 1, 1)
actDCf_8 = act_DCF(scores, labels, 0.8, 1, 1)
table.add_row([f'GMMTied + RBFSVM + LR', final_prior, minDCF_5, actDCf_5, minDCF_2, actDCf_2, minDCF_8, actDCf_8])

print(table.get_string())
