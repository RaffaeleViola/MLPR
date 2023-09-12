from Models import LogisticRegression
import numpy as np
from utils import KFold_CV


class Fusion:
    def __init__(self):
        self.p_t = None
        self.scores = None
        self.labels = None
        self.K = None
        self.lmd = None

    def fit(self, K, lmd, prior):
        self.p_t = prior
        self.lmd = lmd
        self.K = K

    def transform(self, score_list, labels):
        scores = np.array(np.vstack(score_list))
        labels = labels
        scores, labels = KFold_CV(scores, labels, self.K, LogisticRegression.LogisticRegression,
                                  pca_m=0, seed=13, pre_process=None, lmd=self.lmd, prior=self.p_t)
        return scores
