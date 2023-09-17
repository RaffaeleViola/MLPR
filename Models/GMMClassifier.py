import math
import numpy as np
import scipy
from utils import vcol, vrow, center_data

N_CLASS = 2


class GMM:
    def __init__(self, G, alpha, tresh, psi, diag=False, tied=False):
        self.n_classes = N_CLASS
        self.params = None
        self.psi = psi
        self.diag = diag
        self.tied = tied
        self.tresh = tresh
        self.alpha = alpha
        self.G = G

    @staticmethod
    def logpdf_GAU_ND(x, mu, C):
        log_vect = []
        for i in range(x.shape[1]):
            e1 = (C.shape[0] / 2) * math.log(2 * math.pi)
            e2 = np.linalg.slogdet(C)[1] / 2
            t = np.dot((x[:, i:i + 1] - mu).T, np.linalg.inv(C))
            e3 = 0.5 * np.dot(t, (x[:, i:i + 1] - mu))
            log_vect.append((-e1 - e2 - e3))
        return np.array(log_vect).ravel()

    @staticmethod
    def logpdf_GMM(X, gmm):
        S = np.array([GMM.logpdf_GAU_ND(X, cluster[1], cluster[2]) for cluster in gmm])
        w = vcol(np.array([cluster[0] for cluster in gmm]))
        S += np.log(w)
        logdens = scipy.special.logsumexp(S, axis=0)
        return logdens, S

    @staticmethod
    def constraining(sigma, psi):
        U, s, _ = np.linalg.svd(sigma)
        s[s < psi] = psi
        sigma = np.dot(U, vcol(s) * U.T)
        return sigma

    @staticmethod
    def expectation_maximization(X, gmm, tresh, psi, diag=False, tied=False):
        logdens, S = GMM.logpdf_GMM(X, gmm)
        ll_avg_prev = logdens.sum() / logdens.shape[0]
        while True:
            # E-step
            S -= logdens
            Ygi = np.exp(S)
            # M-step
            zg_vec = []
            mu_vec = []
            sigma_vec = []
            for g in range(len(gmm)):
                Yg = Ygi[g, :]
                Zg = Yg.sum()
                zg_vec.append(Zg)
                tmp = vrow(Yg) * X
                Fg = tmp.sum(axis=1)
                Sg = np.dot(X, tmp.T)
                mu = vcol(Fg / Zg)
                sigma = Sg / Zg - np.dot(mu, mu.T)
                if diag:
                    sigma = sigma * np.eye(sigma.shape[0])
                # constraining
                if not tied:
                    sigma = GMM.constraining(sigma, psi)
                mu_vec.append(mu)
                sigma_vec.append(sigma)
            if tied:
                tied_sum = 0
                for i in range(len(zg_vec)):
                    tied_sum += zg_vec[i] * sigma_vec[i]
                sigma_tied = tied_sum / X.shape[1]
                sigma_tied = GMM.constraining(sigma_tied, psi)
                gmm_new = [(zg_vec[i] / X.shape[1], mu_vec[i], sigma_tied) for i in range(len(gmm))]
            else:
                gmm_new = [(zg_vec[i] / X.shape[1], mu_vec[i], sigma_vec[i]) for i in range(len(gmm))]
            logdens, S = GMM.logpdf_GMM(X, gmm_new)
            ll_avg_new = logdens.sum() / logdens.shape[0]
            if ll_avg_new - ll_avg_prev < tresh:
                break
            ll_avg_prev = ll_avg_new

        return gmm_new

    @staticmethod
    def LBG(X, G, alpha, tresh, psi, diag=False, tied=False):
        n_its = int(math.log2(G))
        mu = vcol(X.mean(1))
        C = np.dot((X - mu), (X - mu).T) / X.shape[1]
        if diag:
            C = C * np.eye(C.shape[0])
        # constraining
        C = GMM.constraining(C, psi)
        gmm = [(1.0, mu, C)]
        if n_its == 0:
            return GMM.expectation_maximization(X, gmm, tresh, psi, diag=diag, tied=tied)
        for g in range(n_its):
            gmm_new = []
            for cluster in gmm:
                U, s, Vh = np.linalg.svd(cluster[2])
                d = vcol(U[:, 0:1] * s[0] ** 0.5 * alpha)
                gmm_new.append((cluster[0] / 2, cluster[1] + d, cluster[2]))
                gmm_new.append((cluster[0] / 2, cluster[1] - d, cluster[2]))
            gmm = GMM.expectation_maximization(X, gmm_new, tresh, psi, diag=diag, tied=tied)
        return gmm

    @staticmethod
    def compute_log_score_matrix(DTE, params, n_classes):
        score_matrix = []
        for i in range(n_classes):
            class_vec = []
            for j in range(DTE.shape[1]):
                logdens, _ = GMM.logpdf_GMM(vcol(DTE[:, j]), params[i])
                class_vec.append(logdens.sum())
            score_matrix.append(class_vec)
        return np.array(score_matrix)

    def fit(self, D, L):
        if isinstance(self.G, list):
            self.params = [GMM.LBG(D[:, L == i], self.G[i], self.alpha,
                                   self.tresh, self.psi, self.diag, self.tied)
                           for i in range(self.n_classes)]
        else:
            self.params = [GMM.LBG(D[:, L == i], self.G, self.alpha,
                                   self.tresh, self.psi, self.diag, self.tied)
                           for i in range(self.n_classes)]

    def transform(self, DTE):
        log_score_matrix = GMM.compute_log_score_matrix(DTE, self.params, self.n_classes)
        llr = log_score_matrix[1, :] - log_score_matrix[0, :]
        return llr
