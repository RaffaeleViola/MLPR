import numpy as np
import scipy.optimize as so


class SVM:
    def __init__(self, p_T=0.9, C=1.0, k=1):
        self.w_s = None
        self.p_T = p_T
        self.C = C
        self.k = k

    @staticmethod
    def linear_svm_obj_wrap(DTR, LTR, K=1):
        ones = np.ones(DTR.shape[1]).reshape((1, DTR.shape[1]))
        D = np.vstack((DTR, K * ones))
        G = np.dot(D.T, D)
        z = (2 * LTR - 1).reshape((1, DTR.shape[1]))
        H = np.dot(z.T, z) * G

        def linear_svm_obj(_alpha):
            alpha = _alpha.reshape((DTR.shape[1], 1))
            dual = 0.5 * np.dot(alpha.T, H).dot(alpha) - np.dot(alpha.T, ones.reshape((DTR.shape[1], 1)))
            gradient = (np.dot(H, alpha) - ones.reshape((DTR.shape[1], 1))).reshape((1, DTR.shape[1]))
            return dual, gradient

        return linear_svm_obj

    @staticmethod
    def w_star(DTR, LTR, _alpha, K=1):
        ones = K * np.ones(DTR.shape[1]).reshape((1, DTR.shape[1]))
        D = np.vstack((DTR, ones))
        a_z = (_alpha * (2 * LTR - 1)).reshape((1, DTR.shape[1]))
        return (a_z * D).sum(axis=1)

    @staticmethod
    def weighted_bounds(LTR, p_T, C):
        p_Temp = (LTR == 1).sum() / LTR.shape[0]
        C_T = C * p_T / p_Temp
        C_F = C * (1 - p_T) / (1 - p_Temp)
        bounds = [(0, C_T) if LTR[i] == 1 else (0, C_F) for i in range(LTR.shape[0])]
        return bounds

    @staticmethod
    def scores(DTE, w_s, K):
        ones = K * np.ones(DTE.shape[1]).reshape((1, DTE.shape[1]))
        D_mapped = np.vstack((DTE, ones))
        return np.dot(w_s.reshape((w_s.shape[0]), 1).T, D_mapped).reshape((DTE.shape[1],))

    def fit(self, DTR, LTR):
        dual_obj = SVM.linear_svm_obj_wrap(DTR, LTR, K=self.k)
        bounds = SVM.weighted_bounds(LTR, self.p_T, self.C)
        x0 = np.zeros(DTR.shape[1])
        x, f, d = so.fmin_l_bfgs_b(dual_obj, x0, bounds=bounds, factr=1.0)
        self.w_s = SVM.w_star(DTR, LTR, x, K=self.k)

    def transform(self, DTE):
        return SVM.scores(DTE, self.w_s, self.k)


class PolynomialSVM(SVM):

    def __init__(self, p_T=0.5, C=1.0, k=1.0, d=2, c=1):
        super().__init__(p_T, C, k)
        self.LTR = None
        self.DTR = None
        self.alpha = None
        self.d = d
        self.c = c

    @staticmethod
    def polynomial_kernel(x1, x2, kwargs):
        c = kwargs['c']
        d = kwargs['d']
        eps = kwargs['eps']
        return np.power(np.dot(x1.T, x2) + c, d) + eps

    @staticmethod
    def kernel_svm_obj_wrap(DTR, LTR, kernel, **kwargs):
        ones = np.ones(DTR.shape[1]).reshape((1, DTR.shape[1]))
        z = (2 * LTR - 1).reshape((1, DTR.shape[1]))
        H = np.dot(z.T, z) * kernel(DTR, DTR, kwargs)

        def kernel_svm_obj(_alpha):
            alpha = _alpha.reshape((DTR.shape[1], 1))
            dual = 0.5 * np.dot(alpha.T, H).dot(alpha) - np.dot(alpha.T, ones.reshape((DTR.shape[1], 1)))
            gradient = (np.dot(H, alpha) - ones.reshape((DTR.shape[1], 1))).reshape((DTR.shape[1],))
            return dual, gradient

        return kernel_svm_obj

    @staticmethod
    def kernel_scores(DTR, DTE, LTR, alpha, kernel, **kwargs):
        z = (2 * LTR - 1)
        az = (alpha * z).reshape((DTR.shape[1], 1))
        return (az * kernel(DTR, DTE, kwargs)).sum(axis=0)

    def fit(self, DTR, LTR):
        self.DTR = DTR
        self.LTR = LTR
        bounds = SVM.weighted_bounds(LTR, self.p_T, self.C)
        eps = self.k ** 2
        k_svm_obj = PolynomialSVM.kernel_svm_obj_wrap(DTR, LTR, PolynomialSVM.polynomial_kernel,
                                                      c=self.c, d=self.d, eps=eps)
        x0 = np.zeros(DTR.shape[1])
        x, f, d_ = so.fmin_l_bfgs_b(k_svm_obj, x0, bounds=bounds, factr=1.0)
        self.alpha = x

    def transform(self, DTE):
        return PolynomialSVM.kernel_scores(self.DTR, DTE, self.LTR, self.alpha,
                                           PolynomialSVM.polynomial_kernel, c=self.c, d=self.d, eps=(self.k**2))

class RBFSVM(SVM):

    def __init__(self, p_T=0.5, C=1.0, k=1.0, gamma=1.0):
        super().__init__(p_T, C, k)
        self.LTR = None
        self.DTR = None
        self.alpha = None
        self.gamma = gamma

    @staticmethod
    def func1D(a, *args, **kwargs):
        x2 = kwargs['x2']
        return np.apply_along_axis(lambda x: ((a - x) * (a - x)).sum(), axis=0, arr=x2)

    @staticmethod
    def radial_basis_kernel(x1, x2, kwargs):
        gamma = kwargs['gamma']
        eps = kwargs['eps']
        norm = np.apply_along_axis(func1d=RBFSVM.func1D, axis=0, arr=x1, x2=x2).T
        return np.exp(-gamma * norm) + eps

    def fit(self, DTR, LTR):
        self.DTR = DTR
        self.LTR = LTR
        bounds = SVM.weighted_bounds(LTR, self.p_T, self.C)
        eps = self.k ** 2
        k_svm_obj = PolynomialSVM.kernel_svm_obj_wrap(DTR, LTR, RBFSVM.radial_basis_kernel, gamma=self.gamma, eps=eps)
        x0 = np.zeros(DTR.shape[1])
        x, f, d_ = so.fmin_l_bfgs_b(k_svm_obj, x0, bounds=bounds, factr=1.0)
        self.alpha = x

    def transform(self, DTE):
        return PolynomialSVM.kernel_scores(self.DTR, DTE, self.LTR, self.alpha,
                                           RBFSVM.radial_basis_kernel, gamma=self.gamma, eps=(self.k ** 2))

