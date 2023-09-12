from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns
from pandas import DataFrame
import os
import scipy as sp
from Models import LogisticRegression



def center_data(dataset):
    mu = vcol(dataset.mean(1))
    centered_dataset = dataset - mu
    return centered_dataset, mu


def zscore(DTR, DTE):
    mu = vcol(DTR.mean(1))
    std = vcol(DTR.std(1))
    return (DTR - mu) / std, (DTE - mu) / std


def covariance(X):
    mu = vcol(X.mean(1))
    return np.dot(X - mu, (X - mu).T) / X.shape[1]


def vrow(v):
    return v.reshape((1, v.shape[0]))


def vcol(v):
    return v.reshape((v.shape[0], 1))


class PCA:
    def __init__(self):
        self.U = None
        self.s = None

    def fit(self, D):
        DC, _ = center_data(D)
        C = np.dot(DC, DC.T) / DC.shape[1]
        s, U = np.linalg.eigh(C)
        self.U = U
        self.s = s

    def transform(self, m, D):
        return np.dot(self.U[:, ::-1][:, 0:m].T, D)

    def explained_variance(self):
        return self.s[::-1].cumsum() / self.s.sum()


class LDA:
    def __init__(self, C):
        self.C = C
        self.U = None

    def fit(self, D, L):
        # compute Sw
        Sw = 0
        for i in range(self.C):
            Dci, _ = center_data(D[:, L == i])
            Sw += Dci.dot(Dci.T)
        Sw /= D.shape[1]
        # compute Sb
        Sb = 0
        tot_mean = vcol(D.mean(1))
        for i in range(self.C):
            class_mean = vcol(D[:, L == i].mean(1))
            nc = D[:, L == i].shape[1]
            Sb += nc * ((class_mean - tot_mean).dot((class_mean - tot_mean).T))
        Sb /= D.shape[1]
        s, U = sp.linalg.eigh(Sb, Sw)
        self.U = U

    def transform(self, m, D):
        return np.dot(self.U[:, ::-1][:, 0:m].T, D)


def KFold_CV(D, L, K, Classifier, wpoint=None, pca_m=0, seed=0, pre_process=None, **kwargs):
    nTest = int(D.shape[1] / K)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    scores = np.array([0.0]*D.shape[1])
    labels = np.array([0.0]*D.shape[1])
    PCA_reducer = PCA()
    for i in tqdm(range(K)):
        start = nTest * i
        idxTrain = np.concatenate((idx[0:start], idx[(start + nTest):]))
        idxTest = idx[start: (start + nTest)]
        DTR = D[:, idxTrain]
        DTE = D[:, idxTest]
        LTR = L[idxTrain]
        LTE = L[idxTest]
        if pre_process is not None:
            DTR, DTE = pre_process(DTR, DTE)
        if pca_m != 0:
            PCA_reducer.fit(DTR)
            DTR = PCA_reducer.transform(pca_m, DTR)
            DTE = PCA_reducer.transform(pca_m, DTE)
        classifier = Classifier(**kwargs)
        classifier.fit(DTR, LTR)
        llr = classifier.transform(DTE)
        scores[idxTest] = llr
        labels[idxTest] = LTE
    return scores, labels


def calibrate(scrs, labels, pT):
    scrs, labels = KFold_CV(vrow(scrs), labels, 5, LogisticRegression.LogisticRegression,
                            pca_m=0, seed=13, pre_process=None, lmd=0, prior=0.5)
    scrs = scrs - np.log(pT / (1 - pT))
    return scrs


def evaluation(DTR, LTR, DTE, Classifier, m=0, pre_process=None, **kwargs):
    DTR_r = DTR
    DTE_r = DTE
    if pre_process is not None:
        DTR_r, DTE_r = pre_process(DTR, DTE)
    if m != 0:
        PCA_reducer = PCA()
        PCA_reducer.fit(DTR)
        DTR_r = PCA_reducer.transform(m, DTR_r)
        DTE_r = PCA_reducer.transform(m, DTE_r)
    clf = Classifier(**kwargs)
    clf.fit(DTR_r, LTR)
    scores = clf.transform(DTE_r)
    return scores


def num_corrects(Pred, LTE):
    res_vec = Pred - LTE
    corr_pred = 0
    for i in range(res_vec.shape[0]):
        if res_vec[i] == 0:
            corr_pred += 1
    return corr_pred


def load_dataset(file_name):
    dataset = []
    labels = []
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    filename = absolute_path + f'/{file_name}'
    with open(filename, "r") as file:
        for line in file.readlines():
            feats = line.rstrip().split(",")
            dataset.append(vcol(np.array([float(feats[i]) for i in range(12)])))
            labels.append(int(feats[12]))
    return np.concatenate(dataset, axis=1), np.array(labels)


def plot_hist(dataset, labels, prefix=""):
    make_dir("Images")
    make_dir("hist")
    label_names = ["Male", "Female"]
    mask0 = (labels == 0.0)
    d0 = dataset[:, mask0]
    mask1 = (labels == 1.0)
    d1 = dataset[:, mask1]
    for attr in range(dataset.shape[0]):
        fig = plt.figure()
        plt.hist(d0[attr, :], bins=50, density=True, ec='black', color='Blue', alpha=0.5)
        plt.hist(d1[attr, :], bins=50, density=True, ec='black', color='Red', alpha=0.5)
        plt.legend(label_names)
        plt.title(f'Feature no. {attr}')
        plt.savefig(f'./Images/hist/{prefix}_feat_{attr}.png')
        plt.close(fig)


def make_dir(dirname):
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(f'{absolute_path}/Images/{dirname}'):
        # If it doesn't exist, create it
        os.mkdir(f'{absolute_path}/Images/{dirname}')


def plot_scatter(dataset, labels, prefix=""):
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    make_dir("Images")
    make_dir("scatter")
    label_names = ["Male", "Female"]
    mask0 = (labels == 0.0)
    d0 = dataset[:, mask0]
    mask1 = (labels == 1.0)
    d1 = dataset[:, mask1]
    mask2 = (labels == 2.0)
    d2 = dataset[:, mask2]
    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[0]):
            if i == j:
                continue
            fig = plt.figure()
            plt.scatter(d0[i, :], d0[j, :])
            plt.scatter(d1[i, :], d1[j, :])
            plt.scatter(d2[i, :], d2[j, :])
            plt.xlabel(f'Feature no. {i}')
            plt.ylabel(f'Feature no. {j}')
            plt.legend(label_names)
            plt.savefig(f'{absolute_path}/Images/scatter/{prefix}_feat_{i}_{j}.png')
            plt.close(fig)


def corr_map(D, name, cmap="Greys"):
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    make_dir("Images")
    make_dir("correlation")
    corr = DataFrame(D.T).corr(method="pearson").abs()
    fig = plt.figure()
    sns.heatmap(corr, cmap=cmap)
    plt.savefig(f'{absolute_path}/Images/correlation/{name}.png')
    plt.close(fig)

