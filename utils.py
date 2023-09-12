from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns
from pandas import DataFrame
import os


def load_iris():
    D, L = datasets.load_iris()['data'].T, datasets.load_iris()['target']
    return D, L


def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1] * 2.0 / 3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)


def center_data(dataset):
    mu = vcol(dataset.mean(1))
    centered_dataset = dataset - mu
    return centered_dataset, mu


def zscore(dataset):
    mu = vcol(dataset.mean(1))
    return (dataset - mu) / vcol(dataset.std(1))


def covariance(X):
    mu = vcol(X.mean(1))
    return np.dot(X - mu, (X - mu).T) / X.shape[1]


def vrow(v):
    return v.reshape((1, v.shape[0]))


def vcol(v):
    return v.reshape((v.shape[0], 1))


def PCA(m, D):
    DC, _ = center_data(D)
    C = np.dot(DC, DC.T) / DC.shape[1]
    s, U = np.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]
    return P


def KFold_CV(D, L, K, Classifier, wpoint, pca_m=0, seed=0, pre_process=None, **kwargs):
    nTest = int(D.shape[1] / K)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    scores = np.array([0.0]*D.shape[1])
    labels = np.array([0.0]*D.shape[1])
    for i in tqdm(range(K)):
        start = nTest * i
        idxTrain = np.concatenate((idx[0:start], idx[(start + nTest):]))
        idxTest = idx[start: (start + nTest)]
        DTR = D[:, idxTrain]
        DTE = D[:, idxTest]
        LTR = L[idxTrain]
        LTE = L[idxTest]
        if pre_process is not None:
            DTR = pre_process(DTR)
            DTE = pre_process(DTE)
        if pca_m != 0:
            P = PCA(pca_m, DTR)
            DTR = np.dot(P.T, DTR)
            DTE = np.dot(P.T, DTE)
        classifier = Classifier(**kwargs)
        classifier.fit(DTR, LTR)
        llr = classifier.transform(DTE)
        scores[idxTest] = llr
        labels[idxTest] = LTE
        # scores = np.hstack((scores, llr))
        # labels = np.hstack((labels, LTE))
        # min_DCF(scores, labels, wpoint[0], wpoint[1], wpoint[2]),
    return scores, labels


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
        plt.hist(d0[attr, :], bins=20, density=True, ec='black', color='Blue', alpha=0.5)
        plt.hist(d1[attr, :], bins=20, density=True, ec='black', color='Red', alpha=0.5)
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

