import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
# PCA

N_ATTR = 4
N_CLASS = 3
label_names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
label_map = {"Iris-setosa": 0.0, "Iris-versicolor": 1.0, "Iris-virginica": 2.0}


def center_data(dataset):
    mu = dataset.mean(1)
    centered_dataset = dataset - mu.reshape((dataset.shape[0], 1))
    return centered_dataset


def load(filename):
    matrix = list()
    labels = list()

    with open(filename, "r") as f:
        for line in f:
            words = line.split(",")
            label_name = words[4].rstrip()
            matrix.append(np.array([float(words[0]), float(words[1]), float(words[2]), float(words[3])],
                                   dtype=np.float32).reshape(4, 1))
            labels.append(label_map[label_name])

    return np.concatenate(matrix, axis=1, dtype=np.float32), np.array(labels)


def pca(m, D):
    DC = center_data(D)
    C = DC.dot(DC.T) / DC.shape[1]
    s, U = np.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]
    return P


def mean_array(dataset):
    return dataset.mean(1).reshape((dataset.shape[0], 1))


def lda(m, D, L):
    # compute Sw
    Sw = 0
    for i in range(N_CLASS):
        Dci = center_data(D[:, L == i])
        Sw += Dci.dot(Dci.T)
    Sw /= D.shape[1]
    # compute Sb
    Sb = 0
    tot_mean = mean_array(D)
    for i in range(N_CLASS):
        class_mean = mean_array(D[:, L == i])
        nc = D[:, L == i].shape[1]
        Sb += nc * ((class_mean - tot_mean).dot((class_mean - tot_mean).T))
    Sb /= D.shape[1]
    s, U = sp.linalg.eigh(Sb, Sw)
    W = U[:, ::-1][:, 0:m]
    return W


def scatter(D, L):
    D0 = D[:, L == 0.0]
    D1 = D[:, L == 1.0]
    D2 = D[:, L == 2.0]
    plt.figure()
    plt.scatter(D0[0, :], D0[1, :])
    plt.scatter(D1[0, :], D1[1, :])
    plt.scatter(D2[0, :], D2[1, :])
    plt.show()


if __name__ == '__main__':
    D, L = load("iris.csv")
    #PCA
    P = pca(2, D)
    DP = np.dot(P.T, D)
    scatter(DP, L)

    #LDA
    W = lda(2, D, L)
    DW = np.dot(W.T, D)
    scatter(DW, L)


