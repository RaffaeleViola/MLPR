import numpy as np
import matplotlib.pyplot as plt

N_ATTR = 4

label_names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
label_map = {"Iris-setosa": 0.0, "Iris-versicolor": 1.0, "Iris-virginica": 2.0}
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


def plot_hist(dataset, labels):
    mask0 = (labels == 0.0)
    d0 = dataset[:, mask0]
    mask1 = (labels == 1.0)
    d1 = dataset[:, mask1]
    mask2 = (labels == 2.0)
    d2 = dataset[:, mask2]
    plt.figure()
    f, ax = plt.subplots(2, 2)
    cor = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for attr in range(N_ATTR):
        ax[cor[attr]].hist(d0[attr, :], bins=20, density=True, ec='black', color='Blue', alpha=0.5)
        ax[cor[attr]].hist(d1[attr, :], bins=20, density=True, ec='black', color='Red', alpha=0.5)
        ax[cor[attr]].hist(d2[attr, :], bins=20, density=True, ec='black', color='Green', alpha=0.5)
        ax[cor[attr]].legend(label_names)

    plt.show()


def plot_scatter(dataset, labels):
    mask0 = (labels == 0.0)
    d0 = dataset[:, mask0]
    mask1 = (labels == 1.0)
    d1 = dataset[:, mask1]
    mask2 = (labels == 2.0)
    d2 = dataset[:, mask2]
    plt.figure()
     # cor = [(i, j) for i in range(1) for j in range(3)]
    for i in range(N_ATTR):
        f, ax = plt.subplots(3, 1)
        n = 0
        for j in range(N_ATTR):
            if i == j:
                continue
            ax[n].scatter(d0[i, :], d0[j, :])
            ax[n].scatter(d1[i, :], d1[j, :])
            ax[n].scatter(d2[i, :], d2[j, :])
            ax[n].legend(label_names)
            n += 1
        plt.show()



def center_data(dataset):
    mu = dataset.mean(1)
    centered_dataset = dataset - mu.reshape((dataset.shape[0], 1))
    return centered_dataset


if __name__ == '__main__':
    D, L = load("iris.csv")
    plot_hist(D, L)
    DC = center_data(D)
    plot_hist(DC, L)
    plot_scatter(D, L)


