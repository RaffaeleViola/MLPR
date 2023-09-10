import numpy as np
import matplotlib.pyplot as plt

N_ATTR = 4
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
    label_names = ["Male", "Female"]
    mask0 = (labels == 0.0)
    d0 = dataset[:, mask0]
    mask1 = (labels == 1.0)
    d1 = dataset[:, mask1]
    for attr in range(dataset.shape[0]):
        plt.hist(d0[attr, :], bins=20, density=True, ec='black', color='Blue', alpha=0.5)
        plt.hist(d1[attr, :], bins=20, density=True, ec='black', color='Red', alpha=0.5)
        plt.legend(label_names)
        plt.title(f'Feature no. {attr}')
        plt.show()


def plot_scatter(dataset, labels):
    label_names = ["Male", "Female"]
    mask0 = (labels == 0.0)
    d0 = dataset[:, mask0]
    mask1 = (labels == 1.0)
    d1 = dataset[:, mask1]
    mask2 = (labels == 2.0)
    d2 = dataset[:, mask2]
    plt.figure()
    for i in range(N_ATTR):
        for j in range(N_ATTR):
            if i == j:
                continue
            plt.scatter(d0[i, :], d0[j, :])
            plt.scatter(d1[i, :], d1[j, :])
            plt.scatter(d2[i, :], d2[j, :])
            plt.xlabel(f'Feature no. {i}')
            plt.ylabel(f'Feature no. {j}')
            plt.legend(label_names)
            plt.show()



def center_data(dataset):
    mu = dataset.mean(1)
    centered_dataset = dataset - mu.reshape((dataset.shape[0], 1))
    return centered_dataset


if __name__ == '__main__':
    D, L = load("iris.csv")
    # plot_hist(D, L)
    DC = center_data(D)
    # plot_hist(DC, L)
    plot_scatter(D, L)


