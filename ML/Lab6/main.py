import numpy as np

sigma = 0.001


def load_data():
    lInf = []

    f = open('data/inferno.txt', encoding="ISO-8859-1")

    for line in f:
        lInf.append(line.strip())
    f.close()

    lPur = []

    f = open('data/purgatorio.txt', encoding="ISO-8859-1")

    for line in f:
        lPur.append(line.strip())
    f.close()

    lPar = []

    f = open('data/paradiso.txt', encoding="ISO-8859-1")

    for line in f:
        lPar.append(line.strip())
    f.close()

    return lInf, lPur, lPar


def split_data(l, n):
    lTrain, lTest = [], []
    for i in range(len(l)):
        if i % n == 0:
            lTest.append(l[i])
        else:
            lTrain.append(l[i])
    return lTrain, lTest


def compute_frequency(w_list):
    freq = dict()
    count_list = []
    i = 0
    for tercet in w_list:
        for word in tercet.split(" "):
            index = freq.setdefault(word, i)
            if index == i:
                count_list.append(1 + sigma)
                i += 1
            else:
                count_list[index] += 1
    Nc = np.array(count_list).sum()
    freq_list = [val/Nc for val in count_list]
    return {key: freq_list[val] for key, val in freq.items()}


def likelihood(dataset, c1, c2, c3):
    score_matrix = []
    for tercet in dataset:
        score_line = [0]*3
        for word in tercet.split(" "):
            score_line[0] += np.log(c1.get(word, 1))
            score_line[1] += np.log(c2.get(word, 1))
            score_line[2] += np.log(c3.get(word, 1))
        score_matrix.append(score_line)
    return np.exp(np.array(score_matrix).T)



if __name__ == '__main__':
    # Load the tercets and split the lists in training and test lists

    lInf, lPur, lPar = load_data()
    lInf_train, lInf_evaluation = split_data(lInf, 4)
    lPur_train, lPur_evaluation = split_data(lPur, 4)
    lPar_train, lPar_evaluation = split_data(lPar, 4)

    inf_freq = compute_frequency(lInf_train)
    pur_freq = compute_frequency(lPur_train)
    par_freq = compute_frequency(lPar_train)

    score_matrix = likelihood(lInf_evaluation, inf_freq, pur_freq, par_freq)
    score_matrix /= 3
    posterior = score_matrix / score_matrix.sum(0)
    prediction = np.argmax(posterior, axis=0)
    res = prediction - 0
    corr = 0
    for i in range(res.shape[0]):
        if res[i] == 0:
            corr += 1
    print(corr / res.shape[0])
