from utils import *


def datasetAnalysis():
    # import training data
    D, L = load_dataset("Train.txt")

    # RAW DATA
    plot_hist(D, L, prefix="centered")
    plot_scatter(D, L, prefix="centered")
    corr_map(D, "raw_whole_dataset", cmap="Greys")  # Check cmaps on matplot website
    corr_map(D[:, L == 0], "raw_Male_class", cmap="Reds")  # Check cmaps on matplot website
    corr_map(D[:, L == 1], "raw_Female_class", cmap="Blues")  # Check cmaps on matplot website

    # Explained Variance
    m_list = np.array(range(D.shape[0]))
    PCA_reducer = PCA()
    PCA_reducer.fit(D)
    cumsum = PCA_reducer.explained_variance()
    fig = plt.figure()
    plt.plot([m + 1 for m in m_list], cumsum,  color='red', marker='o', linestyle='dashed', linewidth=2, markersize=6)
    plt.xticks([m + 1 for m in m_list])
    plt.xlabel("PCA Dimensions m")
    plt.ylabel("Fraction of Explained Variance")
    plt.suptitle("PCA -- Explained Variance")
    plt.grid()
    plt.savefig("Images/correlation/explained_variance.png")
    plt.close(fig)

    #LDA
    LDA_reducer = LDA(2)
    LDA_reducer.fit(D, L)
    D_red = LDA_reducer.transform(1, D)
    plot_hist(D_red, L, "LDA_discrimation")


    # Centered Data
    D_C, _ = center_data(D)
    plot_hist(D_C, L, prefix="centered")
    plot_scatter(D_C, L, prefix="centered")
    corr_map(D_C, "centered_whole_dataset", cmap="Greys")  # Check cmaps on matplot website
    corr_map(D_C[:, L == 0], "centered_Male_class", cmap="Reds")  # Check cmaps on matplot website
    corr_map(D_C[:, L == 1], "centered_Female_class", cmap="Blues")  # Check cmaps on matplot website


