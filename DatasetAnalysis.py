from utils import *

# import training data
D, L = load_dataset()

# RAW DATA
plot_hist(D, L, prefix="centered")
plot_scatter(D, L, prefix="centered")
corr_map(D, "raw_whole_dataset", cmap="Greys")  # Check cmaps on matplot website
corr_map(D[:, L == 0], "raw_Male_class", cmap="Reds")  # Check cmaps on matplot website
corr_map(D[:, L == 1], "raw_Female_class", cmap="Blues")  # Check cmaps on matplot website

# Centered Data
D, _ = center_data(D)
plot_hist(D, L, prefix="centered")
plot_scatter(D, L, prefix="centered")
corr_map(D, "centered_whole_dataset", cmap="Greys")  # Check cmaps on matplot website
corr_map(D[:, L == 0], "centered_Male_class", cmap="Reds")  # Check cmaps on matplot website
corr_map(D[:, L == 1], "centered_Female_class", cmap="Blues")  # Check cmaps on matplot website


