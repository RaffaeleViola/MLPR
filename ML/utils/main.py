
import numpy as np

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    x = np.array([[1, 3, 2, 5, 3, 1, 4, 9]]).reshape((4, 2))
    print(x)
    x_T = np.repeat(x, repeats=x.shape[0], axis=0)
    print(x_T)
    x_stacked = x
    for _ in range(x.shape[0] - 1):
        x_stacked = np.vstack((x_stacked, x))
    mapped = np.vstack(((x_stacked * x_T), x))
    print(mapped)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
