import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join as pjoin

# Load data
data_dir = "../data/corrupted_mnist"
X = np.load(pjoin(data_dir, "foreground.npy"))
Y = np.load(pjoin(data_dir, "background.npy"))
X_labels = np.load(pjoin(data_dir, "foreground_labels.npy"))
# digits_test = np.load(pjoin(data_dir, "mnist_digits_test.npy"))

X_mean, Y_mean = np.mean(X, axis=0), np.mean(Y, axis=0)
X = (X - X_mean) / np.std(X, axis=0)
Y = (Y - Y_mean) / np.std(Y, axis=0)

# digits_test_mean = np.mean(digits_test, axis=0)
# digits_test = (digits_test - digits_test_mean) / np.std(digits_test, axis=0)

X, Y = X.T, Y.T
# digits_test = digits_test.T
n = X.shape[1]
m = Y.shape[1]

import ipdb; ipdb.set_trace()