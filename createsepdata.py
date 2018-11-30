import numpy as np
import scipy.linalg as la


# Input: number of samples n
#        number of features d
# Output: numpy matrix X of features, with n rows (samples), d columns (features)
# X[i,j] is the j-th feature of the i-th sample
# numpy vector y of labels, with n rows (samples), 1 column
# y[i] is the label (+1 or -1) of the i-th sample


# Example on how to call the script:
#     import creat  esepdata
#     X, y = createsepdata.run(10,3)
def run(n, d):
    y = np.ones((n, 1))
    y[n / 2:] = -1
    X = np.random.random((n, d))
    idx_row, idx_col = np.where(y == 1)
    X[idx_row, 0] = 0.1 + X[idx_row, 0]
    idx_row, idx_col = np.where(y == -1)
    X[idx_row, 0] = -0.1 - X[idx_row, 0]
    U = la.orth(np.random.random((d, d)))
    X = np.dot(X, U)
    return X, y
