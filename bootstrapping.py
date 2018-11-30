# Input: number of bootstraps B
#        numpy matrix X of features, with n rows (samples), d columns (features)
#        numpy vector y of scalar values, with n rows (samples), 1 column
# Output: numpy vector z of B rows, 1 column

import numpy as np
import linreg


def run(B, X, y):
    n = len(X)
    d = len(X[0])
    z = np.zeros((B, 1))
    for i in range(0, B):
        u = [0] * n
        S = []
        for j in range(0, n):
            k = np.random.randint(n)
            u[j] = k
            if not (k in S):
                S.append(k)
        Se = []
        for k in range(0, n):
            Se.append(k)
        T = list(set(Se) - set(S))
        thetahat = linreg.run(X[u, :], y[u, :])
        thetahat = thetahat.T
        thetahat = np.reshape(thetahat, (np.product(thetahat.shape),))
        sumi = 0
        for t in T:
            sumi = sumi + (y[t][0] - np.dot(thetahat, X[t])) ** 2
        z[i][0] = sumi / len(T)

    return z
