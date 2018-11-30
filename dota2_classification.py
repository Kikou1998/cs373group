import numpy as np
import csv
from sklearn.svm import SVC  # "Support vector classifier"


def bootstrapping(B, X, y, C_p, gamma_p, K):
    # B: number of times to do bootstrapping
    # C_p: slack variable of SVM
    # gamma_p: gamma of SVM
    # K: kernel of SVM

    n = len(X)
    d = len(X[0])
    z = np.zeros((B, 1))
    bs_err = np.zeros(B)
    for i in range(0, B):
        u = [0] * n
        S = []  # sampling
        for j in range(0, n):
            k = np.random.randint(n)
            u[j] = k
            if not (k in S):
                S.append(k)
        Se = []
        for k in range(0, n):
            Se.append(k)
        T = list(set(Se) - set(S))  # testing
        alg = SVC(C=C_p, kernel=K, gamma=gamma_p)
        alg.fit(X[S], y[S])
        bs_err[i] = np.mean(y[T] != alg.predict(X[T]))
    err = np.mean(bs_err)
    return err


reader = csv.reader(open("dota2.csv", "rb"), delimiter=",")
fbuffer = list(reader)
x = np.array(fbuffer).astype("int")
x = x[0:2000]
y = x[:, 0]
y = y.ravel()
x = np.delete(x, 0, 1)  # delete 1st column of x
x = np.delete(x, 0, 1)  # delete 1st column of x
x = np.delete(x, 0, 1)  # delete 1st column of x
x = np.delete(x, 0, 1)  # delete 1st column of x

# we are not interested in game mode, cluster, etc for now and instead just focusing on the heros picked each game, so the first column indicating
# the result of the game plus the following three columns which are about general game info are removed.


# alg = SVC(kernel='rbf', C=1, gamma=0.1)
# alg.fit(x, y)
# y_pred = alg.predict(x)
# err = np.mean(y != y_pred)

# print "err:"
# print err
positive_y = []
negative_y = []
for i in range(0, 2000):
    if y[i] == 1:
        positive_y.append(i)
    else:
        negative_y.append(i)

positive_len = len(positive_y)
negative_len = len(negative_y)

print "positive:"
print positive_len
print "negative:"
print negative_len

samples_fold1 = positive_y[0:positive_len / 2] + negative_y[0:negative_len / 2]
samples_fold2 = positive_y[positive_len / 2:] + negative_y[negative_len / 2:]

C_list = [0.1, 1, 10]
B = 30
gamma_list = [1, 10, 100]
# first have a fixed c of 1 , do hyperparameter tuning for gamma
best_gamma = 0
best_error = 1.1
for gamma in gamma_list:
    err = bootstrapping(B, x[samples_fold1], y[samples_fold1], 100, gamma, "rbf")
    print "gamma =", gamma, ",error = ", err
    if err < best_error:
        best_error = err
        best_gamma = gamma
best_C = 0
best_error = 1.1
# best gamma obtained, do hyperparameter tuning for C based on this gama
for slack in C_list:
    err = bootstrapping(B, x[samples_fold1], y[samples_fold1], slack, best_gamma, "rbf")
    print "C =", slack, ",error = ", err
    if err < best_error:
        best_error = err
        best_C = slack
print "Results of hyperparameter tuning:"
print "C: ", best_C
print "gamma: ", best_gamma

# y_pred = np.zeros(len(x), int)
alg = SVC(C=best_C, kernel="rbf", gamma=best_gamma)
alg.fit(x[samples_fold2], y[samples_fold2])
y_pred = alg.predict(x[samples_fold1])
# y_pred is the list of predicted label of x in samples_fold1
err = np.mean(y[samples_fold1] != y_pred)

print "error: ", err
