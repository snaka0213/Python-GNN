#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from myclass import Case, Theta
from functions import loss, grad_loss

# line_connected_matrix
def line_connected_matrix(n):
    tmp = np.zeros((n,n), dtype = int)
    for i in range(n-1):
        tmp[i][i+1] = 1
    return tmp + tmp.T

# normal_form_matrix
# This is the (m, n) matrix whose elements are Gaussian normal.
def normal_form_matrix(mu, sigma, m, n):
    if n == 1:
        return np.random.normal(loc = mu, scale = sigma, size = m).T
    else:
        return np.random.normal(loc = mu, scale = sigma, size = m*n).reshape((m,n))

# testCase
testCase = [
Case(
    np.zeros((N,N), dtype = int),
    np.identity(D, dtype = int),
    np.arange(N*D).reshape((N,D)),
    0
    ),
Case(
    np.ones((N,N), dtype = int) - np.identity(N, dtype = int),
    np.identity(D, dtype = int),
    np.arange(N*D).reshape((N,D)),
    (N-1)**T*np.sum(np.arange(N*D).reshape((N,D)), axis=0).T
    ),
Case(
    line_connected_matrix(N),
    np.identity(D, dtype = int),
    np.arange(N*D).reshape((N,D)),
    4*np.sum(np.arange(N*D).reshape((N,D)), axis=0).T - np.array([6*(N-1),6*N]).T
    ),
]

# main
index = 0
for case in testCase:
    index += 1
    loss_list = []
    theta = Theta(
        normal_form_matrix(0, 0.4, N, N),
        normal_form_matrix(0, 0.4, N, 1),
        0
    )
    for i in range(10000):
        loss_list.append(loss(case.adj_matrix, 1, theta))
        theta += grad_loss(case.adj_matrix, 1, theta)*(-alpha)
    print("##########Case {}: Done.##########".format(index))
    plt.title("Case {}".format(index))
    plt.xlabel("time")
    plt.ylabel("loss")
    plt.plot(range(10000), loss_list)
    plt.show()
