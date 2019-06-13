#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt # for output generalization error graphs
from myClass import Case, Theta # for test case and classifier's parameters
from functions import loss, grad_loss # for loss and gradient descent
from hyperParameters import hyperParameters # defines hyperParameters

# hyperParameters
N               = 10
T               = hyperParameters.T
D               = hyperParameters.D
alpha           = hyperParameters.alpha

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
]

# main
if __name__ == '__main__':
    num_of_cases = 0 # the index number of cases
    times_of_descent = 1000 # times of descent by gradient
    toolbar_width = 20 # for progress bar
    for case in testCase:
        num_of_cases += 1
        # progress bar (begin)
        sys.stdout.write("Case {}: ".format(num_of_cases))
        sys.stdout.write("[%s]" % (" " * toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width+1))

        loss_list = []
        theta = Theta(
        normal_form_matrix(0, 0.4, N, N),
        normal_form_matrix(0, 0.4, N, 1),
        0
        )
        for i in range(times_of_descent):
            loss_list.append(loss(case.adj_matrix, 1, theta))
            theta += grad_loss(case.adj_matrix, 1, theta)*(-alpha)

            # progress bar
            if i % (times_of_descent//toolbar_width) == 0:
                sys.stdout.write("#")
                sys.stdout.flush()

        # progress bar (end)
        sys.stdout.write("] Done. \n")

        plt.title("Case {}".format(num_of_cases))
        plt.xlabel("time")
        plt.ylabel("loss")
        plt.plot(range(times_of_descent), loss_list)
        plt.show()
