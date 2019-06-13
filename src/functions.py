#!/usr/bin/env python3

import numpy as np
from myClass import Theta
from readout import readout
from hyperParameters import hyperParameters

# hyperParameters
T, epsilon = hyperParameters.T, hyperParameters.epsilon

# Let adj be the adjacency matrix of given graph G,
#  y be the label of G, W be a learnable matrix,
#  A \in \R^{D} and b \in \R be parameters of the classifier.

# function s
def func_s(adj, theta):
    W = theta.lea_matrix
    A = theta.para_a
    b = theta.para_b
    N, D = adj.shape[0], A.size
    x = np.zeros((N,D))
    for i in range(N):
        x[i][1] = 1

    return np.dot(A, readout(adj, W, x, T)) + b

# binary cross-entropy loss
def loss(adj, y, theta):
    s = func_s(adj, theta)
    if s < 700:
        return y*np.log(1+np.exp(-s)) + (1-y)*np.log(1+np.exp(s))
    else:
        return y*np.log(1+np.exp(-s)) + (1-y)*s

# matrix whose (i, j) element == 1
def delta_matrix(m, n, i, j):
    if n != 1:
        matrix = np.zeros((m,n))
        matrix[i][j] = 1
    else:
        matrix = np.zeros(m)
        matrix[i] = 1
    return matrix

# gradient of loss
def grad_loss(adj, y, theta):
    N, D = adj.shape[0], theta.para_a.size

    # numerical diff along lea_matrix (W)
    list_0 = []
    for j in range(D):
        for i in range(D):
            _theta = Theta(
                theta.lea_matrix + epsilon*delta_matrix(D, D, i, j),
                theta.para_a,
                theta.para_b
            )
            diff = (1/epsilon)*(loss(adj, y, _theta)-loss(adj, y, theta))
            list_0.append(diff)

    # numerical diff along para_a (A)
    list_1 = []
    for i in range(D):
        _theta = Theta(
            theta.lea_matrix,
            theta.para_a + epsilon*delta_matrix(D, 1, i, 1),
            theta.para_b
        )
        diff = (1/epsilon)*(loss(adj, y, _theta)-loss(adj, y, theta))
        list_1.append(diff)

    # numerical diff along para_b (b)
    _theta = Theta(
        theta.lea_matrix,
        theta.para_a,
        theta.para_b + epsilon
    )
    diff_b = (1/epsilon)*(loss(adj, y, _theta)-loss(adj, y, theta))

    # result
    return Theta(
        np.array(list_0).reshape((D,D)),
        np.array(list_1).T,
        diff_b
    )
