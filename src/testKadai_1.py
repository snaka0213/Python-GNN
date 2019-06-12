#!/usr/bin/env python3

import numpy as np
from myclass import Case
from kadai_1 import readout

# line_connected_matrix, the adjacency matrix of the following graph
#   (1) - (2) - ... - (n).
def line_connected_matrix(n):
    tmp = np.zeros((n,n), dtype = int)
    for i in range(n-1):
        tmp[i][i+1] = 1
    return tmp + tmp.T

# hyper parameters
N, D, T = 10, 2, 2

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
    if len(case.initial_rep.shape) == 1: # if D == 1:
        assert readout(case.adj_matrix, case.lea_matrix, case.initial_rep, T) == case.readout
    else: # if D => 2:
        assert all(readout(case.adj_matrix, case.lea_matrix, case.initial_rep, T) == case.readout)
    print("Case {}: OK".format(index))
