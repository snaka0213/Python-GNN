#!/usr/bin/env python3

import numpy as np

# ReLU function
# Let M be a N \times D matrix.
def ReLU(M):
    return np.vectorize(lambda x: max(0, x))(M)

# Aggregate function
# Let adj be the adjacency matrix of given graph G,
#  and x be the representative vectors (N \times D matrix).
def aggregate_1(adj, x):
    return np.dot(adj, x)

# Let lea be a learnable matrix.
def aggregate_2(adj, lea, x):
    tmp_list = []
    for a in aggregate_1(adj, x):
        tmp_list.append(ReLU(np.dot(lea, a.T)).T)
    return np.array(tmp_list)

# Let x be the initial representative vectors,
#  and T be the number of steps for neural network.
def readout(adj, lea, x, T):
    tmp = x
    for i in range(T):
        tmp  = aggregate_2(adj, lea, tmp)
    return np.sum(tmp, axis=0).T
