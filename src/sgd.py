#!/usr/bin/env python3

import os, re, random
import numpy as np
from myClass import Theta
from functions import grad_loss
from hyperParameters import hyperParameters

# hyperParameters
T, D = hyperParameters.T, hyperParameters.D
alpha, epsilon = hyperParameters.alpha, hyperParameters.epsilon
moment = hyperParameters.moment

# directory (train files)
dir = os.getcwd() + '/train/'

# train graph_files
files = [file for file in os.listdir(dir) if re.search('_graph.txt', file)]
num_of_files = len(files)

# stochastic gradient descent (sgd)
# Let b_files be a list of file_name of batchs, theta be learnable parameters.
def sgd(b_files, theta):
    batch_size = len(b_files)
    tmp_theta = Theta(
        np.zeros((D,D)),
        np.zeros(D).T,
        0
    )
    for graph_file in b_files:
        label_file = graph_file.rstrip('_graph.txt') + '_label.txt'

        file = open(dir+graph_file)
        N, adj = int(file.readline()), []
        for i in range(N):
            adj.append([int(x) for x in file.readline().split()])
        adj = np.array(adj)
        file.close()

        file = open(dir+label_file)
        y = int(file.readline())
        file.close()

        tmp_theta += grad_loss(adj, y, theta)

    delta_theta = tmp_theta*(1/batch_size)
    return theta + delta_theta*(-alpha)

# momentum stochastic gradient descent (momentum_sgd)
def momentum_sgd(b_files, theta, w):
    batch_size = len(b_files)
    tmp_theta = Theta(
        np.zeros((D,D)),
        np.zeros(D).T,
        0
    )
    for graph_file in b_files:
        label_file = graph_file.rstrip('_graph.txt') + '_label.txt'

        file = open(dir+graph_file)
        N, adj = int(file.readline()), []
        for i in range(N):
            adj.append([int(x) for x in file.readline().split()])
        adj = np.array(adj)
        file.close()

        file = open(dir+label_file)
        y = int(file.readline())
        file.close()

        tmp_theta += grad_loss(adj, y, theta)

    delta_theta = tmp_theta*(1/batch_size)
    theta += delta_theta*(-alpha) + w*moment
    w += delta_theta*(-alpha) + w*moment
    return (theta, w)
