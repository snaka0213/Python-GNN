#!/usr/bin/env python3

import os, re, random
import numpy as np
from myclass import Theta
from functions import grad_loss

# directory
dir = os.getcwd() + '/train/'

# train files
files = [file for file in os.listdir(dir) if re.search('_graph.txt', file)]
num_of_files = len(files)

# hyper parameters
T, D = 2, 8
alpha = 0.001
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-8
epsilon_hat = epsilon*np.sqrt(1-beta_2)

# unit theta
theta_ones = Theta(
    np.ones((D,D)),
    np.ones(D),
    1
)

def elementwise_square(theta):
    return Theta(
        np.vectorize(lambda x: x**2)(theta.lea_matrix),
        np.vectorize(lambda x: x**2)(theta.para_a),
        theta.para_b**2
    )

def elementwise_sqrt(theta):
    return Theta(
        np.sqrt(theta.lea_matrix),
        np.sqrt(theta.para_a),
        np.sqrt(theta.para_b)
    )

# adam (https://arxiv.org/pdf/1412.6980.pdf)
def adam(b_files, theta, moment_1, moment_2):
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

    gradient = tmp_theta*(1/batch_size)
    moment_1 = moment_1*beta_1 + gradient*(1-beta_1)
    moment_2 = moment_2*beta_2 + elementwise_square(gradient)*(1-beta_2)
    theta = theta + moment_1/(elementwise_sqrt(moment_2)+theta_ones*epsilon)*(-alpha*np.sqrt((1-beta_2))/(1-beta_1))
    return (theta, moment_1, moment_2)
