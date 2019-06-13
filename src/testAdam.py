import os, re, random, copy, time
import numpy as np
import matplotlib.pyplot as plt
from myclass import Theta
from kadai_2 import func_s
from kadai_4 import adam

#directory
dir = os.getcwd() + '/train/'

#train files
files = [file for file in os.listdir(dir) if re.search('_graph.txt', file)]
num_of_files = len(files)

#hyper parameters
T, D = 2, 8
batch_size = 500
train_size = num_of_files
num_of_epochs = 10

#normal_form_matrix
#This is the (m, n) matrix whose elements are Gaussian normal.
def normal_form_matrix(mu, sigma, m, n):
    if n == 1:
        return np.random.normal(loc = mu, scale = sigma, size = m).T
    else:
        return np.random.normal(loc = mu, scale = sigma, size = m*n).reshape((m,n))

#initialization
theta = Theta(
    normal_form_matrix(0, 0.4, D, D),
    normal_form_matrix(0, 0.4, D, 1),
    0
)
moment_1 = Theta(
    np.zeros((D,D)),
    np.zeros(D).T,
    0
)
moment_2 = Theta(
    np.zeros((D,D)),
    np.zeros(D).T,
    0
)

#classifier
def classifier(graph_file, theta):
    file = open(graph_file)
    N, adj = int(file.readline()), []
    for i in range(N):
        adj.append([int(x) for x in file.readline().split()])
    adj = np.array(adj)
    file.close()

    s = func_s(adj, theta)
    p = 1/(1+np.exp(-s))

    if p > 1/2:
        return 1
    else:
        return 0

#average loss
def avg_loss(b_files, theta):
    tmp_loss = 0
    batch_size = len(b_files)
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

        tmp_loss += loss(adj, y, theta)

    return tmp_loss/batch_size

#validation
def avg_accuracy(v_files, theta):
    hit_counter = 0
    for graph_file in v_files:
        label_file = graph_file.rstrip('_graph.txt') + '_label.txt'

        file = open(dir+label_file)
        y = int(file.readline())
        file.close()

        if classifier(graph_file, theta) == y:
            hit_counter += 1

    return hit_counter/len(v_files)

#main
for i in range(num_of_epochs):
    tmp_train_files = copy.copy(files)

    #an epoch
    for j in range(train_size//batch_size):
        batch_files = random.sample(tmp_train_files, batch_size)
        for file in batch_files:
            tmp_train_files.remove(file)
        theta, moment_1, moment_2 = adam(batch_files, theta, moment_1, moment_2)

    print("Epoch {}: Done.".format(i+1))

dir = os.getcwd() + '/test/'
tests = [file for file in os.listdir(dir) if re.search('_graph.txt', file)]
num_of_tests = len(tests)

file = open("prediction.txt", 'w')
for i in range(num_of_tests):
    graph_file = dir+'{}_graph.txt'.format(i)
    file.write("{}\n".format(classifier(graph_file, theta)))

file.close()
