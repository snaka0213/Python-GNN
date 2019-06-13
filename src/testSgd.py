#!/usr/bin/env python3

import os, re, random, copy, time
import numpy as np
import matplotlib.pyplot as plt
from myclass import Theta
from functions import loss, func_s
from sgd import sgd, momentum_sgd

# directory (train files)
dir = os.getcwd() + '/train/'

# train graph_files
files = [file for file in os.listdir(dir) if re.search('_graph.txt', file)]
num_of_files = len(files)

# hyper parameters
T, D = 2, 8

train_size = num_of_files//2
valid_size = num_of_files - train_size

# normal_form_matrix
# This is the (m, n) matrix whose elements are Gaussian normal.
def normal_form_matrix(mu, sigma, m, n):
    if n == 1:
        return np.random.normal(loc = mu, scale = sigma, size = m).T
    else:
        return np.random.normal(loc = mu, scale = sigma, size = m*n).reshape((m,n))

# initialization
theta = Theta(
    normal_form_matrix(0, 0.4, D, D),
    normal_form_matrix(0, 0.4, D, 1),
    0
) # learnable parameters
w = Theta(
    np.zeros((D,D)),
    np.zeros(D).T,
    0
) # momentum

# classifier
def classifier(graph_file, theta):
    file = open(dir+graph_file)
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

# average loss
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

# validation
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

# main
loss_list_for_train, loss_list_for_valid = [], []
accuracy_list_for_train, accuracy_list_for_valid = [], []

def main():
    global theta, w
    global loss_list_for_train, loss_list_for_valid
    global accuracy_list_for_train, accuracy_list_for_valid

    # split the dataset to training dataset and validation dataset
    train_files = random.sample(files, train_size)
    valid_files = [x for x in files if x not in train_files]

    # sgd
    # momentum_sgd
    for i in range(num_of_epochs):
        tmp_train_files = copy.copy(train_files)

        #an epoch
        for j in range(train_size//batch_size):
            batch_files = random.sample(tmp_train_files, batch_size)
            for file in batch_files:
                tmp_train_files.remove(file)
            #theta = sgd(batch_files, theta)
            theta, w = momentum_sgd(batch_files, theta, w)

        print("Epoch {}: Done.".format(i+1))
        loss_list_for_train.append(avg_loss(train_files, theta))
        loss_list_for_valid.append(avg_loss(valid_files, theta))
        accuracy_list_for_train.append(avg_accuracy(train_files, theta))
        accuracy_list_for_valid.append(avg_accuracy(valid_files, theta))

main()
title = "SGD loss on train_data"
title = "M " + title
list = loss_list_for_train
plt.subplot(2,2,1)
plt.title(title)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(range(num_of_epochs), list)

title = "SGD loss on valid_data"
title = "M " + title
list = loss_list_for_valid
plt.subplot(2,2,2)
plt.title(title)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(range(num_of_epochs), list)

title = "Accuracy on train_data"
list = accuracy_list_for_train
plt.subplot(2,2,3)
plt.title(title)
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.ylim(0,1)
plt.plot(range(num_of_epochs), list)

title = "Accuracy on valid_data"
list = accuracy_list_for_valid
plt.subplot(2,2,4)
plt.title(title)
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.ylim(0,1)
plt.plot(range(num_of_epochs), list)

plt.show()
