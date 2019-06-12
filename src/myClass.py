#!/usr/bin/env python3

import numpy as np

# True if m is number.
def is_number(m):
    return type(m) in (int, float, complex, np.float64)

# Case class
class Case():
    def __init__(self, adj_matrix, lea_matrix, initial_rep, readout):
        self.adj_matrix = adj_matrix # matrix A
        self.lea_matrix = lea_matrix # matrix W
        self.initial_rep = initial_rep # vectors x
        self.readout = readout

# Theta class
class Theta():
    def __init__(self, lea_matrix, para_a, para_b):
        self.lea_matrix = lea_matrix
        self.para_a = para_a
        self.para_b = para_b

    def __add__(self, other):
        return Theta(
            self.lea_matrix + other.lea_matrix,
            self.para_a + other.para_a,
            self.para_b + other.para_b
        )

    def __mul__(self, other):
        if is_number(other):
            return Theta(
                other*self.lea_matrix,
                other*self.para_a,
                other*self.para_b
            )
        else:
            raise TypeError("Right term must be number.")

    def __truediv__(self, other):
        return Theta(
            self.lea_matrix / other.lea_matrix,
            self.para_a / other.para_a,
            self.para_b / other.para_b
        )
