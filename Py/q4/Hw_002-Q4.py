"""
Created by: Matt Myers
02/15/2023

Q4
"""
import sklearn
from sklearn import tree
import numpy as np
import pandas as pd
import os
import graphviz
from matplotlib import pyplot as plt
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial

#Functions
def error(set):
    return sum(np.abs((poly(set)-np.sin(set))))/len(set)

# Main code
# Defining consatnts
step_num= 25
min = -np.pi
max = np.pi

# # Setting up the training data
# train_set = np.array(sorted(np.random.uniform(min, max, step_num)))
# train_y = np.sin(train_set)

# # Lagrangian 
# poly = lagrange(train_set, train_y)

# # Test set
# test_set = np.array(sorted(np.random.uniform(min, max, step_num)))

# # Calculating error
# train_error = error(train_set)
# test_error = error(test_set)
# #print(train_error, test_error)

# # plotting for no Epsilon
# plt.scatter(train_set,poly(train_set),label='Train')
# plt.scatter(test_set, poly(test_set), label='Test: Polynomial')
# plt.plot(train_set, train_y, label=r"Sin(X)", linestyle='-.')
# plt.legend()
# plt.axis([-2,2,-2,2])
# plt.show()

ep_min = 0.001
ep_max = 1
steps = 20
true_x = np.arange(min,max,(max-min)/steps)
true_y = np.sin(true_x)

lis_of_epsilon = [0, 0.001, 0.01, 0.1, 1]
for epsilon in lis_of_epsilon:
    adjustment = np.random.normal(0, epsilon, steps)
    #print(adjustment)
    
    # Setting up the training data
    train_set = np.array(sorted(np.random.uniform(min, max, steps)))
    train_y = np.sin(train_set)+adjustment

    # Lagrangian 
    poly = lagrange(train_set, train_y)

    # Test set
    test_set = np.array(sorted(np.random.uniform(min, max, steps)))

    # Calculating error
    train_error = error(train_set)
    test_error = error(test_set)
    print(f'Train Error: {train_error}\nTest Error: {test_error}')
    
    # Plotting
    plt.scatter(train_set,poly(train_set),label='Train')
    plt.scatter(test_set, poly(test_set), label='Test: Polynomial')
    plt.plot(true_x, true_y, label=r"Sin(X)", linestyle='-.')
    plt.legend()
    plt.title(f'{epsilon}')
    plt.axis([min,max,min,max])
    plt.show()
