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

step_num= 50
min = -np.pi
max = np.pi

train_set = np.array(sorted(np.random.uniform(min, max, step_num)))
train_y = np.sin(train_set)
poly = lagrange(train_set, train_y)

test_set = np.array(sorted(np.random.uniform(min, max, step_num)))
plt.scatter(train_set,poly(train_set),label='Train')
plt.scatter(test_set, poly(test_set), label='Test: Polynomial')
plt.plot(train_set, train_y, label=r"Sin(X)", linestyle='-.')
plt.legend()
plt.axis([-2,2,-2,2])
plt.show()

def error(set):
    return sum(np.abs((poly(set)-np.sin(set))))/len(set)

train_error = error(train_set)
test_error = error(test_set)
print(train_error, test_error)

# plt.plot(x,fun_y)
# plt.show()
