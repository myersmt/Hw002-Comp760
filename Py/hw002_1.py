"""
Creator: Matt Myers
Due Date: 02/15/2023
Class:  Comp Sci 760

Question 001:

Topic: Implement a decision-tree learner for classification

Assume:
    * Each item has two continous features x in R^2
    * The class label is binary and encoded as y in {0,1}
    * Data files are in plain text with one labeled item per line, seperated by whitespace

Goal:
    * Candidate splits (j, c) for numeric features should use a threshold c in feature dimension j in the form of x_j >= c.
    * c should be on values of that dimension present in the training data; i.e. the threshold is on training points,
        not in between training points. You may enumerate all features, and for each feature, use all possible values
        for that dimension.
    * You may skip those candidate splits with zero split information (i.e. the entropy of the split), and continue the enumeration.
    * The left branch of such a split is the “then” branch, and the right branch is “else”.
    * Splits should be chosen using information gain ratio. If there is a tie you may break it arbitrarily.
    * The stopping criteria (for making a node into a leaf) are that
        - the node is empty, or
        - all splits have zero gain ratio (if the entropy of the split is non-zero), or
        - the entropy of any candidates split is zero
    * To simplify, whenever there is no majority class in a leaf, let it predict y = 1.
"""
import numpy as np
import pandas as pd
import os

# read in info
path_to_data = r'\Data'
os.chdir(os.getcwd()+path_to_data)
# print(os.listdir(os.getcwd()+path_to_data))
data_raw = {}
for file in os.listdir(os.getcwd()):
    #print(file)
    data_raw[file]=(pd.read_csv(file, sep=' ', names=["x_n1","x_n2","y_n"], index_col=False))
    #data_raw[file]=data_raw[file].reset_index(drop=True)

# Previewing the dataframes in the dictionaries
# for key, val in data_raw.items():
#     print(f'\n{key}\n{val.head()}')

# Organize the data

# Determine splits
def DetermineCandidateSplits(D, Xi):
    C=[] # initialize set of candidate splits for feature Xi
    data_sort = D[[Xi, 'y_n']].sort_values(by=Xi).reset_index(drop=True)
    # print(data_sort)
    for j in range(len(D)-1):
        if data_sort['y_n'][j] != data_sort['y_n'][j+1]:
            C.append(D[Xi][j])
    return(C)

print(len(DetermineCandidateSplits(data_raw['D1.txt'],'x_n2')))

# find the best splits
def FindBestSplits(D,C):
    pass

# Create function for creating the decision tree
#   Send in data and export decision tree
def MakeSubtree(D):
    # C = DetermineCandidateSplits(D)
    # if node-empty or zero-gain-ratio or entropy-zero:
    #     make_leaf_node_N
    #     determine class_label-probabilites for N
    # else:
    #     make internal node N
    #     S = FindBestSplits(D,C)
    #     for outcomeK in S:
    #         Dk = subset of instances that have outcome k
    #         kth child of N = MakeSubtree(Dk)
    # return(subtree@N)
    pass