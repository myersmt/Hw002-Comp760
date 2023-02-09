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

