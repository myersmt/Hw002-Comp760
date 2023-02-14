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
import graphviz

# read in info
path_to_data = r'\Data'
os.chdir(os.getcwd()+path_to_data)
data_raw = {}
for file in os.listdir(os.getcwd()):
    data_raw[file]=(pd.read_csv(file, sep=' ', names=["x_n1","x_n2","y_n"], index_col=False))

# Organize the data
def sortData(D, Xi):
    return D[[Xi,'y_n']].sort_values(by=Xi).reset_index(drop=True)

# Determine splits
def DetermineCandidateSplits(D, Xi):
    C=[] # initialize set of candidate splits for feature Xi
    data_sort = sortData(D,Xi)
    # print(data_sort)
    for j in range(len(D)-1):
        if data_sort['y_n'][j] != data_sort['y_n'][j+1]:
            C.append(data_sort[Xi][j])
    return(C)

def entropy(splitDict):
    # assume that we are getting just one split (i.e. before or after the split)
    if type(splitDict) == dict:
        wins = splitDict[0]
        loss = splitDict[1]
    elif type(splitDict) == tuple:
        wins = splitDict[1][1]+splitDict[0][1]
        loss = splitDict[1][0]+splitDict[0][0]
    #print(splitDict)
    tot = wins+loss
    # if tot == 0:
    #     print('error')
    #     return(0)
    pwin = wins/tot
    plos = loss/tot
    #print(f'Total: {tot}, Wins: {wins}, Loss: {loss}')
    if pwin == 0:
        return (-1)*plos*np.log2(plos)
    elif plos == 0:
        return (-1)*(pwin)*np.log2(pwin)
    else:
        return (-1)*((pwin)*np.log2(pwin)+plos*np.log2(plos))

# # Joint Entropy
def InfoGain(splitDict):
    totals = {1:splitDict[0][1]+splitDict[1][1], 0:splitDict[0][0]+splitDict[1][0]}
    afterSplit = splitDict[1]
    beforeSplit = splitDict[0]
    beforeP = (beforeSplit[1]+beforeSplit[0])/(totals[1]+totals[0])
    afterP = (afterSplit[1]+afterSplit[0])/(totals[1]+totals[0])
    return entropy(totals)-(((afterP)*(entropy(afterSplit)))+((beforeP)*(entropy(beforeSplit))))

def intrinsicEntropy(Y):
    beforeTot = Y[0][1]+Y[0][0]
    afterTot = Y[1][1]+Y[1][0]
    tot = beforeTot+afterTot
    pbefore = beforeTot/tot
    pafter = afterTot/tot
    if pbefore == 0:
        return (-1)*pafter*np.log2(pafter)
    elif pafter == 0:
        return (-1)*(pbefore)*np.log2(pbefore)
    else:
        return (-1)*((pbefore)*np.log2(pbefore)+pafter*np.log2(pafter))

def GainRatio(splitDict):
    #print(splitDict)
    return InfoGain(splitDict)/intrinsicEntropy(splitDict)

def SplitCount(D, splitVal):
    Xi = list(D)[0]
    beforeCount={1:0,0:0}
    afterCount={1:0,0:0}
    for i, d in enumerate(D[Xi]):
        if d <= splitVal:
            if D['y_n'][i] == 0:
                beforeCount[0]+=1
            if D['y_n'][i] == 1:
                beforeCount[1]+=1
        elif d > splitVal:
            if D['y_n'][i] == 0:
                afterCount[1]+=1
            if D['y_n'][i] == 1:
                afterCount[0]+=1
    #print(beforeCount, afterCount)
    return beforeCount, afterCount


def SplitGain(D, C, Xi):
    gains = {}
    dsort = sortData(D,Xi)
    #print(dsort)
    for d in dsort[Xi]:
        if d in C:
            gains.update({d:SplitCount(D,d)})
    return(gains)


# find the best splits
def FindBestSplit(D,C, Xi):
    dict_of_gains=SplitGain(D,C, Xi)
    # print('best split candidates:',C, 'D:',dict_of_gains)
    max = 0
    lis_num = 0
    comp_max=0
    c_num=0
    for x in range(len(dict_of_gains)):
        #print(list(dict_of_gains.items())[x][1],entropy(list(dict_of_gains.items())[x][1]),GainRatio(list(dict_of_gains.items())[x][1]))
        if GainRatio(list(dict_of_gains.items())[x][1]) > comp_max:
            #print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
            #print(GainRatio(list(dict_of_gains.items())[0][1]))
            comp_max = GainRatio(list(dict_of_gains.items())[x][1])
            entropy_max = entropy(list(dict_of_gains.items())[x][1])
            info_gain = InfoGain(list(dict_of_gains.items())[x][1])
            max = list(dict_of_gains.items())[x][0]
            c_num=x
        #print(list(dict_of_gains.items())[x][0],f'{x}',GainRatio(list(dict_of_gains.items())[x][1]))
    for row, val in enumerate(sortData(D, Xi)[Xi]):
        # print(row, val, sortData(D, Xi)['y_n'][row])
        if val == max:
            lis_num = row
            # print(row, val, '<====')
    return([lis_num, c_num, max, comp_max, entropy_max, info_gain])

def listSplit(dict, ind):
    return(dict.iloc[:ind+1,:].reset_index(drop=True),dict.iloc[ind+1:,:].reset_index(drop=True))


# Create function for creating the decision tree
#   Send in data and export decision tree
def stoppingCriteria(D,C,Xi):
    dict_cand = SplitGain(D,C,Xi)
    ratioCount=0
    #get_row(D, C)
    if C == []:
        return True
    for candidate in dict_cand.items():
        if entropy(candidate[1]) == 0:
            #print("stop 2")
            return True
    for candidate in dict_cand.items():
        #print(candidate[1],entropy(candidate[1]))
        if GainRatio(candidate[1]) == 0 and entropy(candidate[1]) != 0:
            #print(ratioCount)
            ratioCount+=1
            #print("ratio")
        if ratioCount==len(dict_cand.items()):
            print("\ntest\n")
            return True
    else:
        return False
    # for candidate in SplitGain(D,C,Xi).items():
    #     if all(GainRatio(candidate))

class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None, gainRatio=None):


        # for the decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gainRatio = gainRatio


        # for leaf node
        self.value = value

def calcLeafValue(D):
    return(max(list(D), key=list(D).count))

class Tree():
    def __init__(self, min_splits=None, max_depth=None):
        # setting the root of the tree
        self.root = None

        # Stopping conditions
        self.min_splits = min_splits
        self.max_depth = max_depth

    def MakeSubtree(self, D, Xi):
        C = DetermineCandidateSplits(D,Xi)
        if stoppingCriteria(D, C, Xi):
            print(f'Make Leaf: {calcLeafValue(D["y_n"])}')
        #     make_leaf_node_N
        #     determine class_label-probabilites for N
        else:
            S = FindBestSplit(D,C,Xi)
            print(f'Make Internal Node: {S[2]}')
            for sub in listSplit(sortData(D,Xi),S[0]):
                #print(sub)
                Dk = sub
                Nchild = self.MakeSubtree(Dk, Xi)
                #print(Nchild)
        #return(Nchild)

Tree().MakeSubtree(data_raw['D1.txt'], 'x_n1')
# MakeSubtree(data_raw['D1.txt'], 'x_n1')
# print(data_raw['D1.txt'])
# for data, vals in data_raw['D1.txt'].items():
#     print(data,vals)
#     print(entropy(data))