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
data_raw = {}
for file in os.listdir(os.getcwd()):
    data_raw[file]=(pd.read_csv(file, sep=' ', names=["x_n1","x_n2","y_n"], index_col=False))

# Previewing the dataframes in the dictionaries
# for key, val in data_raw.items():
#     print(f'\n{key}\n{val.head()}')

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

#print(DetermineCandidateSplits(data_raw['D1.txt'],'x_n1'))
#print(sortData(data_raw['D1.txt'],'x_n1'))

# # Entropy working
# def entropy(tot, wins, loss):
#     # print(tot,wins,loss)
#     pwin = wins/tot
#     plos = loss/tot
#     if pwin == 0:
#         return (-1)*plos*np.log2(plos)
#     elif plos == 0:
#         return (-1)*(pwin)*np.log2(pwin)
#     else:
#         return (-1)*((pwin)*np.log2(pwin)+plos*np.log2(plos))
def entropy(splitDict):
    # assume that we are getting just one split (i.e. before or after the split)
    if type(splitDict) == dict:
        wins = splitDict[0]
        loss = splitDict[1]
    elif type(splitDict) == tuple:
        wins = splitDict[1][1]+splitDict[1][0]
        loss = splitDict[1][1]+splitDict[1][0]
    tot = wins+loss
    pwin = wins/tot
    plos = loss/tot
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

# print(infoGain(data_raw['D1.txt']))
# Entropy
# def InfoGain(Y):
#     total = Y[0][0]+Y[0][1]+Y[1][0]+Y[1][1]
#     totwins = Y[0][1]+Y[1][1]
#     totloss = Y[0][0]+Y[1][0]
#     beforeTotal = Y[0][0]+Y[0][1]
#     beforeWins = Y[0][1]
#     beforeLoss = Y[0][0]
#     afterTotal = Y[1][0]+Y[1][1]
#     afterWins = Y[1][1]
#     afterLoss = Y[1][0]
#     # en = np.sum((-1)*prob*np.log2(prob))
#     return infoGain(Y)

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
    return InfoGain(splitDict)/intrinsicEntropy(splitDict)

# Stopping criteria
# def stopping_criteria(candidate_splits, total, wins, loss):
#     if candidate_splits == [] or entropy(total, wins, loss) == 0 or all(GainRatio(total, wins, loss, at, aw, al, bt, bw, bl, (at + bt)) == 0 for at, aw, al, bt, bw, bl in candidate_splits):
#         return True
#     return False


def SplitCount(D, splitVal):
    Xi = list(D)[0]
    beforeCount={1:0,0:0}
    afterCount={1:0,0:0}
    for i, d in enumerate(D[Xi]):
        if d < splitVal:
            if D['y_n'][i] == 0:
                beforeCount[0]+=1
            if D['y_n'][i] == 1:
                beforeCount[1]+=1
        elif d >= splitVal:
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
    # SplitGain(data_raw['D1.txt'],'x_n1')
    dict_of_gains=SplitGain(D,C, Xi)
    #print(dict_of_gains)
    #print(GainRatio(list(dict_of_gains.items())[0][1]))
    # print(f'Key\t\tBefore:\t\tAfter:')
    # # for key,val in dict_of_gains.items():
    # #     print(key, val)
    # for key,val in dict_of_gains.items():
    #     print(key,GainRatio(val))


    max = 0
    lis_num = 0
    comp_max=0
    for x in range(len(dict_of_gains)):
        if GainRatio(list(dict_of_gains.items())[x][1]) > comp_max:
            #print(GainRatio(list(dict_of_gains.items())[0][1]))
            comp_max = GainRatio(list(dict_of_gains.items())[x][1])
            max = list(dict_of_gains.items())[x][0]
            c_num=x
        #print(list(dict_of_gains.items())[x][0],f'{x}',GainRatio(list(dict_of_gains.items())[x][1]))
    for row, val in enumerate(sortData(D, Xi)[Xi]):
        # print(row, val, sortData(D, Xi)['y_n'][row])
        if val == max:
            lis_num = row
            # print(row, val, '<====')
    #print(max, comp_max)
    return([lis_num, c_num, max, comp_max])

def listSplit(dict, ind):
    return(dict.iloc[:ind+1,:].reset_index(drop=True),dict.iloc[ind+1:,:].reset_index(drop=True))

# FindBestSplit(data_raw['D1.txt'])
# Create function for creating the decision tree
#   Send in data and export decision tree
def stoppingCriteria(D,C,Xi):
    dict_cand = SplitGain(D,C,Xi)
    ratioCount=0
    #get_row(D, C)
    #print(D)
    #print(D)
    if C == []:
        return True
    for candidate in dict_cand.items():
        if entropy(candidate[1]) == 0:
            return True
    for candidate in dict_cand.items():
        #print(len(SplitGain(D,C,Xi)))
        if GainRatio(candidate[1]) == 0 and entropy(candidate[1]) != 0:
            print(ratioCount)
            ratioCount+=1
        if ratioCount==len(dict_cand.items()):
            print("test")
            return True
    # for candidate in SplitGain(D,C,Xi).items():
    #     if all(GainRatio(candidate))


def MakeSubtree(D, Xi):
    C = DetermineCandidateSplits(D,Xi)
    #print(C)
    #print(C)
    # print(C)
    # if stopping_criteria(C)
    if stoppingCriteria(D, C, Xi):
        print(f'Make Leaf: {C}')
    #     make_leaf_node_N
    #     determine class_label-probabilites for N
    else:
    #     make internal node N
        print('Make Internal Node')
        S = FindBestSplit(D,C,Xi)
        #print(listSplit(sortData(D,Xi),S[0]))
        # print(S[1])
        # print(D[Xi].index(S[1]))
        for sub in listSplit(sortData(D,Xi),S[0]):
            Dk = sub
            Nchild = MakeSubtree(Dk, Xi)
            #print(Nchild)
    # return(subtree@N)

MakeSubtree(data_raw['D1.txt'], 'x_n1')
