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
    #print(file)
    data_raw[file]=(pd.read_csv(file, sep=' ', names=["x_n1","x_n2","y_n"], index_col=False))
    #print(file)

# Organize the data
def sortData(D, Xi):
    #print(D,Xi)
    #print(D, Xi, D[[Xi,'y_n']].sort_values(by=Xi),"done")
    return D.sort_values(by=Xi).reset_index(drop=True)

# Determine splits
def DetermineCandidateSplits(D, Xi):
    C={} # initialize set of candidate splits for feature Xi
    count = 0 
    for feature in Xi:
        count+=1
        lis = []
        #print(feature, Xi)
        #print(D, feature, Xi)
        data_sort = sortData(D,feature)
        # print(data_sort)
        for j in range(len(D)-1):
            if data_sort['y_n'][j] != data_sort['y_n'][j+1]:
                lis.append(j)
        C[feature]=lis
        #print(feature,':',C[feature])
    return(C)

def entropy(splitDict):
    # assume that we are getting just one split (i.e. before or after the split)
    if type(splitDict) == dict:
        wins = splitDict[1]
        loss = splitDict[0]
    elif type(splitDict) == tuple:
        wins = splitDict[1][1]+splitDict[0][1]
        loss = splitDict[1][0]+splitDict[0][0]
    #print(splitDict, wins, loss)
    tot = wins+loss
    # if tot == 0:
    #     print(splitDict)
    #     print('error')
    #     return 1
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
    #print(splitDict)
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

def SplitCount(D, splitInd):
    Xi = list(D)[0]
    #print('Split index:',splitInd)
    beforeCount={1:0,0:0}
    afterCount={1:0,0:0}
    for i, d in enumerate(D[Xi]):
        if i <= splitInd:
            if D['y_n'][i] == 0:
                beforeCount[0]+=1
            if D['y_n'][i] == 1:
                beforeCount[1]+=1
        elif i > splitInd:
            if D['y_n'][i] == 0:
                afterCount[0]+=1
            if D['y_n'][i] == 1:
                afterCount[1]+=1
    return beforeCount, afterCount


def SplitGain(D, C, Xi):
    dic = {}
    for feature in Xi:
        dsort = sortData(D,feature)
        gains = []
        for ind, val in enumerate(dsort[feature]):
            if ind in C[feature]:
                gains.append([val,SplitCount(D,ind)])
        dic[feature] = gains
    #print(dic)
    return(dic)

def SplitGainFull(D, C, Xi):
    dic = {}
    for feature in Xi:
        dsort = sortData(D,feature)
        #print(dsort)
        gains = []
        for ind, val in enumerate(dsort[feature]):
            gains.append([val,SplitCount(D,ind)])
        dic[feature] = gains
    #print(dic)
    return(dic)

# find the best splits
def FindBestSplit(D,C, Xi):
    #print('Beginning Find best split:')
    dict_of_gains=SplitGain(D,C,Xi)
    full_dict = SplitGainFull(D,C,Xi)
    #print(D)
    #print(dict_of_gains)
    for feature in Xi:
        # for x in range(len(full_dict[feature])):
        #     print(x, full_dict[feature][x],GainRatio(list(full_dict[feature][x])[1]))
        max = 0
        lis_num = 0
        comp_max=0
        entropy_max=0
        c_num=0
        #print(len(dict_of_gains))
        #print(D[[feature,'y_n']])
        # for c in range(len(D)):
        #     print(list(D[[feature, 'y_n']])[c])
        #     print(x,list(D[feature])[x][1],list(D[feature])[x],GainRatio(list(D[feature])[x][1]))
        for x in range(len(dict_of_gains[feature])):
            #print(list(dict_of_gains.items())[x][1],entropy(list(dict_of_gains.items())[x][1]),GainRatio(list(dict_of_gains.items())[x][1]))
            #print(x,list(dict_of_gains[feature])[x][1],list(dict_of_gains[feature])[x],GainRatio(list(dict_of_gains[feature])[x][1]))
            print('\\\\Feature:',feature,'\\\\\n\t\\quad Split Value:',list(dict_of_gains[feature])[x][0],'\\\\\n\t\\quad Gain Ratio:',GainRatio(list(dict_of_gains[feature])[x][1]),'\\\\\n\t\\quad InfoGain:',InfoGain(list(dict_of_gains[feature])[x][1]),'\\\\\n\t\\quad Entropy:',intrinsicEntropy(list(dict_of_gains[feature])[x][1]), '\\\\')
            if GainRatio(list(dict_of_gains[feature])[x][1]) >= comp_max :
                #print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
                #print(GainRatio(list(dict_of_gains.items())[0][1]))
                best_feature = feature
                comp_max = GainRatio(list(dict_of_gains[feature])[x][1])
                entropy_max = intrinsicEntropy(list(dict_of_gains[feature])[x][1])
                info_gain = InfoGain(list(dict_of_gains[feature])[x][1])
                max = list(dict_of_gains[feature])[x][0]
                #maxNEXT = list(dict_of_gains[feature])[x+1][0]
                c_num=x
            #print(list(dict_of_gains.items())[x][0],f'{x}',GainRatio(list(dict_of_gains.items())[x][1]))
        #print(enumerate(sortData(D[[feature]],feature)))
        nextC = False
        for index, row in sortData(D,feature).iterrows():
            if row[feature] == max:
                lis_num = index
                nextC = True
            elif nextC:
                maxNEXT = row[feature]
                nextC = False
                # print(row, val, '<====')
    print([lis_num, c_num, max, comp_max, entropy_max, info_gain, best_feature])
    return([lis_num, c_num, max, comp_max, entropy_max, info_gain, best_feature], maxNEXT)

def listSplit(dict, ind):
    return(dict.iloc[:ind+1,:].reset_index(drop=True),dict.iloc[ind+1:,:].reset_index(drop=True))

def Zeros(lis):
    for val in lis:
        if val != 0:
            return False
    return True

# Create function for creating the decision tree
#   Send in data and export decision tree
def stoppingCriteria(D,Xi):
    #print('Stop C starting:')
    C = DetermineCandidateSplits(D, Xi)
    dict_cand = SplitGain(D,C,Xi)
    #print(dict_cand)

    for feature in Xi:
        gain_list = []
        #print(C[feature])
        if C[feature] == []:
            print('stop 1')
            return True
        for dic in dict_cand[feature]:
            gain_list.append(GainRatio(dic[1]))
        #print(gain_list)
        for candidate in dict_cand[feature]:
            print('stop 2')
            #print(candidate,feature, candidate[1], entropy(candidate[1]))
            if intrinsicEntropy(candidate[1]) == 0:
                #print(candidate[1], entropy(candidate[1]), "stop 2")
                return True
            if Zeros(gain_list):
                print('stop 3')
                return True
            else:
                return False


class Tree():
    nCount = 0
    def __init__(self, min_splits=None, max_depth=None):
        # setting the root of the tree
        self.root = None

        # Stopping conditions
        self.min_splits = min_splits
        self.max_depth = max_depth

    def calcLeafValue(self, D):
        return(max(list(D), key=list(D).count))

    def leafNode(self, D, feature, graph, leafName):
        # Calculating the leaf value
        leaf_val = self.calcLeafValue(D['y_n'])
        print(f'Make leaf: {leaf_val}, feature: {feature}')

        # Creating and returning the node
        graph.node(feature + str(leaf_val), str(leaf_val))
        graph.edge(feature, feature + str(leaf_val), label=leafName)
        return(str(leaf_val))

    def intNode(self, feature, node_name, graph, edge_name):
        global nCount
        nCount += 1
        print(f'Make Internal Node: {node_name}, feature: {feature}')
        
        # Checking the type of node (i.e. root or internal)
        if feature == None:
            graph.node(node_name)
        else:
            graph.node(node_name)
            graph.edge(feature, node_name, label=edge_name)

    def MakeSubtree(self, D, best_feature, truth_val, graph, edge_name, Xi, node_lis, node_name):
        if stoppingCriteria(D, Xi):
            self.leafNode(D, node_name, graph, truth_val)
        else:
            C = DetermineCandidateSplits(D,Xi)
            S, printMax = FindBestSplit(D,C,Xi)
            best_Xi = S[6]
            node_name = f'{best_Xi} >= {printMax}'
            #print(node_name)
            node_lis.append(node_name)
            if best_feature == None:
                prev_node = None
            else:
                prev_node = node_lis[nCount-1]
            self.intNode(prev_node, node_name, graph, edge_name)
            for sub in listSplit(sortData(D,best_Xi),S[0]):
                #print('sub\n',sub)
                # print(min(sub[best_Xi]), printMax)
                if min(sub[best_Xi]) == printMax:
                    truth_val = "True"
                    edge_name = "True"
                else:
                    truth_val = "False"
                    edge_name = "False"
                Nchild = self.MakeSubtree(sub, best_Xi, truth_val, graph, edge_name, Xi, node_lis, node_name)
                #print(Nchild)
        #return(Nchild)

    def makeTree(self, D, name):
        global nCount
        nCount = 0
        graph = graphviz.Digraph()
        rules = self.MakeSubtree(D, None, None, graph, None, ['x_n1','x_n2'], [], None)
        graph.render(name)
        #print(nCount)
        return graph, rules

#Tree().MakeSubtree(data_raw['Druns.txt'], None, None, None, None, ['x_n1','x_n2'])
os.environ["PATH"] += 'C:/Program Files/Graphviz/bin'
file = 'D1'
Tree().makeTree(data_raw[file+'.txt'], f'../tree/{file}_tree')

# MakeSubtree(data_raw['D1.txt'], 'x_n1')
# print(data_raw['D1.txt'])
# for data, vals in data_raw['D1.txt'].items():
#     print(data,vals)
#     print(entropy(data))