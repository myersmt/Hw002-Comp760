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
    return D.sort_values(by=Xi)

# Determine splits
def DetermineCandidateSplits(D, Xi):
    C={} # initialize set of candidate splits for feature Xi
    for feature in Xi:
        max_index = max(D[feature].index.values)
        lis = []
        #print(feature, Xi)
        #print(D, feature, Xi)
        #print(D,'DDDDDDDDDDDDDDDDDDDDDDDDDDDD')
        data_sort = sortData(D,feature)
        #new_index=sorted_index(D, data_sort, max_index)
        for j in range(len(D)-1):
            #print((max_index,data_sort[data_sort.index==j].index.values==data_sort[data_sort.index==max_index].index.values)[-1])
            #if data_sort['y_n'][j] != data_sort['y_n'][j+1] and ((data_sort[data_sort.index==4].index.values==data_sort[data_sort.index==max_index].index.values)==[True]):
            #print(data_sort.iloc[j].name==D.iloc[max_index].name)
            #if data_sort['y_n'][j] != data_sort['y_n'][j+1] and data_sort.iloc[j].name==D.iloc[max_index].name:
            if data_sort['y_n'][j] != data_sort['y_n'][j+1]:
                lis.append(j)
                # print('->>>>>>>>>>>>>>>>',j,max_index)
                # if j == max_index:
                #     print('->>>>>>>>>>>>>>>>',j,max_index)
        C[feature]=lis
        #print('Feature: ', feature, ', Candidates:', C[feature])
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
    #print('split gain candidates:',C)
    for feature in Xi:
        dsort = sortData(D,feature)
        #print('dsort:',dsort, feature)
        gains = []
        for ind, val in enumerate(dsort[feature]):
            if ind in C[feature]:
                #print('gains index:', ind, val)
                gains.append([val,SplitCount(D,ind)])
                #print('Gains:',gains)
        dic[feature] = gains
    #print('split gain return:',dic)
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
    #print('Candidate splits in find best split:',C, dict_of_gains)
    full_dict = SplitGainFull(D,C,Xi)
    #print(D)
    #print('............',sortData(dict_of_gains['x_n1'],'x_n1'))
    for feature in Xi:
        #dict_of_gains=sortData(dict_of_gains,feature)
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
            print(x,list(dict_of_gains[feature])[x][1],list(dict_of_gains[feature])[x],GainRatio(list(dict_of_gains[feature])[x][1]))
            if GainRatio(list(dict_of_gains[feature])[x][1]) >= comp_max :
                #print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
                #print(GainRatio(list(dict_of_gains.items())[0][1]))
                best_feature = feature
                comp_max = GainRatio(list(dict_of_gains[feature])[x][1])
                entropy_max = entropy(list(dict_of_gains[feature])[x][1])
                info_gain = InfoGain(list(dict_of_gains[feature])[x][1])
                max = list(dict_of_gains[feature])[x]
                #max = list(dict_of_gains[feature])[x][0]
                #maxNEXT = list(dict_of_gains[feature])[x+1][0]
                c_num=x
            #print(list(dict_of_gains.items())[x][0],f'{x}',GainRatio(list(dict_of_gains.items())[x][1]))
        #print(enumerate(sortData(D[[feature]],feature)))
        nextC = False
        maxNEXT=0
        for index, row in enumerate(full_dict[feature]):
            if row == max:
                #print(row, max, 'KKKKKKKKKKKKKKKKKKKKKKKKK')
                #print(row[0],max[0])
                lis_num = index
                nextC = True
            elif nextC:
                maxNEXT = row[0]
                nextC = False
                break
        # nextC = False
        # maxNEXT = 0
        # for index, row in sortData(D,feature).iterrows():
        #     print('row[feature]:',row[feature],', max:', max,', index:', index, "<<<<<<<<<<<<<<<<<<<")
        #     if row[feature] == max:
        #         lis_num = index
        #         nextC = True
        #     elif nextC:
        #         maxNEXT = row[feature]
        #         nextC = False
        #         break
        #         print(row, val, '<====')
    print([lis_num, c_num, 'max:',max[0], comp_max, entropy_max, info_gain, best_feature], maxNEXT)
    return([lis_num, c_num, max[0], comp_max, entropy_max, info_gain, best_feature], maxNEXT)

def listSplit(dict, ind):
    #print('split index:',ind, dict)
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
            #print(candidate,feature, candidate[1], entropy(candidate[1]))
            if entropy(candidate[1]) == 0:
                print('stop 2')
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
        return(feature)

    def intNode(self, feature, node_name, graph, edge_name):
        global nCount
        nCount += 1
        print(f'Make Internal Node: {node_name}, feature: {feature}')
        
        # Checking the type of node (i.e. root or internal)
        if feature == None:
            graph.node(node_name)
        else:
            #print('Edge Name:', edge_name)
            graph.node(node_name)
            graph.edge(feature, node_name, label=edge_name)
    
    def MakeSubtree(self, D, best_feature, truth_val, graph, edge_name, Xi, node_lis, node_name):
        if stoppingCriteria(D, Xi):
            self.leafNode(D, node_name, graph, truth_val)
        else:
            C = DetermineCandidateSplits(D,Xi)
            #print('Candidate splits:',C)
            S, printMax = FindBestSplit(D,C,Xi)
            print('->>>>>>>>>>>>>>',printMax)
            best_Xi = S[6]
            node_name = f'{best_Xi} >= {printMax}'
            #print(node_name)
            node_lis.append(node_name)
            print('->>>>>>>>>>>>>>>',node_lis)
            if best_feature == None:
                #print(D)
                prev_node = None
            else:
                prev_node = node_lis[nCount-1]
            #print('splits:', C, 'S[0]:', S[0])
            self.intNode(prev_node, node_name, graph, edge_name)
            for sub in listSplit(sortData(D,best_Xi),S[0]):
                print(S)
                print('sub\n',sub)
                print('max:', S[2], 'min:',min(sub[best_Xi]))
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
Tree().makeTree(data_raw['Druns.txt'], '../tree/drun_tree')

# MakeSubtree(data_raw['D1.txt'], 'x_n1')
# print(data_raw['D1.txt'])
# for data, vals in data_raw['D1.txt'].items():
#     print(data,vals)
#     print(entropy(data))