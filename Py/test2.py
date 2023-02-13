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

# Define a function to sort the data based on a feature
def sortData(D, Xi):
    return D[[Xi,'y_n']].sort_values(by=Xi).reset_index(drop=True)

# Define a function to determine the candidate splits for a feature
def DetermineCandidateSplits(D, Xi):
    C=[] # initialize set of candidate splits for feature Xi
    data_sort = sortData(D,Xi)
    # print(data_sort)
    for j in range(len(D)-1):
        if data_sort['y_n'][j] != data_sort['y_n'][j+1]:
            C.append(data_sort[Xi][j])
    return(C)

# Define a function to calculate the entropy of a split
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

# Define a function to calculate the information gain of a split
def InfoGain(splitDict):
    totals = {1:splitDict[0][1]+splitDict[1][1], 0:splitDict[0][0]+splitDict[1][0]}
    afterSplit = splitDict[1]
    beforeSplit = splitDict[0]
    beforeP = (beforeSplit[1]+beforeSplit[0])/(totals[1]+totals[0])
    afterP = (afterSplit[1]+afterSplit[0])/(totals[1]+totals[0])
    if beforeP == 0:
        beforeEntropy = 0
    else:
        beforeEntropy = -beforeP*np.log2(beforeP) - (1-beforeP)*np.log2(1-beforeP)
    if afterP == 0:
        afterEntropy = 0
    else:
        afterEntropy = -afterP*np.log2(afterP) - (1-afterP)*np.log2(1-afterP)
        infoGain = beforeEntropy - afterEntropy
    return infoGain

# Compute the gain ratio
def gainRatio(splitDict, intrinsicInformation):
    infoGain = InfoGain(splitDict)
    return infoGain/intrinsicInformation

# Split the data
def splitData(splitValue, D, Xi):
    before = []
    after = []
    data_sort = sortData(D,Xi)
    for i in range(len(data_sort[Xi])):
        if data_sort[Xi][i] >= splitValue:
            after.append(list(data_sort.iloc[i]))
        else:
            before.append(list(data_sort.iloc[i]))
    # Return as tuple (data before split, data after split)
    return (before, after)

# Recursive function to build the decision tree
def DecisionTree(data, features, depth):
    if depth == 0 or data == []:
        return None
    majority = int(np.mean(data['y_n']) > 0.5)
    # initialize node for current level in the tree
    node = {
        'value': majority,
        'feature': None,
        'left': None,
        'right': None
    }
    bestGain = 0
    # consider all candidate splits
    for feature in features:
        candidateSplits = DetermineCandidateSplits(data, feature)
        for split in candidateSplits:
            before, after = splitData(split, data, feature)
            before = pd.DataFrame(before, columns=["x_n1","x_n2","y_n"])
            after = pd.DataFrame(after, columns=["x_n1","x_n2","y_n"])
            beforeSum = before.sum(numeric_only=True)
            afterSum = after.sum(numeric_only=True)
            splitDict = ({0:beforeSum[2], 1:beforeSum[2]}, {0:afterSum[2], 1:afterSum[2]})
            intrinsicInformation = entropy(splitDict)
            gain = gainRatio(splitDict, intrinsicInformation)
            # if gain is found, update the node and best gain
            if gain > bestGain:
                node['feature'] = feature
                node['split'] = split
                bestGain = gain
                node['left'] = DecisionTree(before, features, depth-1)
                node['right'] = DecisionTree(after, features, depth-1)
    return node

# Display the decision tree
def DisplayTree(node, depth=0):
    if node == None:
        return
    if node['left'] == None and node['right'] == None:
        print("Leaf node: value = ", node['value'])
        return
    print("Node at depth ", depth, ": feature = ", node['feature'], "split = ", node['split'])
    DisplayTree(node['left'], depth+1)
    DisplayTree(node['right'], depth+1)

# Plot the decision tree
def PlotTree(node, dot, depth=0):
    if node == None:
        return
    if node['left'] == None and node['right'] == None:
        name = """ + str(node['class_label']) + """
        dot.node(str(id(node)), name, shape="box")
    else:
        name = """ + str(node['attribute']) + """
        dot.node(str(id(node)), name)
        PlotTree(node['left'], dot, depth + 1)
        PlotTree(node['right'], dot, depth + 1)
        dot.edge(str(id(node)), str(id(node['left'])), "0")
        dot.edge(str(id(node)), str(id(node['right'])), "1")

# Define a function to create the decision tree
def create_decision_tree(data, features, depth):
    if depth == 0 or data.shape[0] == 0:
        return None
    majority = int(np.mean(data['y_n']) > 0.5)
    # initialize node for current level in the tree
    node = {
        'value': majority,
        'feature': None,
        'left': None,
        'right': None
    }
    bestGain = 0
    # consider all candidate splits
    for Xi in features:
        candidate_splits = DetermineCandidateSplits(data, Xi)
        for splitValue in candidate_splits:
            split_data = splitData(splitValue, data, Xi)
            gain = InfoGain(split_data)
            if gain > bestGain:
                bestGain = gain
                bestSplitValue = splitValue
                bestSplitFeature = Xi
    if bestGain > 0:
        node['feature'] = bestSplitFeature
        node['value'] = bestSplitValue
        beforeSplit, afterSplit = splitData(bestSplitValue, data, bestSplitFeature)
        node['left'] = create_decision_tree(beforeSplit, features, depth-1)
        node['right'] = create_decision_tree(afterSplit, features, depth-1)
    return node

# Define a function to visualize the decision tree using graphviz
def visualize_decision_tree(tree):
    dot = graphviz.Digraph(format='png')
    node_count = 0

    def add_nodes(node, parent_node):
        nonlocal node_count
        if node is None:
            return None
        node_count += 1
        current_node = str(node_count)
        if node['feature'] is not None:
            dot.node(current_node, label=f"{node['feature']} >= {node['value']}")
        else:
            dot.node(current_node, label=f"class = {node['value']}")
        if parent_node is not None:
            dot.edge(parent_node, current_node)
        add_nodes(node['left'], current_node)
        add_nodes(node['right'], current_node)

    add_nodes(tree, None)
    dot.render('decision_tree')

# Create the decision tree
tree = create_decision_tree(data_raw[list(data_raw.keys())[0]], ['x_n1', 'x_n2'], 10)

# Visualize the decision tree
visualize_decision_tree(tree)
