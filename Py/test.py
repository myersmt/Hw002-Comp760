import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def read_data(path_to_data):
    os.chdir(os.getcwd() + path_to_data)
    data_raw = {}
    for file in os.listdir(os.getcwd()):
        data_raw[file] = (pd.read_csv(file, sep=' ', names=["x_n1", "x_n2", "y_n"], index_col=False))
    return data_raw

def sort_data(data, feature):
    print(feature)
    return data[[feature, 'y_n']].sort_values(by=feature).reset_index(drop=True)

def determine_candidate_splits(data, feature):
    candidate_splits = []
    data_sort = sort_data(data, feature)
    for j in range(len(data) - 1):
        if data_sort['y_n'][j] != data_sort['y_n'][j + 1]:
            candidate_splits.append(data_sort[feature][j])
    return candidate_splits

def entropy(split_dict):
    wins = split_dict[0][1]
    loss = split_dict[1][0]
    total = wins + loss
    pwin = wins / total
    plos = loss / total
    if pwin == 0:
        return (-1) * plos * np.log2(plos)
    elif plos == 0:
        return (-1) * pwin * np.log2(pwin)
    else:
        return (-1) * (pwin * np.log2(pwin) + plos * np.log2(plos))

def info_gain(total, wins, loss, at, aw, al, bt, bw, bl):
    return entropy(total, wins, loss) - ((at / total) * entropy(at, aw, al) + (bt / total) * entropy(bt, bw, bl))

def info_Gain_ratio(total, wins, loss, at, aw, al, bt, bw, bl, total_split):
    return info_gain(total, wins, loss, at, aw, al, bt, bw, bl) / entropy(total_split, 0, 0)

def stopping_criteria(candidate_splits, total, wins, loss):
    if candidate_splits == [] or entropy(total, wins, loss) == 0 or all(info_Gain_ratio(total, wins, loss, at, aw, al, bt, bw, bl, (at + bt)) == 0 for at, aw, al, bt, bw, bl in candidate_splits):
        return True
    return False

def make_leaf(data):
    if data['y_n'].sum() >= len(data) / 2:
        return 1
    return 0

def plot_decision_tree(tree, parent_node=None, x_pos=0, y_pos=0, y_offset=1):
    node_positions = {}
    y_pos -= y_offset
    
    for node, branches in tree.items():
        node_positions[node] = (x_pos, y_pos)
        x_pos += 1
        
        for branch_name, branch in branches.items():
            if type(branch) == dict:
                plot_decision_tree(branch, node, x_pos, y_pos, y_offset/2)
            else:
                node_positions[branch] = (x_pos, y_pos)
                x1, y1 = node_positions[node]
                x2, y2 = node_positions[branch]
                plt.plot([x1, x2], [y1, y2], '-o')
    
    for node, (x, y) in node_positions.items():
        plt.annotate(node, (x, y))
    
    plt.show()

def create_decision_tree(data, feature_names):
    # If stopping criteria met, return leaf node
    if stopping_criteria(determine_candidate_splits(data, feature_names), len(data), data['y_n'].sum(), len(data) - data['y_n'].sum()):
        return make_leaf(data)
    # Store best feature and best split
    best_feature = None
    best_split = None
    max_info_gain_ratio = -1
    for feature in feature_names:
        candidate_splits = determine_candidate_splits(data, feature)
        for split in candidate_splits:
            data_split_1 = data[data[feature] <= split]
            data_split_2 = data[data[feature] > split]
            info_gain_ratio_ = info_Gain_ratio(len(data), data['y_n'].sum(), len(data) - data['y_n'].sum(), len(data_split_1), data_split_1['y_n'].sum(), len(data_split_1) - data_split_1['y_n'].sum(), len(data_split_2), data_split_2['y_n'].sum(), len(data_split_2) - data_split_2['y_n'].sum(), [(len(data_split_1), 0, 0), (len(data_split_2), 0, 0)])
            if info_gain_ratio_ > max_info_gain_ratio:
                max_info_gain_ratio = info_gain_ratio_
                best_feature = feature
                best_split = split

    # Create decision tree with best feature and best split
    decision_tree = {f"{best_feature} <= {best_split}": {}}
    data_split_1 = data[data[best_feature] <= best_split]
    data_split_2 = data[data[best_feature] > best_split]
    feature_names.remove(best_feature)
    decision_tree[f"{best_feature} <= {best_split}"]["Left"] = create_decision_tree(data_split_1, feature_names)
    decision_tree[f"{best_feature} <= {best_split}"]["Right"] = create_decision_tree(data_split_2, feature_names)

    return decision_tree

path_to_data = r'\Data'
os.chdir(os.getcwd()+path_to_data)
data_raw = {}
for file in os.listdir(os.getcwd()):
    data_raw[file]=(pd.read_csv(file, sep=' ', names=["x_n1","x_n2","y_n"], index_col=False))

feature_names = ["x_n1", "x_n2"]
decision_tree = create_decision_tree(data_raw['D1.txt'], feature_names)

plot_decision_tree(decision_tree)

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import os

# # Read data from files
# path_to_data = r'\Data'
# os.chdir(os.getcwd() + path_to_data)
# data_raw = {}
# for file in os.listdir(os.getcwd()):
#     data_raw[file] = (pd.read_csv(file, sep=' ', names=["x_n1", "x_n2", "y_n"], index_col=False))

# # Organize data
# def sort_data(data, feature):
#     return data[[feature, 'y_n']].sort_values(by=feature).reset_index(drop=True)

# # Determine candidate splits
# def determine_candidate_splits(data, feature):
#     candidate_splits = []
#     data_sort = sort_data(data, feature)
#     for j in range(len(data) - 1):
#         if data_sort['y_n'][j] != data_sort['y_n'][j + 1]:
#             candidate_splits.append(data_sort[feature][j])
#     return candidate_splits

# # Entropy of one split
# def entropy(split_dict):
#     wins = split_dict[0][1]
#     loss = split_dict[1][0]
#     total = wins + loss
#     pwin = wins / total
#     plos = loss / total
#     if pwin == 0:
#         return (-1) * plos * np.log2(plos)
#     elif plos == 0:
#         return (-1) * pwin * np.log2(pwin)
#     else:
#         return (-1) * (pwin * np.log2(pwin) + plos * np.log2(plos))

# # Information gain
# def info_gain(total, wins, loss, at, aw, al, bt, bw, bl):
#     return entropy(total, wins, loss) - ((at / total) * entropy(at, aw, al) + (bt / total) * entropy(bt, bw, bl))

# # Info gain ratio
# def info_gain_ratio(total, wins, loss, at, aw, al, bt, bw, bl, total_split):
#     return info_gain(total, wins, loss, at, aw, al, bt, bw, bl) / entropy(total_split, 0, 0)

# # Stopping criteria
# def stopping_criteria(candidate_splits, total, wins, loss):
#     if candidate_splits == [] or entropy(total, wins, loss) == 0 or all(info_gain_ratio(total, wins, loss, at, aw, al, bt, bw, bl, (at + bt)) == 0 for at, aw, al, bt, bw, bl in candidate_splits):
#         return True
#     return False

# # Make leaf node
# def make_leaf(data):
#     if data['y_n'].sum() >= len(data) / 2:
#         return 1
#     return 0

# def plot_decision_tree(tree, parent_node=None):
#     # Get the root node of the tree
#     root = list(tree.keys())[0]
#     # Get the number of nodes
#     num_nodes = len(tree.keys())
#     # Initialize the node positions
#     node_positions = dict()
#     # Calculate the x and y positions of the nodes
#     for i, node in enumerate(tree.keys()):
#         node_positions[node] = (i / (num_nodes + 1), 0)
#     # Plot the tree
#     for node, branches in tree.items():
#         for branch_name, branch in branches.items():
#             if type(branch) == dict:
#                 # Recursively call the function for the branches
#                 plot_decision_tree(branch, node)
#             else:
#                 # Plot the line connecting the parent node to the branch
#                 parent_x, parent_y = node_positions[node]
#                 branch_x, branch_y = node_positions[branch]
#                 plt.plot([parent_x, branch_x], [parent_y, branch_y], '-o')
#     # Add the labels to the nodes
#     for node, (x, y) in node_positions.items():
#         plt.annotate(node, (x, y))
#     # Show the plot
#     plt.show()

# # Example tree
# tree = {'Root': {'Branch 1': 'Leaf 1', 'Branch 2': 'Leaf 2'}}

# # Plot the tree
# plot_decision_tree(tree)
