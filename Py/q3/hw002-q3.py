"""
Created by: Matt Myers
02/15/2023

Q3
"""
# Needed libraries
from sklearn import tree
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt

# read in info
path_to_data = r'\Data'
os.chdir(os.getcwd()+path_to_data)
data_raw = {}
for file in os.listdir(os.getcwd()):
    data_raw[file]=(pd.read_csv(file, sep=' ', names=["x_n1","x_n2","y_n"], index_col=False))

# Creating the random sample
df = data_raw['Dbig.txt']
df = df.sample(frac=1)

# Initializing the sets of data
df_32= df.iloc[:32]
df_128= df.iloc[:128]
df_512= df.iloc[:512]
df_2048= df.iloc[:2048]
df_8192= df.iloc[:8192]
df_test= df.iloc[8192:]
features=['x_n1','x_n2']
y = 'y_n'

df_dic = {'df_32': df_32, 'df_128': df_128,'df_512': df_512,'df_2048': df_2048,'df_8192': df_8192}

# Creating a dictionary of all of the desired data and looping through the initial dicitonary
fit_dic = {}
for key, val in df_dic.items():
    new_name = key+'_tree'
    key_tree = tree.DecisionTreeClassifier()
    key_tree = key_tree.fit(val[features], val[[y]])
    node = key_tree.tree_.node_count
    predict = key_tree.predict(df_test[features])
    error = sum(np.abs(predict-df_test[y]))/len(df_test)
    print(f'{new_name}: Number of nodes: {node}')
    print(f'{new_name}: Error: {error}')
    fit_dic[new_name] = {'Nodes': node, 'Error': error}

errors = []
nodes = []

# Extracting the data from the dictionary
for key, val in fit_dic.items():
    for k, v in val.items():
        if k == 'Error':
            errors.append(v)
        elif k == 'Nodes':
            nodes.append(v)

# Plotting
plt.plot(nodes,errors)
plt.xlabel('nodes')
plt.ylabel('error')
plt.tight_layout()
plt.title('Plot of nodes versus error: Q3')
plt.show()
