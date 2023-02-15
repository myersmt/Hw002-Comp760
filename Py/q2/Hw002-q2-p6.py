"""
Created by: Matt Myers
02/15/2023

Q2-Part 6
"""
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt

# read in info
path_to_data = r'\Data'
os.chdir(os.getcwd()+path_to_data)
data_raw = {}
for file in os.listdir(os.getcwd()):
    if file in ['D1.txt', 'D2.txt']:
        data_raw[file]=(pd.read_csv(file, sep=' ', names=["x_n1","x_n2","y_n"], index_col=False))

# make plot
for key, val in data_raw.items():
    title = f'Scatterplot_of_'+key[0:2]
    plt.scatter(x=val["x_n1"], y=val["x_n2"], c=val["y_n"].map({0: 'orange', 1: 'purple'}))
    plt.xlabel('x_n1')
    plt.ylabel('x_n2')
    plt.title(title+'.txt')
    plt.savefig(f'../plots/Q6/{title}.png')
    plt.show()
    plt.clf()


