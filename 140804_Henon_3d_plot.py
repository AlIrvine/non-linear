# -*- coding: utf-8 -*-
# Determinism in the UK GDP series. A python-based analysis.
# this will be the basic workings of my system bashed together...

# General idea: 1. Take a single series.
# 2. make an embedding routine (for use elsewhere) V
# 3. Check the embedding dimension by false near neighbour V
# 4. Calculate lyapunov exponent.
# 5. Prediction? ? 

# Importing various things:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from __future__ import division
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from numpy import *
from pandas import *

# 1. Importing my series
hen_base = ('/Users/Al/Documents/work/Edinburgh/Msc/Dissertation/'+
                            'local_work/Henon_new_stata.csv') #sets the path
base_data = pd.read_csv(hen_base) # imports the data
print base_data.ix[:4]     # Check how data has imported
# date = base_data.pop('date') 
# base_data.index = date # Set the index to the date variable
# print len(base_data.index) # establish length of series
data_len = len(base_data.index)
print data_len

def time_delay_embed(array, dimension, time_dif):
    """ A way to generate the time-delay embedding vectors
        for use in later steps, and as the dataset
        array: The name of the series being embedded
        dimension: how many elements to the vector
        time_dif: for if we want to delay by more than one time period (not used for now)"""
    emb = array.values # Converts the panda dataframe to an array
    emb = np.delete(emb, np.s_[0,2],1) # Given data structure, delete year and t-1 column
    emb = np.squeeze(np.asarray(emb)) # Make a 1-d array of all values
    i = data_len-1 # sets up a counter
    new_vec = [] # target for each row
    embed = [] # target for full set
    while i >= dimension-1:
        a = 0  # the dimensional counter
        b = 0  # time_dif counter
        while a< dimension:
            new_vec.append(emb[i-b])
            a+=1
            b+= time_dif
        embed.append(new_vec)
        new_vec = []
        i -=1  
    return embed
# Create a set of dimensions to check through
embed1 = np.asarray(time_delay_embed(base_data, 1,1))
embed2 = np.asarray(time_delay_embed(base_data, 2,1))
embed3 = np.asarray(time_delay_embed(base_data, 3,1))
embed4 = np.asarray(time_delay_embed(base_data, 4,1))
embed5 = np.asarray(time_delay_embed(base_data, 5,1))
embed6 = np.asarray(time_delay_embed(base_data, 6,1))
embed7 = np.asarray(time_delay_embed(base_data, 7,1))
embed8 = np.asarray(time_delay_embed(base_data, 8,1))
embed9 = np.asarray(time_delay_embed(base_data, 9,1))
embed10 = np.asarray(time_delay_embed(base_data, 10,1))
embed11 = np.asarray(time_delay_embed(base_data, 11,1))


# Creating a 1-d
def oned_graph():
    plt.clf() 
    x = embed1
    y = np.zeros(10000)
    concol = []
    for i in x:
        if -0.55<= i <=-0.45:
            concol.append('r')
        else:
            concol.append('b') 
    plt.title('Henon Map - 1-dimension', fontsize=20)
    plt.xlabel(r'$X_t$', fontsize=22)
    #plt.ylabel(r'$Y_t$', fontsize=22)
    plt.scatter(x,y, c = concol, s = 0.5, edgecolors = 'none')
    plt.show()

embed2_g = embed2.T
embed2_g[:2]


# Creating a 2-d
def twod_graph():
    plt.clf() 
    x,y = embed2_g
    concol = []
    for i in x:
        if -0.55<= i <=-0.45:
            concol.append('r')
        else:
            concol.append('b')
    plt.title('Henon Map - 2-dimension', fontsize=20)
    plt.xlabel(r'$X_t$', fontsize=22)
    plt.ylabel(r'$Y_t$', fontsize=22)
    plt.scatter(x,y, s=0.5,c=concol, edgecolors ='none')
    plt.show()  

# Creating a 3-d
embed3_g = embed3.T
embed3_g[:3]

def threed_graph():
    x,y,z = embed3_g
    concol = []
    for i in x:
        if -0.55<= i <=-0.45:
            concol.append('r')
        else:
            concol.append('b')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z , s=0.5, zdir = 'y', c = concol, edgecolors = 'none')
    plt.title('Henon Map - 3-dimension', fontsize=20)
    ax.set_xlabel(r'$X_t$', fontsize=22)
    ax.set_ylabel(r'$Y_t$', fontsize=22)
    ax.set_zlabel(r'$X_{t-2}$', fontsize=22)
    plt.show()
    

oned_graph()
twod_graph()
threed_graph()




