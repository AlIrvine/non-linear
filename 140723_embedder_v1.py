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
from numpy import *
from pandas import *


# 1. Importing my series
gdp_base = ('/Users/Al/Documents/work/Edinburgh/Msc/Dissertation/'+
                            'local_work/140723GDP1.csv') #sets the path
base_data = pd.read_csv(gdp_base) # imports the data
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
    emb = np.delete(emb, np.s_[0],1) # Given data structure, delete year column
    emb = np.squeeze(np.asarray(emb)) # Make a 1-d array of all values
    i = data_len-1 # sets up a counter
    new_vec = [] # target for each row
    embed = [] # target for full set
    while i >= dimension-1:
        a = 0  # the dimensional counter
        while a< dimension:
            new_vec.append(emb[i-a])
            a+=1
        embed.append(new_vec)
        new_vec = []
        i -=1  
    return embed
# Create a set of dimensions to check through
embed1 = np.asarray(time_delay_embed(base_data, 1,0))
embed2 = np.asarray(time_delay_embed(base_data, 2,0))
embed3 = np.asarray(time_delay_embed(base_data, 3,0))
embed4 = np.asarray(time_delay_embed(base_data, 4,0))
embed5 = np.asarray(time_delay_embed(base_data, 5,0))
embed6 = np.asarray(time_delay_embed(base_data, 6,0))
embed7 = np.asarray(time_delay_embed(base_data, 7,0))
embed8 = np.asarray(time_delay_embed(base_data, 8,0))
embed9 = np.asarray(time_delay_embed(base_data, 9,0))
embed10 = np.asarray(time_delay_embed(base_data, 10,0))
embed11 = np.asarray(time_delay_embed(base_data, 11,0))

print len(embed1) #embed 1 having index issues 
print len(embed6)
print len(embed7) #embed 7 seems to cause issues (as lower and higher)
print len(embed8)
print len(embed9) #embed 9 has index issue as lower dimension
print len(embed10)
print len(embed11)

print embed7[:3]
print embed3
print embed4[:3]
print embed6[:3]
print embed3[1]

print embed3.shape
print embed4.shape
            

# make the distance-ratio list for all points in an embedding:

def near_neighbour_checker(array1,array2):
    """ An approach to checking the distance between neighbours as dimension
    increases, as in Kennel et al 1992.
    array1: An array of dimension n
    array2: An array of dimension n+1"""
    maxlen = min(len(array1), len(array2)) # The last point in dimension n
                                        # Is not in dimension n+1
    i = 0
    ratiolist = []
    while i <maxlen:
        distlist = [] # target list for distances between points
    
        m=0
        while m<maxlen:
            dist = np.linalg.norm(array1[i] - array1[m]) # Index 134 used as example
            distlist.append(dist)
            m+=1
    
        y = distlist.index(min(x for x in distlist if x != 0))
        
        dist_min_n = np.linalg.norm(array1[i]- array1[y]) # Euclidean norm distance
        dist_min_n1 = np.linalg.norm(array2[i] - array2[y]) # Euclidean norm distance
        dist_min_n_sq = (dist_min_n)**2 # For use in ratio difference
        dist_min_n1_sq = (dist_min_n1)**2 # For use in ratio difference
        dist_ratio = math.sqrt((dist_min_n1_sq - dist_min_n_sq) / dist_min_n_sq)
        ratiolist.append(dist_ratio)
        i +=1
    return ratiolist

# Checking which dimension comparisons are having index issues
near_neighbour_checker(embed1,embed2)
near_neighbour_checker(embed2,embed3)
near_neighbour_checker(embed3,embed4)
near_neighbour_checker(embed4,embed5)
near_neighbour_checker(embed5,embed6)
near_neighbour_checker(embed6,embed7)
near_neighbour_checker(embed7,embed8)
near_neighbour_checker(embed8,embed9)
near_neighbour_checker(embed9,embed10)
near_neighbour_checker(embed10,embed11)

# Count the number of near neighbours whose ratio is over some critical size!
# need to generate the list, use a defined tolerance to compare, get the proportion

def near_neighbour_method1(tolerance):
    """ Calculates the proportions of near neighbours using Kennel et al. method1
    False Near Neighbours (FNN) are points that move far away from each other as
    dimension increases
    Tolerance: The extra distance proportion for the point to be a FNN"""
    method1_list = []
    
    # Count the number of FFNs in each dimension shift
    tol_count1 = sum(1 for x in near_neighbour_checker(embed1,embed2) if  x >tolerance)
    tol_count2 = sum(1 for x in near_neighbour_checker(embed2,embed3) if  x >tolerance)
    tol_count3 = sum(1 for x in near_neighbour_checker(embed3,embed4) if  x >tolerance)
    tol_count4 = sum(1 for x in near_neighbour_checker(embed4,embed5) if  x >tolerance)
    tol_count5 = sum(1 for x in near_neighbour_checker(embed5,embed6) if  x >tolerance)
    tol_count6 = sum(1 for x in near_neighbour_checker(embed6,embed7) if  x >tolerance)
    tol_count7 = sum(1 for x in near_neighbour_checker(embed7,embed8) if  x >tolerance)
    tol_count8 = sum(1 for x in near_neighbour_checker(embed8,embed9) if  x >tolerance)
    tol_count9 = sum(1 for x in near_neighbour_checker(embed9,embed10) if  x >tolerance)
    tol_count10 = sum(1 for x in near_neighbour_checker(embed10,embed11) if  x >tolerance)
    
    # Pull in the number of points for each embedding for a proportion to be taken
    embed_tot1 = len(near_neighbour_checker(embed1,embed2))
    embed_tot2 = len(near_neighbour_checker(embed2,embed3))
    embed_tot3 = len(near_neighbour_checker(embed3,embed4))
    embed_tot4 = len(near_neighbour_checker(embed4,embed5))
    embed_tot5 = len(near_neighbour_checker(embed5,embed6))
    embed_tot6 = len(near_neighbour_checker(embed6,embed7))
    embed_tot7 = len(near_neighbour_checker(embed7,embed8))
    embed_tot8 = len(near_neighbour_checker(embed8,embed9))
    embed_tot9 = len(near_neighbour_checker(embed9,embed10))
    embed_tot10 = len(near_neighbour_checker(embed10,embed11))
    
    # Actually create the proportions
    false_near_prop1 = tol_count1 / embed_tot1
    false_near_prop2 = tol_count2 / embed_tot2
    false_near_prop3 = tol_count3 / embed_tot3
    false_near_prop4 = tol_count4 / embed_tot4
    false_near_prop5 = tol_count5 / embed_tot5
    false_near_prop6 = tol_count6 / embed_tot6
    false_near_prop7 = tol_count7 / embed_tot7
    false_near_prop8 = tol_count8 / embed_tot8
    false_near_prop9 = tol_count9 / embed_tot9
    false_near_prop10 = tol_count10 / embed_tot10
    
    # Add results to a list
    method1_list.extend([false_near_prop1,false_near_prop2,
                        false_near_prop3,false_near_prop4,
                        false_near_prop5,false_near_prop6,
                        false_near_prop7,false_near_prop8,
                        false_near_prop9,false_near_prop10])
    method1_array = np.array(method1_list)
    print method1_list
    print method1_array
    return method1_list

near_neighbour_method1(2)

def near_neighbour_graph(tolerance):
    """Plots the proportion of False Near Neighbours (FFNs) in a given series
    as dimension increases.
    Tolerance: Passed to 'near_neighbour_method1' to generate the data"""
    plt.clf()
    xticks = np.arange(1,11,1)
    yticks = np.arange(-0.1,0.8,0.1)
    plt.plot(xticks,near_neighbour_method1(tolerance))
    plt.xlabel('Dimension')
    plt.ylabel('Proportion of False Near Neighbours')
    plt.axis([1,10,-0.1,0.8])
    xticks = np.arange(1,10,1)
    yticks = np.arange(-0.1,0.8,0.1)
    plt.yticks(yticks)
    plt.axhline(y=0.05, xmin=0, xmax=10,color='r')
    plt.show()
    
near_neighbour_graph(10)
    
# Near neighbour measures:
# Ratio difference: count numbers of nearest differences that exceed some tolerance
# (method 1 of Kennel et al.)
# A_tol method (see kennel et al. pg 94 in copin w/ chaos)


# Where to go from here:

# Calculate Maximal Lyapunov exponent for 
# chosen dimensionalities (and one either side...)

