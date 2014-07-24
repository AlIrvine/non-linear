# Determinism in the UK GDP series. A python-based analysis.
# this will be the basic workings of my system bashed together...

# General idea: 1. Take a single series.
# 2. make an embedding routine (for use elsewhere)
# 3. Check the embedding dimension by false near neighbour
# 4. Calculate lyapunov exponent.

# Work out the rest later

# Importing various things:
import numpy as np
import pandas as pd
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
embed2 = np.asarray(time_delay_embed(base_data, 2,0))
embed3 = np.asarray(time_delay_embed(base_data, 3,0))
embed4 = np.asarray(time_delay_embed(base_data, 4,0))
embed5 = np.asarray(time_delay_embed(base_data, 5,0))
embed6 = np.asarray(time_delay_embed(base_data, 6,0))

print embed3[:3]
print embed3
print embed4[:3]
print embed6[:3]
print embed3[1]

print embed3.shape
print embed4.shape

dist = np.linalg.norm(embed3[1] - embed3[2])
dist2 = np.linalg.norm(embed4[1] - embed4[2])

print dist, dist2
print (dist2 - dist)/dist

"""def near_neighbour_checker(array1, array2):
    An approach to checking the distance to every other point,
    and how far they move as dimension increases
    array1: an array in dimension n
    array2: an array in dimension n+1)
    maxlen = min(len(array1), len(array2))
    distlist = []
    for i in [0,maxlen]:
        m=1
        dist = np.linalg.norm(array1[i] - array1[i+m]
        distlist.append"""
        
# Everything below this is where I'm stuck:

maxlen = min(len(embed3), len(embed4)) # The last index item in the n-dimension is not in the n+1-dimension
print maxlen # check length
distlist = [] # target list for distances between points
ratiolist = []
# I want to wrap what's below this to go through each point
m=0 
while m<=maxlen:
    dist = np.linalg.norm(embed3[134] - embed3[m]) # Index 134 used as example
    distlist.append(dist)
    m+=1

print distlist  # prints a list of all the distances between index 134 and itself
y = distlist.index(min(x for x in distlist if x != 0))
print y
dist_min_n = np.linalg.norm(embed3[134]- embed3[y])
dist_min_n1 = np.linalg.norm(embed4[134] - embed4[y])
dist_ratio = (dist_min_n1 - dist_min_n) / dist_min_n
print dist_ratio
ratiolist.append(dist_ratio)
 
# Where to go from here:
 # find the second smallest value's index. (what I'm stuck on!)
 # pull in the n+1 dimensional array
 # abusing the index being the same for each point in the embedding, only compare
 # the distance between the point we're on and it's nearest neighbour (the index found above)
 # using the ratio on line 70. Store this.

# In a different routine - determine a critical ratio value (say 10%) and count
# how many 'false neighbours' (dist change >10%) there are moving up dimensions
# graph the proportions of false near neighbours. 
