# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 17:53:21 2015

@author: tabea
"""

import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np

def arrowplot(array):
    for i in range(len(array)-1):
        ax.arrow(array[i][1],array[i][0],array[i+1][1]-array[i][1],\
                 array[i+1][0]-array[i][0],fc='k',ec='k',head_width=0.05,\
                 head_length=1, length_includes_head=True)
    
with open('simulation.txt','rb') as f:
    # number of trials per starting point
    number = pickle.load(f)  
    # data array
    data = pickle.load(f)
    
#data[np.isnan(data)]=0
    
numberstartingpoints = data.shape[1]/(number*2)

## get empty array for the plot
processeddata = np.empty((len(data),numberstartingpoints*2))

## processing of data
for i in range(numberstartingpoints):
    # only data with the same starting point
    samestart = data[:,number*2*i:number*2*(i+1)]
    # pick only even columns and average for the x coordinates
    avrgvalx = np.mean(samestart[:,::2],1)
    # pick only odd columns and average for the y coordinates
    avrgvaly = np.mean(samestart[:,1::2],1)
    # store the new data in array
    processeddata[:,i*2] = avrgvalx
    processeddata[:,(i*2)+1] = avrgvaly

## find maximal value of sigma
sigmas = processeddata[:,1::2]
sigmamax = np.max(sigmas)

## plot graph
ax = plt.axes()
for i in np.arange(0,processeddata.shape[1],2):
    arrowplot(processeddata[:,i:i+2])
plt.xlabel(r'$\sigma$ [ms]')
plt.ylabel('a [spikes]')
plt.ylim(0,100)
plt.xlim(0,sigmamax+0.3)
plt.show()