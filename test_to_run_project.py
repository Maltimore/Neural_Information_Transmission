# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 10:15:00 2015

@author: maltimore
"""
import model_neuron
import matplotlib.pyplot as plt
import numpy as np


N = 5000
n_neurons = 10
my_linewidth = .3
dt = .0001


volt_matrix = np.empty((n_neurons, N))

voltage_vec = np.empty(N)
random_neuron_sample = np.array([0 ,4  ,5  ,9  ,3])
E_syn_list           = np.array([24 ,10 ,99 ,2  ,12])
neuronlist = []

# create neurons
for i in np.arange(n_neurons):
    neuronlist.append(model_neuron.Neuron(i, dt))

i = 0
for neuron in neuronlist:
    neuron.set_input_connections([],np.ones((n_neurons)))
    i += 1

j = 0
for i in np.arange(N):
    for neuron in neuronlist:
        
#        neuron.set_external_current(3e-10)
        
        neuron.update()
        
        volt_matrix[j,i] = neuron.get_voltage()
        j += 1
    j = 0


for i in np.arange(2,n_neurons):
    
    plt.plot(np.linspace(0,N/10,N),volt_matrix[i], linewidth = my_linewidth)

plt.xlabel('time [ms]')
plt.ylabel('voltage [V]')

#plt.plot(volt_matrix[0], linewidth = .5)
#plt.plot(volt_matrix[1], linewidth = .5)
#plt.plot(volt_matrix[4], linewidth = .5)