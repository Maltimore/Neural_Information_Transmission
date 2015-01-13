# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 10:15:00 2015

@author: maltimore
"""
import model_neuron
import matplotlib.pyplot as plt
import numpy as np


N = 1000
n_neurons = 20
my_linewidth = .1


volt_matrix = np.empty((n_neurons, N))

voltage_vec = np.empty(N)
random_neuron_sample = np.array([0 ,4  ,5  ,9  ,3])
E_syn_list           = np.array([24 ,10 ,99 ,2  ,12])
neuronlist = []

# create neurons
for i in np.arange(n_neurons):
    neuronlist.append(model_neuron.Neuron(i))
    
for neuron in neuronlist:
    neuron.set_input_connections(np.arange(0,n_neurons),np.ones((n_neurons)))
    neuron.set_output_connections(neuronlist[:2])

j = 0
for i in np.arange(N):
    for neuron in neuronlist:
        
        neuron.set_external_current(np.random.random() * 3e-10)
        
        neuron.update()
        
        volt_matrix[j,i] = neuron.get_voltage()
        j += 1
    j = 0


for i in np.arange(n_neurons):
    
    plt.plot(volt_matrix[i], linewidth = my_linewidth)
    
plt.plot(volt_matrix[0], linewidth = 1)
plt.plot(volt_matrix[4], linewidth = 1)