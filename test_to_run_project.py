# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 10:15:00 2015

@author: maltimore
"""
import model_neuron
import matplotlib.pyplot as plt
import numpy as np


#####################################
# Set constant variables
N = 5000
n_neurons = 50
my_linewidth = .2
dt = .0001

#####################################
# Create some vectors and matrices
volt_matrix = np.empty((n_neurons, N))
voltage_vec = np.empty(N)
neuronlist = []

#####################################
# create neurons
for i in np.arange(n_neurons):
    neuronlist.append(model_neuron.Neuron(i, dt))
    
#####################################
# set output connections
i = 0
for neuron in neuronlist:
    neuron.set_output_connections([neuronlist[0]])
    i += 1

#####################################
# Run simulation
j = 0
for i in np.arange(N):
    for neuron in neuronlist:
        neuron.update()
        volt_matrix[j,i] = neuron.get_voltage()
        j += 1
    j = 0

# Plot Voltage for all simulated neurons
plt.figure()
for i in np.arange(n_neurons):    
    plt.plot(np.linspace(0,N/10,N),volt_matrix[i], linewidth = my_linewidth)

plt.plot(np.linspace(0,N/10,N), volt_matrix[0], linewidth = 2)
plt.xlabel('time [ms]')
plt.ylabel('voltage [V]')


#N = 100000
## find best I_mu
#I_mu = np.arange(2.7e-10,2.74e-10,1e-13)
#firing_rate = np.empty(len(I_mu))
#
#
#i = 0
#for I in I_mu:
#    testneuron = model_neuron.Neuron(0,dt)
#    testneuron.set_input_connections([],1)
#    testneuron.set_I_mu(I)
#    
#    for j in np.arange(N):
#        testneuron.update()
#    
#    firing_rate[i] = testneuron.get_firing_rate()
#    i += 1
#
#plt.figure()
#plt.plot(I_mu, firing_rate)
#plt.xlabel('I_mu [A]')
#plt.ylabel('Firing rate [Hz]')