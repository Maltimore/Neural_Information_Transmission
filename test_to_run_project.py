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
N_timesteps = 500
N_per_group = 100
N_groups    = 4
N_neurons   = N_per_group * N_groups
input_spikes = 50
my_linewidth = .2
dt = .0001

#####################################
# Create some vectors and matrices
volt_matrix = np.empty((N_neurons, N_timesteps))
voltage_vec = np.empty(N_timesteps)
neuronlist = []

#####################################
# FUNCTIONS    
def organizeNeurons(N_per_group, N_groups):   
    neuronlist = []
    
    # create neurons
    for i in np.arange(N_per_group * N_groups):
        neuronlist.append(model_neuron.Neuron(i, dt))
    
    # set output connections
    i = 0
    for k in np.arange(N_groups-1):       
        outputvec = neuronlist[ (k+1)*N_per_group  :  (k+2)*N_per_group ]
        
        for l in np.arange(N_per_group):
            neuronlist[i].set_output_connections(outputvec)
            i += 1
            
    return neuronlist
    
def give_initial_input(neuronlist, N_per_group, synchronization, N_spikes):
    for i in np.arange(N_spikes):
        artificial_input_neuron = model_neuron.Neuron(-(i+1), dt)
        artificial_input_neuron.set_output_connections(neuronlist[0:N_per_group])
        artificial_input_neuron.fire()

def rasterplot():
    eventmatrix = np.zeros((N_neurons,N_timesteps))
    for i in range(N_neurons):
        for j in range(len(neuronlist[i].spiketime)):
            eventmatrix[i][neuronlist[i].spiketime[j]/dt] = 1
    plt.figure()
    plt.xlabel('timestep')
    plt.ylabel('neuron number')
    plt.imshow(eventmatrix,cmap='gray_r')

#####################################
# Run simulation

neuronlist = organizeNeurons(N_per_group,N_groups)
give_initial_input(neuronlist, N_per_group, 0, input_spikes)

j = 0
for i in np.arange(N_timesteps):
    for neuron in neuronlist:
        neuron.update()
        volt_matrix[j,i] = neuron.get_voltage()
        j += 1
    j = 0

# Plot Voltage for all simulated neurons
plt.figure()
for i in np.arange(N_neurons):    
    plt.plot(np.linspace(0,N_timesteps*dt*1000,N_timesteps),volt_matrix[i], linewidth = my_linewidth)
plt.xlabel('time [ms]')
plt.ylabel('voltage [V]')

rasterplot()


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