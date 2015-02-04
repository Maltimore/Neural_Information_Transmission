# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 12:26:21 2015

@author: malte
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 10:15:00 2015

@author: maltimore
"""

import model_neuron
import matplotlib.pyplot as plt
import numpy as np
#import sys
import datetime
#import cPickle as pickle



#####################################
# Set constant variables
N_timesteps = 700
N_per_group = 100
N_groups    = 10
N_neurons   = N_per_group * N_groups
my_linewidth = .2
dt = .0001
input_synchronisation = 0
input_spikes = 70


#####################################
# Create some vectors and matrices
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

    
#Calculate number of output spike and its variance
def calculate_output_properties(neuronlist):
    spiketimematrix = np.zeros((N_neurons,N_timesteps))
    for i in range(N_neurons):
        for j in range(len(neuronlist[i].spiketime)):
            spiketimematrix[i][neuronlist[i].spiketime[j]/dt] = neuronlist[i].spiketime[j]*1000         
    
    i=N_groups-1
    # ATTENTION! I NOW HARDCODED THE CUTOFF VALUES
    startcutoff = 53
    endcutoff   = 60
    
    hilf=spiketimematrix[(i)*N_per_group  :  (i+1)*N_per_group][:,startcutoff/dt:endcutoff/dt]
    hilf=np.reshape(hilf,(np.shape(hilf)[0]*np.shape(hilf)[1]))
    hilf=hilf[hilf!=0.0]
    variance_in_group=(np.std(hilf))
    spikes_in_group=len(hilf)
    
   # return eventmatrix, spiketimematrix
    #Return last group properties
    return spikes_in_group, variance_in_group

def rasterplot():
    fig = plt.figure()
    spikes = np.zeros((N_neurons,N_timesteps))
    for i in range(N_neurons):
        for j in range(len(neuronlist[i].spiketime)):
            spikes[i][j] = neuronlist[i].spiketime[j]*1000 
            ax = plt.gca()
    for ith, trial in enumerate(spikes):
        plt.vlines(trial, ith + .5, ith + 1.5, color='k')
        plt.ylim(.5, len(spikes) + .5) 
    ax.invert_yaxis()
    plt.title('Raster plot')
    plt.xlabel('time [ms]')
    plt.ylabel('number of neuron')
    plt.xlim(0,70)
    fig.show()
    
def simulate(N_timesteps, neuronlist, initial_spike_neurons, initial_spike_times):
    N_neurons = len(neuronlist)
    volt_matrix = np.empty((N_neurons, N_timesteps))
    current_artificial_neuron = 0
    
    j = 0
    for i in np.arange(N_timesteps):
        
        current_timestep = i * dt
        if i % 50 == 0:
            print "Current timestep: " + str(current_timestep)
        
        # check whether initial spike times is an array
        # and therefore the user is not asking for zero input spikes
        while ((current_timestep <= initial_spike_times[current_artificial_neuron]) \
        and (initial_spike_times[current_artificial_neuron] <= current_timestep + dt)):
            
            # check whether there were zero input neurons requested
            # in that case, break this loop and go on simulating
            if len(initial_spike_neurons) == 0: 
                break

            initial_spike_neurons[current_artificial_neuron].fire()
            if current_artificial_neuron < len(initial_spike_times)-1:
                current_artificial_neuron += 1
            else:
                break
                
        for neuron in neuronlist:
            neuron.update()
            volt_matrix[j,i] = neuron.get_voltage()
            j += 1
        j = 0
        
    if current_artificial_neuron == 0:
        print "In the last simulation, 0 initial input spikes were fired"
    else:
        print "In the last simulation, " + str(current_artificial_neuron + 1) + \
        ' initial input spikes were fired'
    
    return volt_matrix
    
def get_artificial_neurons(neuronlist, N_per_group, synchronization, N_spikes):
    artificial_neuron_list = []    
    for i in np.arange(N_spikes):
        artificial_input_neuron = model_neuron.Neuron(-(i+1), dt)
        artificial_input_neuron.set_output_connections(neuronlist[0:N_per_group])
        artificial_neuron_list.append(artificial_input_neuron)
    
    if N_spikes == 0:
        print " ATTENTION! Zero input spikes were slected."
        # in case there were zero input neurons, we have to return an
        # empty  neuron list, but one spike time. This is slightly
        # unintuitive, but at the moment it has to be like this because
        # a different function won't work otherwise.
        # This one spike time is ignored anyways.
        return [], [0]
    if synchronization == 0:
        initial_spike_times = np.zeros((N_spikes))
        return artificial_neuron_list, initial_spike_times
        
    if synchronization != 0:    
        initial_spike_times = np.random.normal(scale = synchronization, size = N_spikes)  
        initial_spike_times = initial_spike_times  - np.amin(initial_spike_times)
        initial_spike_times = initial_spike_times / 1000
        initial_spike_times = np.sort(initial_spike_times)
        return artificial_neuron_list, initial_spike_times
    

 
#####################################
# Run simulation
 
  
  
neuronlist = organizeNeurons(N_per_group,N_groups)
artificial_neurons, initial_spike_times = get_artificial_neurons(neuronlist, N_per_group, input_synchronisation, input_spikes)
volt_matrix = simulate(N_timesteps, neuronlist, artificial_neurons, initial_spike_times)
    
    
# plot voltage for all simulated neurons
plt.figure(figsize=(15,20))
for i in np.arange(N_neurons):    
    plt.plot(np.linspace(0,N_timesteps*dt*1000,N_timesteps),volt_matrix[i], linewidth = my_linewidth)
plt.xlabel('time [ms]')
plt.ylabel('voltage [v]')
   
   #a_out, sig_out=calculate_output_properties()
   
rasterplot()

#       
########################################################################
# Single voltage_plot PSP

dt = .0001
N_timesteps = 200

voltage_vec2  = np.empty(N_timesteps)
voltage_vec3 = np.empty(N_timesteps)

neuron1 = model_neuron.Neuron(1,dt)
neuron2 = model_neuron.Neuron(2,dt)
neuron3 = model_neuron.Neuron(3,dt)

neuron2.switch_noise_off()
neuron2.set_voltage(-.07)
neuron3.switch_noise_off()
neuron3.set_voltage(-.07)
neuron3.set_g_syn(5.45e-10)

neuron1.set_output_connections([neuron2, neuron3])
neuron1.fire()

for i in np.arange(N_timesteps):
    neuron2.update()
    voltage_vec2[i] = neuron2.get_voltage()
    neuron3.update()
    voltage_vec3[i] = neuron3.get_voltage()

# plot voltage for all simulated neurons
plt.figure()
plt.plot([0,N_timesteps*dt],[-.06989,-.06989], linewidth = 2, label= 'PSP .11 mV', color='r')
plt.plot([0,N_timesteps*dt],[-.06986,-.06986], linewidth = 2, label= 'PSP .14 mV', color='b')
plt.plot(np.linspace(0,N_timesteps*dt,num=len(voltage_vec2)), voltage_vec2,         color='b')
plt.plot(np.linspace(0,N_timesteps*dt,num=len(voltage_vec3)), voltage_vec3,         color='r')
plt.xlabel('time [s]')
plt.ylabel('voltage [V]')
plt.legend()
plt.xlim(0,N_timesteps*dt)


###############################################################################
# A couple of noise plots

dt = .0001
N_timesteps = 200
N_neurons = 20
neuronlist = []

voltage_mat  = np.empty((N_neurons, N_timesteps))

for i in np.arange(N_neurons):
    neuronlist.append(model_neuron.Neuron(i,dt))

for i in np.arange(N_timesteps):
    for idx, neuron in enumerate(neuronlist):
        neuron.update()
        voltage_mat[idx,i]= neuron.get_voltage()
        
        
        
# plot voltage for all simulated neurons
plt.figure()    
for i in np.arange(N_neurons):
    plt.plot(np.linspace(0,N_timesteps*dt,num=len(voltage_vec2)), voltage_mat[i,:])

plt.xlabel('time [s]')
plt.ylabel('voltage [V]')
plt.legend()
plt.xlim(0,N_timesteps*dt)
