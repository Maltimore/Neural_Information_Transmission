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
N_timesteps = 200
N_per_group = 100
N_groups    = 3
N_neurons   = N_per_group * N_groups
input_spikes = 90
input_synchronisation = 3
my_linewidth = .2
dt = .0001

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
def calculate_output_properties():
    eventmatrix = np.zeros((N_neurons,N_timesteps))
    spiketimematrix = np.zeros((N_neurons,N_timesteps))
    for i in range(N_neurons):
        for j in range(len(neuronlist[i].spiketime)):
            eventmatrix[i][neuronlist[i].spiketime[j]/dt] = 1
            spiketimematrix[i][neuronlist[i].spiketime[j]/dt] = neuronlist[i].spiketime[j]/dt
    spikes_in_group=np.zeros(N_groups)
    variance_in_group=np.zeros(N_groups)
     
    k=0
    for  i in range (N_groups): 
        spikes_in_group[k]=(np.sum(eventmatrix[(i)*N_per_group  :  (i+1)*N_per_group]))
        hilf=spiketimematrix[(i)*N_per_group  :  (i+1)*N_per_group]
        hilf=np.reshape(hilf,(np.shape(hilf)[0]*np.shape(hilf)[1]))
        variance_in_group[k]=(np.std(np.trim_zeros(hilf)))
        k+=1
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
        inputspikes_this_timestep = len(initial_spike_times[initial_spike_times == current_timestep])
        if  inputspikes_this_timestep != 0:
            for k in np.arange(inputspikes_this_timestep):
                initial_spike_neurons[current_artificial_neuron].fire()
                current_artificial_neuron += 1
                
        for neuron in neuronlist:
            neuron.update()
            volt_matrix[j,i] = neuron.get_voltage()
            j += 1
        j = 0
        
    return volt_matrix
    
def get_artificial_neurons(neuronlist, N_per_group, synchronization, N_spikes):
    artificial_neuron_list = []    
    for i in np.arange(N_spikes):
        artificial_input_neuron = model_neuron.Neuron(-(i+1), dt)
        artificial_input_neuron.set_output_connections(neuronlist[0:N_per_group])
        artificial_neuron_list.append(artificial_input_neuron)
    
    if synchronization == 0:
        initial_spike_times = np.zeros((N_spikes))
        return artificial_neuron_list, initial_spike_times
        
    if synchronization != 0:    
        initial_spike_times = np.random.normal(scale = synchronization, size = N_spikes)
        initial_spike_times = np.round(initial_spike_times, decimals = 1)
        initial_spike_times = initial_spike_times  - np.amin(initial_spike_times)
        initial_spike_times = initial_spike_times / 1000
        return artificial_neuron_list, initial_spike_times
    

 
#####################################
# Run simulation

neuronlist = organizeNeurons(N_per_group,N_groups)
artificial_neurons, initial_spike_times = get_artificial_neurons(neuronlist, N_per_group, input_synchronisation, input_spikes)
volt_matrix = simulate(N_timesteps, neuronlist, artificial_neurons, initial_spike_times)


# Plot Voltage for all simulated neurons
plt.figure()
for i in np.arange(N_neurons):    
    plt.plot(np.linspace(0,N_timesteps*dt*1000,N_timesteps),volt_matrix[i], linewidth = my_linewidth)
plt.xlabel('time [ms]')
plt.ylabel('voltage [V]')

a_out, sig_out=calculate_output_properties()

#rasterplot()

############### CODE TO FIND PARAMTERS ################################
#test_timesteps = 200
#tau_syn_vec = np.linspace(.0001,.001,1000)
#g_syn_vec   = np.linspace(1e-10,1e-9,1000)
#maximum_time_vec = np.empty((len(tau_syn_vec)))
#maximum_volt_vec = np.empty((len(tau_syn_vec)))
#
#for i in np.arange(len(tau_syn_vec)):
#    neuron_A = model_neuron.Neuron(9998, dt)
#    neuron_A.switch_noise_off()
#    neuron_A.set_tau_syn(.0003315)
#    neuron_A.set_g_syn(g_syn_vec[i])
#    neuron_B = model_neuron.Neuron(9999, dt)
#    neuron_B.set_output_connections([neuron_A])
#    neuron_B.fire()
#    
#    voltagevec = np.empty((test_timesteps))
#    timescale  = np.arange(test_timesteps)
#    for j in timescale:
#        neuron_A.update()
#        voltagevec[j] = neuron_A.get_voltage()
#    
#    maximum_time_vec[i] = np.argmax(voltagevec) * dt * 1000 - 5
#    maximum_volt_vec[i] = np.amax(voltagevec) * 1000 + 70 # in mV
#
#
# 
#plt.figure()   
#plt.plot(g_syn_vec, maximum_volt_vec)
#plt.xlabel('g_syn')
#plt.ylabel('max_volt [mV]')
#
#
#neuron_A = model_neuron.Neuron(9998, dt)
#neuron_A.switch_noise_off()
#neuron_A.set_tau_syn(.0003315)
#neuron_A.set_g_syn(7e-10)
#neuron_B = model_neuron.Neuron(9999, dt)
#neuron_B.set_output_connections([neuron_A])
#neuron_B.fire()
#
#for j in timescale:
#    neuron_A.update()
#    voltagevec[j] = neuron_A.get_voltage()
#    
#plt.figure()
#ax = plt.gca()
#plt.plot(timescale * dt * 1000, voltagevec * 1000)
#ax.ticklabel_format(useOffset=False)
#plt.xlabel('time in [ms]')
#plt.ylabel('voltage in [mV]')

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