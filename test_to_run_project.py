# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 10:15:00 2015

@author: maltimore
"""

import model_neuron
import matplotlib.pyplot as plt
import numpy as np
import sys
import datetime
import cPickle as pickle


# Here I define that all output will be written to a history file
sys.stdout = open("History.log", "w")

def overallfunction(input_spikes1,sigma):

    print "Current time is: " + str(datetime.datetime.now())
    print "RUNNING SIMULATION with n_spikes " + str(input_spikes1) \
    + " and sigma " + str(sigma)
    #####################################
    # Set constant variables
    N_timesteps = 700
    N_per_group = 100
    N_groups    = 10
    N_neurons   = N_per_group * N_groups
    input_spikes = input_spikes1
    input_synchronisation = sigma
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
#    def calculate_output_properties(neuronlist):
#        spiketimematrix = np.zeros((N_neurons,N_timesteps))
#        for i in range(N_neurons):
#            for j in range(len(neuronlist[i].spiketime)):
#                spiketimematrix[i][neuronlist[i].spiketime[j]/dt] = neuronlist[i].spiketime[j]*1000         
#        
#        i=N_groups-1
#        # ATTENTION! I NOW HARDCODED THE CUTOFF VALUES
#        startcutoff = 0.053
#        endcutoff   = 0.060
#        
#        hilf=spiketimematrix[(i)*N_per_group  :  (i+1)*N_per_group][:,startcutoff/dt:endcutoff/dt]
#        hilf=np.reshape(hilf,(np.shape(hilf)[0]*np.shape(hilf)[1]))
#        hilf=hilf[hilf!=0.0]
#        variance_in_group=(np.std(hilf))
#        spikes_in_group=len(hilf)
#        
#       # return eventmatrix, spiketimematrix
#        #Return last group properties
#        return spikes_in_group, variance_in_group

    def calculate_output_properties(neuronlist):
    
        spiketimematrix = np.zeros((N_neurons,N_timesteps))
        
        for i in range(N_neurons):
        
            for j in range(len(neuronlist[i].spiketime)):
            
                spiketimematrix[i][neuronlist[i].spiketime[j]/dt] = neuronlist[i].spiketime[j]*1000
            
             
            
        spikes_in_group, variance_in_group = np.zeros(N_groups), np.zeros(N_groups)
        
        for i in range(N_groups):
        
            startcutoff = 0.005*i
            
            endcutoff = startcutoff + 0.020
            
            hilf=spiketimematrix[(i)*N_per_group : (i+1)*N_per_group][:,startcutoff/dt:endcutoff/dt]
            
            hilf=np.reshape(hilf,(np.shape(hilf)[0]*np.shape(hilf)[1]))
            
            hilf=hilf[hilf!=0.0]
            
            variance_in_group[i]=(np.std(hilf))
            
            spikes_in_group[i]=len(hilf)
        
         
        
        #Return last group properties
        
        return spikes_in_group, variance_in_group



    
        
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
        
    ## plotting
     
    #####################################
    # Run simulation
     
      
      
    neuronlist = organizeNeurons(N_per_group,N_groups)
    artificial_neurons, initial_spike_times = get_artificial_neurons(neuronlist, N_per_group, input_synchronisation, input_spikes)
    volt_matrix = simulate(N_timesteps, neuronlist, artificial_neurons, initial_spike_times)
    

    
    spikes, std = calculate_output_properties(neuronlist)
    
    return spikes, std


startingvalues =    [[40,0], [50,0], [60,0], [80,0], [100,0],
                             [50,3], [60,3], [80,3], [100,3]]

repetitions  = 5

def create_initial_output_vector(startingvalues, repetitions):
    outputvec = []    
    for i in np.arange(len(startingvalues)):
        for k in np.arange(repetitions):
            outputvec.append(startingvalues[i])
            
    return outputvec


def phase_plane_plot(startingvalues):
     Outputs   = np.zeros((11,2*len(startingvec)))
     
     k=0
     l=1
     for spikes_in, synch_in in startingvalues:

         Outputs[0,k]= spikes_in
         Outputs[0,l]= synch_in
         spikes_out, synch_out= overallfunction(spikes_in,synch_in)
         
         Outputs[1:,k]=spikes_out
         Outputs[1:,l]= synch_out

         with open('Zwischenspeicherung.txt','wb') as f:
             pickle.dump(Outputs,f)

         k+=2
         l+=2
     return Outputs


startingvec         = create_initial_output_vector(startingvalues, repetitions)
Data                = phase_plane_plot(startingvec)


with open('simulation.txt','wb') as f:
    pickle.dump(repetitions,f)
    pickle.dump(Data,f)
