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
    def calculate_output_properties(neuronlist):
        spiketimematrix = np.zeros((N_neurons,N_timesteps))
        for i in range(N_neurons):
            for j in range(len(neuronlist[i].spiketime)):
                spiketimematrix[i][neuronlist[i].spiketime[j]/dt] = neuronlist[i].spiketime[j]*1000         
        
        i=N_groups-1
        startcutoff=N_groups*0.005
        endcutoff=startcutoff + 0.015
        
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
    
#    plt.figure()
#    plt.hist(initial_spike_times)
    
    # Plot Voltage for all simulated neurons
#    plt.figure(figsize=(15,20))
#    for i in np.arange(N_neurons):    
#        plt.plot(np.linspace(0,N_timesteps*dt*1000,N_timesteps),volt_matrix[i], linewidth = my_linewidth)
#    plt.xlabel('time [ms]')
#    plt.ylabel('voltage [V]')
    
    #a_out, sig_out=calculate_output_properties()
    
    #rasterplot()
    
    
    spikes, std = calculate_output_properties(neuronlist)
    
    return spikes, std


startingvalues =    [[0,0], [50,0]]
simsteps=2
repetitions = 2

def create_initial_output_vector(startingvalues, repetitions):
    outputvec = []    
    for i in np.arange(len(startingvalues)):
        for k in np.arange(repetitions):
            outputvec.append(startingvalues[i])
            
    return outputvec


def phase_plane_plot(startingvalues, simsteps):
     k=0
     l=1
     for spikes_in, synch_in in startingvalues:

         j=0
         spikes_out, synch_out = spikes_in, synch_in
         Outputs[0,k]= spikes_out
         Outputs[0,l]= synch_out
         for i in range(simsteps):
            j+=1
            spikes_out, synch_out= overallfunction(spikes_out,synch_out)
            Outputs[j,k]=spikes_out
            Outputs[j,l]= synch_out

	 with open('Zwischenspeicherung.txt','wb') as f:
	    pickle.dump(Outputs,f)

         k+=2
         l+=2
     return Outputs



startingvec                    = create_initial_output_vector(startingvalues, repetitions)
Outputs                        = np.zeros((simsteps+1,2*len(startingvec)))


Data                           = phase_plane_plot(startingvec, simsteps)

## plotting


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
    #
    #N = 100000
    ## find best I_mu
    #I_mu = np.arange(2.7e-10,2.74e-10,1e-13)
    #firing_rate = np.empty(len(I_mu))
    #
    #
    #i = 0
    #for I in I_mu:
    #    testneuron = model_neuron.Neuron(0,dt)
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
    
    # delete all objects



#if os.path.exists('simulation%s.txt'%simnr):
#    with open('simulation%s.txt'%simnr,'rb') as f:
#        variable=pickle.load(f)
#        print variable
#    variable.append(variable2)
#else:
#    variable = []
#    variable.append(variable2)


with open('simulation.txt','wb') as f:
    pickle.dump(repetitions,f)
    pickle.dump(Data,f)
