# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 16:52:43 2015

@author: tabea
"""

import numpy as np
import matplotlib.pyplot as plt
import noise
import model_neuron

#tau_OUP = .005
#sigma = 1e-7
#
#dt = 0.0001
#
#Imu = 1.45e-10
#noise1 = noise.Noise(Imu,dt,tau_OUP,sigma)
#
#N = 1000
#I = np.empty(N)
#
#
#for i in range(N):
#    I[i] = noise1.get_noise()
#plt.plot(range(N),I)
#plt.show()

N = 10000000
dt = 0.0001
#
#
#
#sigmas = 1e-8
#Imus = np.linspace(1.439e-10,1.442e-10,100)
#
#fr = np.empty((len(Imus)))
#
#for i in Imus:
#    neuron1 = model_neuron.Neuron(1,dt)
#    neuron1.set_I_mu(i,sigmas)
#    for j in range(N):
#        neuron1.update()
#    fr[i] = neuron1.get_firing_rate()
# 
#plt.figure()
#plt.plot(Imus,fr)
#plt.show()

voltage = np.empty((N))
neuron1 = model_neuron.Neuron(1,dt)
for j in range(N):
    neuron1.update()
    voltage[j] = neuron1.get_voltage()
    
#plt.plot(np.linspace(0,N*dt*1000,N),voltage)
#plt.show()
print neuron1.get_firing_rate()
    
    
    

