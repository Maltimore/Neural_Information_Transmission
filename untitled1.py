# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 12:03:23 2015

@author: tabea
"""

import model_neuron
import matplotlib.pyplot as plt
import numpy as np

dt = .0001
N_timesteps = 200

voltage_vec = np.empty(N_timesteps)

neuron1 = model_neuron.Neuron(1,dt)
neuron2 = model_neuron.Neuron(2,dt)

neuron1.switch_noise_off()

neuron2.set_output_connections([neuron1])
neuron2.fire()

for i in np.arange(N_timesteps):
    neuron1.update()
    voltage_vec[i] = neuron1.get_voltage()

# plot voltage for all simulated neurons
plt.figure(figsize=(15,20))
plt.plot([0,200],[-.06989,-.06989])
plt.plot(voltage_vec)
plt.xlabel('time [ms]')
plt.ylabel('voltage [v]')