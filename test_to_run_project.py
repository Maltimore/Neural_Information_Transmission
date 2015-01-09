# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 10:15:00 2015

@author: maltimore
"""
import model_neuron
import matplotlib.pyplot as plt
import numpy as np


N = 500

neuron1 = model_neuron.Neuron(3)

voltage_vec = np.empty(N)

for i in np.arange(N):
    neuron1.set_external_current(2e-9)
    neuron1.update()
    voltage_vec[i] = neuron1.get_voltage()


plt.plot(voltage_vec)