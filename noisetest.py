# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 16:52:43 2015

@author: tabea
"""

import numpy as np
import matplotlib.pyplot as plt
import noise

tau_OUP = .5
sigma = 10**-7

dt = 0.0001

Imu = 2.703e-10
noise1 = noise.Noise(Imu,dt,tau_OUP,sigma)

N = 100
I = np.empty(N)


for i in range(N):
    I[i] = noise1.get_noise()
plt.plot(range(N),I)
plt.show()