# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 12:24:06 2015

@author: tabea
"""
import numpy as np
import matplotlib.pyplot as plt
import noise

tau_OUP = .5
sigma = 10**-9

dt = 0.1

Imu = 0
noise1 = noise.Noise(Imu,dt,tau_OUP,sigma)

N = 1000
I = np.empty(N)

I[0] = Imu

for i in range(1,N):
    I[i] = noise1.get_noise()
plt.plot(range(N),I)
