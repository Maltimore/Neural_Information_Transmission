# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 16:15:36 2015

@author: maltimore
"""

import noise


I_mu = 2.703e-10
sigma = 10e-7
tau_noise = .005
dt = .0001


noiseobj = noise.Noise(I_mu, dt, tau_noise, sigma)

noisevec = np.empty(100)
time = np.arange(len(noisevec))

for i in time:
    noisevec[i] = noiseobj.get_noise()

plt.figure()
plt.plot(time, noisevec)