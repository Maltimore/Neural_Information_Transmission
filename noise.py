# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 09:24:54 2015

@author: tabea
"""
import numpy as np

class Noise:
    
    def __init__(self,Imu,dt,tau_OUP,sigma):
        # setting class parameters
        self.step = 0
        self.I = []
        self.sigma = sigma
        self.tau_OUP = tau_OUP
        self.dt = dt
        self.Imu = Imu
    
    def get_noise(self):
        self.I.append(self.Imu)
        self.step += 1 
        
        self.I.append((self.I[self.step-1]+(self.Imu-self.I[self.step-1])/ \
        self.tau_OUP+self.sigma*np.random.normal(loc=0.0,scale=1.0))*self.dt)
                
        return self.I[self.step]
