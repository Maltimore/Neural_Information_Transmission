# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 09:42:08 2015

@author: maltimore
"""
import numpy as np

class Synapse:
    'This is the synapse class that models postsynaptic synapses'
    
    def __init__(self, g_max, tau_syn, dt):
        self.t = 0
        self.g_max = g_max
        self.tau_syn = tau_syn
        self.E_syn = 0 # Volt
        self.dt = dt
        self.disabled = 1
        self.remaining_delay = 0
    
    def start(self):
        self.t = 0
        self.disabled = 0
        self.remaining_delay = .005

    
    def disable(self):
        self.disabled = 1
        

    
    def enable(self):
        self.disabled = 0
        self.remaining_delay = .005
        
    def g_syn_dt(self,t, g_max, tau_syn):
        return float(self.g_max) * t / self.tau_syn * np.exp(-t/self.tau_syn)
        
    def get_I_syn(self, V):
        if self.disabled == 1:
            return 0
        if self.remaining_delay >= 0:
            return 0
        else:
            return self.g_syn_dt(self.t, self.g_max, self.tau_syn) * (V - self.E_syn)
        
    def update(self):
        if self.remaining_delay > 0:
            self.remaining_delay -= self.dt
        else:
            self.t += self.dt
        