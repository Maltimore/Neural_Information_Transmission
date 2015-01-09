# This is the code for the neuron object! Not so much here yet :)

import numpy as np

class Neuron:
    'This is a model neuron class that takes input and gives output'    
    
    params = {\
    'V': -.07,\
    't': 0,\
    'E_m': -.07,\
    'refractory_p': .001,\
    'tau_K': .015,\
    'tau_m': .01,\
    'tau_syn': 0.01,\
    'R_m': 10**7,\
    'g_max': 5e-8,\
    'g_K': 36,\
    'E_K': -.077,\
    'E_syn': 0,\
    't_max': .07,\
    'dt': .0001,\
    'I_e': 0,\
    'threshold': -.055\
    }
    
    remaining_refractory = 0 # remaining absolute refractory period
    
    def __init__(self, projections):
        # setting class parameters
        self.projections = projections
        
    def get_projections(self):
        return "My projections are: " + str(self.projections)
        
    # This is the Euler step function, which will only compute one 
    # next timestep, therefore it is rather slim.
    def eulerstep(self,f_func, startvalue, params):
        return startvalue + f_func(startvalue, self.params) * params['dt']
        
    def dV_dt(self,v, params):
        tau_m = params['tau_m']
        E_m = params['E_m']
        R_m = params['R_m']
        I_e = params['I_e']
        t  = params['t']
        
        return (-v + E_m - R_m * self.I_syn(v, t, params) + R_m * I_e) / tau_m
    
    def I_syn(self,v, t, params):
        E_syn = params['E_syn']
        
        return self.g_syn(t, params['g_max'], params['tau_syn']) * (v - E_syn)
    
    def g_syn(self,t, g_max, tau_syn):
        return g_max * t / tau_syn * np.exp(-t/tau_syn)
        
    def update(self):
        print(self.remaining_refractory)
        # check whether the neuron is in absolute refractory period, then set input currents to zero
        if self.remaining_refractory > 0:
            self.remaining_refractory -= self.params['dt'] # update remaining refractory
            self.params['I_e'] = 0                         # set input to zero
            print(self.remaining_refractory)
        self.params['V'] = self.eulerstep(self.dV_dt,self.params['V'],self.params)
        
        if self.params['V'] > self.params['threshold']:
            self.remaining_refractory = self.params['refractory_p']
            
        
    def get_voltage(self):
        return self.params['V']
        
    def set_external_current(self,new_Ie):
        self.params['I_e'] = new_Ie
    
    def get_external_current(self):
        return self.params['I_e']