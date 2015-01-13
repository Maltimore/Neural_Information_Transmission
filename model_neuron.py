# This is the code for the neuron object! Not so much here yet :)

import numpy as np
import model_synapse

class Neuron:
    'This is a model neuron class that takes input and gives output'    
    


    remaining_refractory = 0 # remaining absolute refractory period
    

    def __init__(self, neuronnumber):
        # setting class parameters
        self.params = {\
        'V': -.07,\
        't': 0,\
        'E_m': -.07,\
        'refractory_p': .001,\
        'tau_K': .015,\
        'tau_m': .01,\
        'tau_syn': 0.01,\
        'R_m': 10e7,\
        'g_max': 5e-8,\
        'g_K': 36e-9,\
        'E_K': -.077,\
        'E_syn': 0,\
        't_max': .07,\
        'dt': .0001,\
        'I_e': 0,\
        'threshold': -.055\
        }
        self.this_neurons_number = neuronnumber
        
    # This is the Euler step function, which will only compute one 
    # next timestep, therefore it is rather slim.
    def eulerstep(self,f_func, startvalue, params):
        return startvalue + f_func(startvalue, self.params) * params['dt']
        
    def dV_dt(self,V, params):
        tau_m = params['tau_m']
        E_m = params['E_m']
        E_K = params['E_K']
        R_m = params['R_m']
        I_e = params['I_e']
        g_K = params['g_K']
            
        g_K = self.eulerstep(self.gK,g_K,self.params)        
        I_K = g_K*(V-E_K)        
        

    
        I_syn = self.get_synaptic_input(V)
        
        return (-V + E_m - R_m * I_syn + R_m * I_e - I_K) / tau_m

    def get_synaptic_input(self, V):
        I_syn = 0
        for synapse in self.synapse_list:
            I_syn += synapse.get_I_syn(V)   
        return I_syn
    
    def gK(self, g_K, params):
        tau_K = params['tau_K']
        return -g_K/tau_K
    
                   
        
    def get_voltage(self):
        return self.params['V']
        
        
    def set_external_current(self,new_Ie):
        self.params['I_e'] = new_Ie
        
    
    def get_external_current(self):
        return self.params['I_e']
        
        
    def set_input_connections(self, neuron_numbers, E_syn_vec):
        self.input_connections = neuron_numbers
        
        #list that will hold input synapses:
        self.synapse_list = []
        
        for i in np.arange(0,len(neuron_numbers)):
            self.synapse_list.append(model_synapse.Synapse(self.params['g_max'], self.params['tau_syn'], \
            self.params['dt'], E_syn_vec[i]))
    
    def set_output_connections(self, outputneurons):
        self.postsynaptic_neurons = outputneurons
    
    
    # receive synaptic input from other neuron        
    def receive_synaptic_input(self, ext_neuron_number):    
        # determine index of inputting neuron
    #  LATER CHANGE THIS TO GET REAL INDEX
        index = self.input_connections[self.input_connections == ext_neuron_number]
        # start the synapse corresponding to inputting neuron
        self.synapse_list[index].start()
            
            
    def fire(self):
        for neuron in self.postsynaptic_neurons:
            neuron.receive_synaptic_input(self.this_neurons_number)
        
        # disable all input synapses    
        for synapse in self.synapse_list:
            synapse.disable()
            
        self.params['V'] = -.07
        

        
            
            
    # this function is called by the test script to compute one timestep further    
    def update(self):
        
        # check whether the neuron is in absolute refractory period, then set input currents to zero
        # LATER HERE WE SHOULD JUST SET THE INPUT SYNAPSES TO ZERO     
        if self.remaining_refractory > 0:
            self.remaining_refractory -= self.params['dt'] # update remaining refractory

            
        self.params['V'] = self.eulerstep(self.dV_dt,self.params['V'],self.params)
        
        for synapse in self.synapse_list:
            synapse.update()
        
        if self.params['V'] > self.params['threshold']:
            self.remaining_refractory = self.params['refractory_p']
            self.fire()