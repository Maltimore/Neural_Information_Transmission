# This is the code for the neuron object.

import model_synapse
import noise

class Neuron:
    'This is a model neuron class that takes input and gives output'   

    def __init__(self, neuronnumber, dt):
        # setting class parameters
        self.params = {\
        'V': -.057,\
        'E_m': -.07,\
        'total_refractory_period': .001,\
        'tau_K': .005,\
        'tau_m': .01,\
        'tau_syn': 0.000335,\
        'R_m': 10e7,\
        'g_max': 6.99e-10,\
        'g_K': 0,\
        'E_K': -.077,\
        'E_syn': 0,\
        't_max': .07,\
        'I_e': 0,\
        'threshold': -.055,\
        'dt': dt,\
        'g_K_add_term': 5e-9\
        }
        self.I_mu = 8.89e-11
        self.sigma = 1e-7
        self.tau_noise = .005
        self.postsynaptic_neurons = []
        self.input_neuron_numbers = []
        self.this_neurons_number = neuronnumber
        self.spiketime = []
        self.noiseobj = noise.Noise(self.I_mu, dt, self.tau_noise, self.sigma)
        self.synapse_list = []
        self.t = 0
        self.number_of_spikes = 0
        self.noise_switch = 1
        self.remaining_refractory = 0 # remaining absolute refractory period
        
        
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
        
        # compute new g_K value, save it in params (and use further down in this function)
        g_K = self.eulerstep(self.gK_dt, g_K ,self.params)  
        self.params['g_K'] = g_K
        
        # compute K current, synaptic current and noise
        I_K = g_K*(V-E_K)        
        I_syn = self.get_synaptic_current(V)
        I_noise = self.noiseobj.get_noise()
        
        # switching off the noise if the noise swtich is off (= zero)
        if self.noise_switch == 0:
            I_noise = 0
            
        return ((-V + E_m) + R_m * (I_e - I_K - I_syn + I_noise)) / tau_m

    def get_synaptic_current(self, V):
        I_syn = 0
        for synapse in self.synapse_list:
            I_syn += synapse.get_I_syn(V)   
        return I_syn

    
    def gK_dt(self, g_K, params):
        tau_K = params['tau_K']
        return -g_K/tau_K
            
            
    def fire(self):
        # increment total number of spikes of this neuron
        self.number_of_spikes += 1
        
        # store spike time in list
        self.spiketime.append(self.t)
        
        # set refractory period
        self.remaining_refractory = self.params['total_refractory_period']
        
        # activate synapses in postsynaptic neurons
        for neuron in self.postsynaptic_neurons:
            neuron.activate_synapse(self.this_neurons_number)
        
        # disable all input synapses on this neuron  
        for synapse in self.synapse_list:
            synapse.disable()
        
        # reset Voltage to -.07 Volt and increase K-conductance
        self.params['V'] = -.07
        self.params['g_K'] = self.params['g_K'] + self.params['g_K_add_term']
        


    ##############################################################################################
    # UPDATE FUNCTION ############################################################################        
    # this function is called by the script to simulate one timestep further    
    def update(self):
        # check whether in this timestep the neuron crossed firing threshold
        if self.params['V'] > self.params['threshold']:
            self.fire()
            
        # update remaining refractory period  
        if self.remaining_refractory > 0:
            self.remaining_refractory -= self.params['dt'] # update remaining refractory

        # update all synapses    
        for synapse in self.synapse_list:
            synapse.update()
        
        # last but not least, compute new voltage and update time
        self.params['V'] = self.eulerstep(self.dV_dt,self.params['V'],self.params)
        self.t += self.params['dt']
        


###################################################################################################
########### GETTER AND SETTER METHODS TO BE USED FROM OUTSIDE THIS CLASS ##########################
###################################################################################################

    # receive synaptic input from other neuron        
    def activate_synapse(self, ext_neuron_number):     
        if self.remaining_refractory <= 0:
            # determine index of inputting neuron
            index = self.input_neuron_numbers.index(ext_neuron_number)
            self.synapse_list[index].start()
            
            
    def get_output_connections(self):
        return self.postsynaptic_neurons
        
       
    def get_firing_rate(self):
        return self.number_of_spikes / self.t

        
    def set_I_mu(self, Imu, sigma):
        self.noiseobj = noise.Noise(Imu, self.params['dt'], self.tau_noise, sigma)

    
    def switch_noise_on(self):
        self.noise_switch = 1

    
    def switch_noise_off(self):
        self.noise_switch = 0

        
    def get_voltage(self):
        return self.params['V']

                
    def set_external_current(self,new_Ie):
        self.params['I_e'] = new_Ie

            
    def get_external_current(self):
        return self.params['I_e']
        
        
    def set_input_connection(self, inputneuron):
        #create a synapse
        synapse = model_synapse.Synapse(self.params['g_max'], self.params['tau_syn'], \
                                        self.params['dt'])      
        self.synapse_list.append(synapse)
        self.input_neuron_numbers.append(inputneuron.get_neuron_number())
            
    
    def set_output_connections(self, outputneurons):
        for outputneuron in outputneurons:
            outputneuron.set_input_connection(self)
            self.postsynaptic_neurons.append(outputneuron)


    def get_neuron_number(self):
        return self.this_neurons_number
    
    
    def get_input_synapses(self):
        return self.synapse_list
        
    def set_tau_syn(self, new_tau_syn):
        self.params['tau_syn'] = new_tau_syn
        
    def set_g_syn(self, new_g_syn):
        self.params['g_max'] = new_g_syn
