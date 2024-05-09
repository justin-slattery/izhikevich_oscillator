'''
Izhikevich neuron and neural network classes for simulating spiking neural networks
'''

import numpy as np

class IzhikevichNeuron:
    def __init__(self, v, a, b, c, d):
        self.Voltage = v    # membrane potential
        self.u = b*v        # membrane recovery variable
        # Neural parameters
        self.a = a          # time scale of the recovery variable u; smaller values = slower recovery
        self.b = b          # sensitivity of the recovery variable u; greater values couple v and u more strongly 
                            # resulting in possible subhtreshold oscillations and low-threshold spiking
        self.c = c          # after-spike reset value of the membrane potential v
        self.d = d          # after-spike reset of the recovery variable u

    def step(self,dt,Input):
        # Computing change in membrane potential based on Izhikevich equation
        dVdt = (0.04 * (self.Voltage**2)) + (5 * self.Voltage) + 140.0 - self.u + Input
        # Change in recovery variable
        dudt = self.a * ((self.b * self.Voltage) - self.u)
        # Updates the membrane potential
        self.Voltage += dt * dVdt
        # Updates the recovery variable
        self.u += dt * dudt
        # If threshold is reached, reset the membrane potential and recovery variable
        if self.Voltage >= 30.0:
            self.Voltage = self.c
            self.u = self.u + self.d

class IzhikevichNetwork:
    def __init__(self, size):
        self.Size = size                        # number of neurons in the network
        self.Voltages = np.zeros(size) - 65     # vector for membrane potentials; resting potential = -65 mV
        self.u = np.zeros(size)                 # vector for membrane recovery variables
        # Neural parameters
        self.a = np.ones(size)                  # vector for a: time scale of the recovery variable u; smaller values = slower recovery
        self.b = np.zeros(size)                 # vector for b: sensitivity of the recovery variable u; greater values couple v and u more strongly 
                                                # resulting in possible subhtreshold oscillations and low-threshold spiking
        self.c = np.zeros(size)                 # vector for c: after-spike reset value of the membrane potential v
        self.d = np.zeros(size)                 # vector for d: after-spike reset of the recovery variable u
        self.Weights = np.zeros((size,size))    # weight matrix
        self.Inputs = np.zeros(size)            # neuron output vector
        self.Firing = np.zeros(size)            # neuron firing vector

    def pcnnParams(self, numberExcitatory):
        for i in range(self.Size):
            # Excitatory neurons
            if i < numberExcitatory:
                self.Weights[i] = 0.5*np.random.uniform(0,1,size=self.Size)
                r = np.random.random()
                # Fixed parameters for excitatory neurons; based on Regular Spiking (RS) neurons
                self.a[i] = 0.02
                self.b[i] = 0.2
                self.c[i] = -65+15*r**2
                self.d[i] = 8-6*r**2
            else:
                # Inhibitory neurons
                # Fixed parameters for inhibitory neurons; based on Fast Spiking (FS) neurons
                self.Weights[i] = -0.5*np.random.uniform(0,1,size=self.Size)
                r = np.random.random()
                self.a[i] = 0.02+0.08*r
                self.b[i] = 0.25-0.05*r
                self.c[i] = -65
                self.d[i] = 2

    def step(self, dt, inputs):
        self.Inputs += inputs
        # Update state
        # Computing change in membrane potential based on Izhikevich equation
        dVdt = (0.04 * (self.Voltages**2)) + (5 * self.Voltages) + 140.0 - self.u + self.Inputs
        # Change in recovery variable
        dudt = self.a * ((self.b * self.Voltages) - self.u)
        # Updates the membrane potential
        self.Voltages += dt * dVdt
        # Updates the recovery variable
        self.u += dt * dudt
        # Detect firings and update firing array 
        self.Firing = np.where(self.Voltages < 30.0, 0, 1)
        # Reset membrane potential of neurons that fired
        self.Voltages = self.Voltages * (1-self.Firing) + self.c * self.Firing
        # Increase the recovery variable of neurons that fired
        self.u = self.u + self.d * self.Firing
        # Prepare synaptic input for next step
        self.Inputs = np.dot(self.Weights, self.Firing) ### Check if it should be .T or not
        #print(self.Weights, firing, self.Inputs)