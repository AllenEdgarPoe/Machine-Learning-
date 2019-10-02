#Activation
import numpy as np

class ActivationLayer:
    pass
class Activation_Sigmoid(ActivationLayer):
    def __init__(self):
        self.out = None
        
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out
    
    def backward(self, gradient):
        dx = gradient * (1.0 - self.out) * self.out
        return dx
    
    
class Activation_Relu(ActivationLayer):
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx# -*- coding: utf-8 -*-

