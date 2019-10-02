#Optimizer
import numpy as np

class SGD:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        
    def update(self, weight_layer, grads):
        for key in weight_layer.keys():
            weight_layer[key] -= self.learning_rate * grads[key] 
            
            
class Momentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v = None
        
    def update(self, weight_layer, grads):
        if self.v is None:
            self.v = {}
            for key, val in weight_layer.items():                                
                self.v[key] = np.zeros_like(val)
                
        for key in weight_layer.keys():
            self.v[key] = self.momentum*self.v[key] - self.learning_rate*grads[key] 
            weight_layer[key] += self.v[key]


class NAG:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v = None
        
    def update(self, weight_layer, grads):
        if self.v is None:
            self.v = {}
            for key, val in weight_layer.items():
                self.v[key] = np.zeros_like(val)
            
        for key in weight_layer.keys():
            self.v[key] *= self.momentum
            self.v[key] -= self.learning_rate * grads[key]
            weight_layer[key] += self.momentum * self.momentum * self.v[key]
            weight_layer[key] -= (1 + self.momentum) * self.learning_rate * grads[key]


class AdaGrad:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.h = None
        
    def update(self, weight_layer, grads):
        if self.h is None:
            self.h = {}
            for key, val in weight_layer.items():
                self.h[key] = np.zeros_like(val)
            
        for key in weight_layer.keys():
            self.h[key] += grads[key] * grads[key]
            weight_layer[key] -= self.learning_rate * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class RMSprop:
    def __init__(self, learning_rate=0.01, decay_rate = 0.99):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.h = None
        
    def update(self, weight_layer, grads):
        if self.h is None:
            self.h = {}
            for key, val in weight_layer.items():
                self.h[key] = np.zeros_like(val)
            
        for key in weight_layer.keys():
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            weight_layer[key] -= self.learning_rate * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class Adam:

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, weight_layer, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in weight_layer.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        learning_rate_t  = self.learning_rate * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in weight_layer.keys():
            #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            weight_layer[key] -= learning_rate_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
            
            #unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            #unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            #weight_layer[key] += self.learning_rate * unbias_m / (np.sqrt(unbisa_b) + 1e-7)# -*- coding: utf-8 -*-

