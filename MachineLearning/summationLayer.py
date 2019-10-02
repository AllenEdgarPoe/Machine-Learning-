#SummationLayer
import numpy as np

class SummationLayer:
    weight_dict = {}
    pass
    
class Perceptron(SummationLayer):
    def __init__(self, hidden_layer_num=10, final_layer=False):
        self.hidden_layer_num = hidden_layer_num
        self.final_layer = final_layer
        self.original_x_shape = None
        self.x = None
        
        self.weight = None
        self.bias = None
        
        self.weight_gradient = None
        self.bias_gradient = None
        
    def update(self, weight, bias):
        self.weight = weight
        self.bias = bias
        
        
    def pass_hidden_layer_num(self):
        if self.final_layer == True:
            hidden_layer_num = "final_layer"
        else:
            hidden_layer_num = self.hidden_layer_num
        
        return hidden_layer_num
    
    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        
        out = np.dot(self.x, self.weight) + self.bias
        
        return out
    
    def backward(self, gradient):
        dot_matrix= np.dot(gradient, self.weight.T)
        self.weight_gradient = np.dot(self.x.T, gradient)
        self.bias_gradient = np.sum(gradient, axis=0)
        
        dot_matrix = dot_matrix.reshape(*self.original_x_shape)
        
        return dot_matrix
    
        # -*- coding: utf-8 -*-

