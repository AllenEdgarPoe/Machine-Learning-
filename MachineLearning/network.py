#!/usr/bin/env python
# coding: utf-8

# In[ ]:
#### ParentClass 이름 불러오는 메쏘드
import collections
import numpy as np
from util import *
from activationLayer import *
from costLayer import *
from summationLayer import *

class DMLP:
    def __init__(self, full_layer, dropout):
        self.full_layer = full_layer
        self.hidden_size_list = []
        
        self.weight_layer = {}                  #weight의 모음
        self.layers = collections.OrderedDict() #쌓여있는 층
        self.final_layer = None             #costlayer가 담겨있는 층
        self.dropout = dropout
        
    def build_model(self, x, y, random_seed = 3, weight_init_std='sigmoid', weight_decay_lambda=0):
        #난수 생성할때 랜덤값 고정하기, 평소에는 이거 지우면 된다. 
        np.random.seed(random_seed)
        self.weight_decay_lambda = weight_decay_lambda
        
        for sum_class in self.full_layer:
            if util.print_base(eval(sum_class)) == "SummationLayer":
                hidden = eval(sum_class).pass_hidden_layer_num()
                if hidden == "final_layer":
                    hidden = y.shape[-1]
                self.hidden_size_list.append(hidden)
        
        self.input_size = x.shape[-1]
        self.hidden_layer_num = len(self.hidden_size_list)
        
        #가중치 초기화
        self.__init_weight(weight_init_std)
        
        #계층 생성
        sum_idx = 1
        act_idx = 1
        for sum_class in self.full_layer:
            if util.print_base(eval(sum_class)) == "SummationLayer":
                self.layers['Perceptron' + str(sum_idx)] = eval(sum_class)
                self.layers['Perceptron' + str(sum_idx)].update(self.weight_layer['W' + str(sum_idx)],self.weight_layer['b' + str(sum_idx)])
                sum_idx+=1
            elif util.print_base(eval(sum_class)) == "ActivationLayer":
                self.layers['Activation_function' + str(act_idx)] = eval(sum_class)
                act_idx+=1
                if eval(self.dropout) == True:
                    self.layers['DropOut' + str(act_idx)] = DropOut()
                
            elif util.print_base(eval(sum_class)) == "CostLayer":
                self.final_layer = eval(sum_class)
        
        
    
    ##가중치 초기화
    def __init_weight(self, weight_init_std):
        '''
        가중치 초기화할때 표준편차를 지정하는 것도 아주 중요한 파라미터.
        'relu'일때는 'He 초깃값'
        'sigmoid'일때는 'Xavier 초깃값'
        '''
        all_size_list = [self.input_size] + self.hidden_size_list  
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ("sigmoid"):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])  #sigmoid사용할때 초깃값
            elif str(weight_init_std).lower() in ("relu"):
                scale = np.sqrt(2.0 / all_size_list[idx -1])   #relu사용할 때 권장값
                
            self.weight_layer['W'+str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.weight_layer['b' + str(idx)] = np.zeros(all_size_list[idx])
            
#             print("weight값: "+str(self.weight_layer))
     
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
            
        return x
    
    
    def loss(self,x,y):
        y_hat = self.predict(x)
        
        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 1):
            W = self.weight_layer['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)
            
        return self.final_layer.forward(y_hat, y) + weight_decay
    
    
    def accuracy(self,x, y):
        y_hat = self.predict(x)
        y_hat = (y_hat == y_hat.max(axis=1)[:,None]).astype(int)
        accuracy = np.all(y_hat == y, axis=1)
        
        return np.sum(accuracy)/len(accuracy)
    
    
    
    def gradient(self, x, y):
        #forward
        self.loss(x,y)
        
        #backward
        gradient = 1
        gradient = self.final_layer.backward(gradient)
        
        layers = list(self.layers.values())
        layers.reverse()
        
        for layer in layers:
            gradient = layer.backward(gradient)
            
        grads = {}
        for idx in range(1, self.hidden_layer_num+1):
            grads['W' + str(idx)] = self.layers['Perceptron' + str(idx)].weight_gradient + self.weight_decay_lambda * self.layers["Perceptron" + str(idx)].weight
            
            grads['b' + str(idx)] = self.layers['Perceptron'+str(idx)].bias_gradient
            
        return grads
        

