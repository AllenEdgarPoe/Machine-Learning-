# -*- coding: utf-8 -*-
from inputLayer import *
from parameter import *
import numpy as np
from util import *
from summationLayer import *
from activationLayer import *
from costLayer import *
from optimizer import *
from network import *

class LoopController:
#     full_layer = []
#     train_loss_list = []
#     train_acc_list = []
#     test_acc_list = []
#     param = None
    
    def __init__(self, file_path, query):
        self.file_path = file_path
        self.query = query
        
        self.session = self.query.split("$")[0]
        self.network = self.query.split("$")[1] #DMLP, CNN, RNN.. 
        string_to_pass = self.query.split("$")[2].split("§")
        
        self.full_layer = []
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []
        self.param = None
        
        for string in string_to_pass:
            string_class =eval(string)
            if util.print_base(string_class) == "InputLayer":
                self.x_train, self.y_train, self.x_test, self.y_test = string_class.build_layer(self.file_path)
                
            elif util.print_base(string_class) == "Parameter":
                self.param = string_class.get_parameter()
                
            else:
                self.full_layer.append(string)        
        
    def train_step(self):
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        y_batch = self.y_train[batch_mask]
        
        grads = self.learn_net.gradient(x_batch, y_batch)
        self.optimizer.update(self.learn_net.weight_layer, grads)
        
        loss = self.learn_net.loss(x_batch, y_batch)
        self.train_loss_list.append(loss)
        
        print("train loss: " + str(loss))
        
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch +=1
            
            x_train_sample, y_train_sample = self.x_train, self.y_train
            x_test_sample, y_test_sample = self.x_test, self.y_test
            
            train_acc = self.learn_net.accuracy(x_train_sample, y_train_sample)
            test_acc = self.learn_net.accuracy(x_test_sample, y_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)
            
            print("=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc) + " ===")
            
            
        self.current_iter += 1
            
            
        
        
        
    def learn(self):
        self.epochs = self.param["epochs"]
        self.batch_size = self.param["batch_size"]
        self.learning_rate = self.param["learning_rate"]
        self.optimizer = eval(self.param["optimizer"])(self.learning_rate)
        self.dropout = self.param["dropout"]
        
        self.learn_net = eval(self.network)(self.full_layer, self.dropout)
        self.learn_net.build_model(self.x_train, self.y_train)
#         print("build_model 완료")
        
        
        
        self.train_size = self.x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / self.batch_size, 1)
        self.max_iter = int(self.epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0
        
        for i in range(self.max_iter):
            self.train_step()
            if self.param["accuracy_limit"] != None:
                if self.test_acc_list[-1] >= self.param["accuracy_limit"]*0.01:
                    print("Reached accuracy_limit")
                    break
            
        test_acc = self.learn_net.accuracy(self.x_test, self.y_test)
        
        print("=====Final Test Accuracy====")
        print("test acc: "+str(test_acc))
        
        
        
            
        
        
        
        
                