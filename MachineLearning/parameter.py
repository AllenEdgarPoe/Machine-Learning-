#Parameter Layer
class Parameter:
    param = dict()
class PerceptronParameter(Parameter):
    def __init__(self,learning_rate=0.1, batch_size=10, epochs=10, optimizer="sgd", accuracy_limit=None, dropout="False"):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.accuracy_limit = accuracy_limit
        self.dropout = dropout
        
    def get_parameter(self):
        self.param["learning_rate"] = self.learning_rate
        self.param["batch_size"] = self.batch_size
        self.param["epochs"] = self.epochs
        self.param["optimizer"] = self.optimizer
        self.param["accuracy_limit"] = self.accuracy_limit
        self.param["dropout"] = self.dropout
        return self.param# -*- coding: utf-8 -*-

