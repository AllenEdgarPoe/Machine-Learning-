#CostLayer
import numpy as np

class CostLayer:
    loss = None   #손실함수
    y_hat = None  #출력 값
    y = None      #정답 레이블(one-hot encoding형태)
    
class MeanSquaredError(CostLayer):
    def __init__(self):
        pass
    def forward(self, y_hat, y):
        self.y_hat = y_hat
        self.y = y
        self.loss = 0.5 * np.sum((y_hat-y)**2)
        return self.loss
    
    def backward(self, gradient=1):
        batch_size = self.y.shape[0]
        if self.y.size == self.y_hat.size:
            dx = (self.y_hat - self.y)/batch_size
        else:
            dx = self.y_hat.copy()
            dx[np.arage(batch_size), self.y]-=1
            dx = dx/batch_size
        
        return dx# -*- coding: utf-8 -*-

