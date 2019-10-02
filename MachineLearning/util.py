# -*- coding: utf-8 -*-

#### ParentClass 이름 불러오는 메쏘드
import numpy as np

class util:
    def print_base(class_name):
        for base in class_name.__class__.__bases__:
            return base.__name__

#드롭아웃
class DropOut:
    def __init__(self, dropout_ratio=0.9):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask