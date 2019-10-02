# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
from loopController import *
from inputLayer import *
import numpy as np

file_path  =r"C:\workspace\django_workspace\prototype4_\framework\PearMountEngine\Dataset\iris.csv"
query = "TirionFordring$DMLP$InputLayer_CSV(y_label_index_starting=4, y_label_index_last=4)§Perceptron(hidden_layer_num=10)§Activation_Relu()§Perceptron(final_layer=True)§Activation_Sigmoid()§MeanSquaredError()§PerceptronParameter(learning_rate=0.001, batch_size=5, epochs=1000, optimizer='SGD', accuracy_limit=90)"

test = LoopController(file_path, query)
test.learn()

plt.style.use("seaborn")
plt.plot(test.train_acc_list, label="train_acc",marker='o', markevery=10)
plt.plot(test.test_acc_list, label="test_acc",marker='o', markevery=10)

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.title("SGD")
plt.show()


plt.plot(test.train_loss_list, label="train_loss")
#
#
optimizer = ["SGD","Momentum","NAG","AdaGrad","RMSprop","Adam"]
new = dict()
for opt in optimizer:
    file_path = file_path
    query = "TirionFordring$DMLP$InputLayer_CSV(y_label_index_starting=4, y_label_index_last=4)§Perceptron(hidden_layer_num=10)§Activation_Relu()§Perceptron(final_layer=True)§Activation_Sigmoid()§MeanSquaredError()§PerceptronParameter(learning_rate=0.001, batch_size=5, epochs=100, optimizer='"+opt+"')"
    query = str(query)
    a = LoopController(file_path,query)
    a.learn()
    new[a.param['optimizer']]=a.test_acc_list
    
    
#optimizer에 따라 달라지는 graph
plt.style.use("seaborn")
plt.figure(figsize=(12,10))

# x_ = list(range(max([len(i) for i in new.values()])))

for key, val in new.items():
    key = str(key)
    plt.plot(val, label=key, marker='o', markevery=5)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.title("Optimizer")
plt.show()
#
#
#
#적합한 hidden_layer 값 찾기!!!!!!!

hidden_layer = np.linspace(1,100,4)

new = dict()
for hidden in hidden_layer:
    file_path = file_path
    query = "TirionFordring$DMLP$InputLayer_CSV(y_label_index_starting=4, y_label_index_last=4)§Perceptron(hidden_layer_num="+str(int(hidden))+")§Activation_Relu()§Perceptron(final_layer=True)§Activation_Sigmoid()§MeanSquaredError()§PerceptronParameter(learning_rate=0.001, batch_size=10, epochs=100, optimizer='Adam')"
    query = str(query)
    a = LoopController(file_path,query)
    a.learn()
    new[str(hidden)]=a.train_acc_list
    
    
#optimizer에 따라 달라지는 graph
plt.style.use("seaborn")
plt.figure(figsize=(12,10))

# x_ = list(range(max([len(i) for i in new.values()])))

for key, val in new.items():
    key = str(key)
    plt.plot(val, label=key, marker='o', markevery=5)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.title("Optimizer")
plt.show()