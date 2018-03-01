
# coding: utf-8

# # ch 5.7 单层RBF神经网络
# 根据式(5.18)和(5.19)，试构造一个能解决异或问题的单层RBF神经网络。

# In[1]:

import numpy as np


# In[32]:

def RBF(x, center_i, beta_i):
    dist = np.sum(pow((x - center_i),2))
    rho = np.exp(-beta_i*dist)
    return rho

# 输出为1
class RBF_network:
    def __init__(self):
        self.hidden_num = 0
        self.y = 0
        
    def createNN(self, input_num, hidden_num, learning_rate, center):
        self.input_num = input_num
        self.hidden_num = hidden_num
        self.center = center
        self.w = np.random.random(self.hidden_num)
        self.rho = np.zeros(self.hidden_num)
        self.beta = np.random.random(self.hidden_num)
        self.lr = learning_rate
        
    def Predict(self, x):
        self.y = 0
        for i in range(self.hidden_num):
            self.rho[i] = RBF(x, self.center[i], self.beta[i])
            self.y += self.w[i] * self.rho[i]
        return self.y
    
    def BackPropagate(self, x, y):
        self.Predict(x)
        grad = np.zeros(self.hidden_num)
        for i in range(self.hidden_num):
            # dE_k/dy_cap = (y_cap-y)  
            # dE_k/dw = (y_cap-y)*rho[i] 
            # dE_k/d_beta = -(y_cap-y)*rho[i]w_i*||x-c_i||
            grad[i] = (self.y - y) * self.rho[i]
            self.w[i] -= self.lr * grad[i]
            self.beta[i] += self.lr * grad[i] * self.w[i] * np.sum(pow((x - center[i]),2))
                          
    def trainNN(self, x, y):
        error_list = []
        for i in range(len(x)):
            self.BackPropagate(x[i], y[i])
            error = (self.y - y[i])**2
            error_list.append(error/2)
        print(error_list)
    
    


# In[33]:

train_x = np.random.randint(0,2,(100,2))
train_y = np.logical_xor(train_x[:,0],train_x[:,1])
test_x = np.random.randint(0,2,(100,2))
test_y = np.logical_xor(test_x[:,0],test_x[:,1])


# In[34]:

center = np.array([[0,0],[0,1],[1,0],[1,1]])
rbf = RBF_network()
rbf.createNN(input_num = 2, hidden_num=4 , learning_rate=0.1, center=center)


# In[35]:

rbf.trainNN(train_x, train_y)


# In[ ]:



