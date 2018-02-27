
# coding: utf-8

# # ch 5.5 BP算法
# 试编程实现标准BP算法和累积BP算法，在西瓜数据集3.0上分别用这两个算法训练一个单隐层网络，并进行比较。

# In[9]:

import numpy as np
import pandas as pd


# In[10]:

def sigmoid(x):
    return 1/(1+np.exp(-x))


# In[11]:

dataset = pd.read_csv('data/table_4_3_watermelon_3_0_num.csv')
dataset = dataset.drop('Idx',axis=1)


# In[20]:

before_error = 0
error = 1
eta = 0.0001
h_num = 10
j_num = 2
i_num = dataset.shape[1] - 1
l_num = 2
w_h_j = np.random.random([h_num, j_num])
v_i_h = np.random.random([i_num, h_num])
theta_j = np.random.random([1,j_num])
gamma_h = np.random.random([1,h_num])
while np.abs(error - before_error) > 0.001:
    before_error = error
    error = 0
    k = np.random.randint(len(dataset))
    # for k in range(len(dataset)):
    x_i, y_i = dataset.ix[k,:-1], dataset.ix[k,-1]  # x_i：1xi的向量
    x_i = np.array(x_i)
    x_i = np.reshape(x_i, [1,i_num])
    y_j = np.zeros([1,l_num])
    y_j[0][int(y_i)] = 1
    alpha_h = np.dot(x_i, v_i_h)
    b_h = sigmoid(alpha_h - gamma_h)
    beta_j = np.dot(b_h, w_h_j)
    # formula 5.3
    y_j_cap = sigmoid(beta_j - theta_j)
    # formula 5.10
    g_j = y_j_cap * (np.ones_like(y_j_cap)-y_j_cap)*(y_j - y_j_cap)
    # formula 5.15
    e_h = b_h *(np.ones_like(b_h)-b_h) * np.dot(g_j, w_h_j.T)

    delta_w_h_j = eta * np.dot(b_h.T, g_j)
    delta_theta_j = -eta * g_j
    delta_v_i_h = eta * (np.dot(e_h.T, x_i)).T
    delta_gamma_h = -eta * e_h

    w_h_j += delta_w_h_j
    theta_j += delta_theta_j
    v_i_h += delta_v_i_h
    gamma_h += delta_gamma_h

    error = 0.5*np.sum((y_j_cap - y_j)*(y_j_cap - y_j))
    print('error: ',error)    


# In[18]:

def predict(x_i, w_h_j, v_i_h, theta_j, gamma_h):
    alpha_h = np.dot(x_i, v_i_h)
    b_h = sigmoid(alpha_h - gamma_h)
    beta_j = np.dot(b_h, w_h_j)        
    y_j_cap = sigmoid(beta_j - theta_j)
    if y_j_cap[0][0] > y_j_cap[0][1]:
        print(0)
        return 0
    else:
        print(1)
        return 1
#     print(y_j_cap)
    
#     return y_j_cap


# In[21]:

for i in range(17):
    predict(np.array(dataset.ix[14,:-1]), w_h_j, v_i_h, theta_j, gamma_h)


# In[ ]:



