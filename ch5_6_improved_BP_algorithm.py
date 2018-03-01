
# coding: utf-8

# # ch 5.6 改进的BP算法
# 试设计一个BP改进算法，能通过动态调整学习率显著提升收敛速度。编程实现该算法，并选择两个UCI数据集与标准BP算法进行比较。
# 
# Note：
# 1. 学习率调整方法包括Adagrad，RMSProp等

# In[1]:

import numpy as np
import pandas as pd


# In[35]:

def sigmoid(x):
    return 1/(1+np.exp(-x))


# In[36]:



# In[37]:

def cal_acc(dataset_array, w_h_j, v_i_h, theta_j, gamma_h):
    dataset = dataset_array[:,:-1]
    label = dataset_array[:,-1]
    acc = 0
    for i in range(len(dataset)):
        pred_label = predict(dataset[i], w_h_j, v_i_h, theta_j, gamma_h)
        if pred_label == label[i]:
            acc += 1
    return acc/len(dataset)


# In[38]:

def predict(x_i, w_h_j, v_i_h, theta_j, gamma_h):
    alpha_h = np.dot(x_i, v_i_h)
    b_h = sigmoid(alpha_h - gamma_h)
    beta_j = np.dot(b_h, w_h_j)        
    y_j_cap = sigmoid(beta_j - theta_j)
#     print(y_j_cap, end=" ")
    if y_j_cap[0][0] > y_j_cap[0][1]:
#         print(0)
        return 0
    else:
#         print(1)
        return 1


# In[42]:

def adagrad_bp(dataset_array, niter):       
    h_num = 30
    j_num = 2
    i_num = dataset_array.shape[1] - 1
    l_num = 2
    epsilon = 0.1
    sigma = 10e-7
    m_num = 20
    w_h_j = np.random.random([h_num, j_num])
    v_i_h = np.random.random([i_num, h_num])
    theta_j = np.random.random([1,j_num])
    gamma_h = np.random.random([1,h_num])
    r_w_h_j = np.zeros([h_num, j_num])
    r_v_i_h = np.zeros([i_num, h_num])
    r_theta_j = np.zeros([1,j_num])
    r_gamma_h = np.zeros([1,h_num])
    error_list = []
    for i in range(niter):
        error = 0
        delta_w_h_j = np.zeros([h_num, j_num])
        delta_theta_j = np.zeros([1, j_num])
        delta_v_i_h = np.zeros([i_num, h_num])
        delta_gamma_h = np.zeros([1, h_num])
        for _ in range(m_num):
            k = np.random.randint(len(dataset_array))
            x_i, y_i = dataset_array[k,:-1], dataset_array[k,-1]  # x_i：1xi的向量
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
            
            delta_w_h_j += np.dot(b_h.T, g_j)
            delta_theta_j += -g_j
            delta_v_i_h += (np.dot(e_h.T, x_i)).T
            delta_gamma_h += -e_h
            error += 0.5 * np.sum((y_j_cap - y_j) * (y_j_cap - y_j))

        delta_w_h_j /= m_num
        delta_theta_j /= m_num
        delta_v_i_h /= m_num
        delta_gamma_h /= m_num

        r_w_h_j += delta_w_h_j*delta_w_h_j
        r_v_i_h += delta_v_i_h*delta_v_i_h
        r_theta_j += delta_theta_j*delta_theta_j
        r_gamma_h += delta_gamma_h*delta_gamma_h

        w_alpha = np.ones_like(delta_w_h_j) * epsilon / (np.ones_like(delta_w_h_j) * sigma + np.sqrt(r_w_h_j))
        theta_alpha = np.ones_like(delta_theta_j)*epsilon/(np.ones_like(delta_theta_j)*sigma+np.sqrt(r_theta_j))
        v_alpha = np.ones_like(delta_v_i_h)*epsilon/(np.ones_like(delta_v_i_h)*sigma+np.sqrt(r_v_i_h))
        gamma_alpha = np.ones_like(delta_gamma_h)*epsilon/(np.ones_like(delta_gamma_h)*sigma+np.sqrt(r_gamma_h))

        w_h_j += w_alpha*delta_w_h_j
        theta_j += theta_alpha*delta_theta_j
        v_i_h += v_alpha*delta_v_i_h
        gamma_h += gamma_alpha*delta_gamma_h


        if i % 10 == 0:
            error_list.append(error/m_num)
    print(error_list)
    return w_h_j, v_i_h, theta_j, gamma_h


# In[43]:

# get_ipython().run_cell_magic('time', '', 'test_num = dataset.shape[0]//5\nw_h_j, v_i_h, theta_j, gamma_h = adagrad_bp(dataset_array[test_num:], niter =350)\nacc = cal_acc(dataset_array[:test_num], w_h_j, v_i_h, theta_j, gamma_h)\nprint(acc)')
dataset = pd.read_csv('data/blood.txt',header=None)
dataset.columns = ['x1','x2','x3','x4','label']
dataset = dataset.apply(lambda x:(x-np.min(x))/(np.max(x)-np.min(x)))
dataset_array = np.array(dataset)
dataset_array = dataset_array[np.random.permutation(len(dataset))]

test_num = dataset.shape[0]//5
w_h_j, v_i_h, theta_j, gamma_h = adagrad_bp(dataset_array[test_num:], niter =350)
acc = cal_acc(dataset_array[:test_num], w_h_j, v_i_h, theta_j, gamma_h)
print(acc)
# In[30]:

# %%time
# test_num = dataset.shape[0]//5
# w_h_j, v_i_h, theta_j, gamma_h = standard_bp(dataset_array[test_num:], niter =350, eta=0.1)
# acc = cal_acc(dataset_array[:test_num], w_h_j, v_i_h, theta_j, gamma_h)
# print(acc)


# In[ ]:



