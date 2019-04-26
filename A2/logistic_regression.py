#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 11:08:29 2019

@author: zikunchen
"""

import loadMNIST as load
import matplotlib.pyplot as plt
import numpy as np
#from autograd import grad
#from autograd import scipy
from scipy import special 

def binarize(train_x):
    train_x[train_x >= 0.5] = 1
    train_x[train_x < 0.5] = 0
    return train_x  

def sum_exp(x): 
   e_x = np.exp(x - np.max(x))
   return e_x / e_x.sum()

def grad_LL(W, train_x, train_c):
    
    N = train_x.shape[0]
    D = train_x.shape[1]
    K = train_c.shape[1]
    
    gradient = np.zeros((D,K))
    labels = np.argmax(train_c, axis=1)
    WX = np.matmul(train_x, W)
    
    for n in range(N):
        wx_n = WX[n]
        exp_wx_n = np.exp(wx_n - np.max(wx_n))
        perdict_n = exp_wx_n / np.sum(exp_wx_n)
        x = train_x[n]
        for c in range(K):    
            if labels[n] == c:
                gradient[:,c] -= x * (perdict_n[c] - 1)
            else:
                gradient[:,c] -= x * perdict_n[c]
    return gradient

def log_likelihood(W, train_x, train_c): 
   N = train_x.shape[0]
   labels = np.argmax(train_c, axis=1)
   wcTx = 0
   LSE = 0
   WX = np.matmul(train_x, W) 
   for n in range(N):
       wcTx += np.matmul(train_x[n], W[:, labels[n]])
       # log of sum (over c) of exp(wcTx)
       LSE += special.logsumexp(WX[n]) 
   return (wcTx - LSE)

def accuracy(W, train_x, train_c):
    N = train_x.shape[0]
    labels = np.argmax(train_c, axis=1)
    WX = np.matmul(train_x, W)
    predictions = np.zeros(N)
    for n in range(N):
        wx_n = WX[n]
        exp_wx_n = np.exp(wx_n - np.max(wx_n))
        perdict_n = exp_wx_n / np.sum(exp_wx_n)
        predictions[n] = np.argmax(perdict_n)    
    correct_count = len(np.where(predictions == labels)[0])
    return correct_count/N


if __name__ == "__main__":
    
# =============================================================================
#     Load and Store Dataset
# =============================================================================

    training_cutoff = 10000
    debug_cutoff = 100
    image_size = 784
    
    N_data, train_images, train_labels, test_images, test_labels = load.load_mnist()
    
    data = {'train_x': binarize(train_images[:training_cutoff]), 
            'train_c': train_labels[:training_cutoff], 
            'test_x': test_images, 'test_c': test_labels}
    
    debug_data = {'train_x': binarize(train_images[:debug_cutoff]), 
                  'train_c': train_labels[:debug_cutoff], 
                  'test_x': test_images, 'test_c': test_labels}

    train_x = data['train_x']
    train_c = data['train_c']
    
    test_x = data['test_x']
    test_c = data['test_c']
    
# =============================================================================
#     2 b) Gradient Based Optimizer
# =============================================================================
    
    N = train_x.shape[0]
    K = train_c.shape[1]
    D = train_x.shape[1]
    
    initial_W = np.zeros((D, K))
    rate = 0.01
    steps = 10000
    
    W = initial_W
    n_step = 0
    old_LL = 0    
    LL = log_likelihood(W, train_x, train_c)
    LL_history = [LL]
    
    while (abs(LL - old_LL) != 0):
        old_LL = LL
        gradient = grad_LL(W, train_x, train_c)
        rate = 0.05/(n_step + 1)
        W += rate * gradient
        LL = log_likelihood(W, train_x, train_c)
        LL_history.append(LL) 
        n_step += 1
        print('epoch: %d | log-likelihood: %f'  % (n_step, LL))
    
    plt.plot(np.arange(len(LL_history[:])), LL_history[:], 'm-')
    
# log-likelihood curve 
        
    plt.title('log-likelihood vs. iteration')
    plt.ylabel('log-likelihood')
    plt.xlabel('iteration')
    plt.plot(np.arange(n_step + 1), LL_history, 'm-')
    plt.show()
    
    # In 3488 steps, Logistic Regression converges to the maximized log-likelihood: -7677.208967
    print('In %d steps, Logistic Regression converges to the maximized log-likelihood: %f'  %(n_step, LL))
    
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)
    load.plot_images(W.T, ax)
    plt.show()
    
    avg_LL_train = log_likelihood(W, train_x, train_c)/N
    avg_LL_test = log_likelihood(W, test_x, test_c)/N
    
    accuracy_train = accuracy(W, train_x, train_c)
    accuracy_test = accuracy(W, test_x, test_c)
    
# Average Likelihood and Accuracy
    
    print('Logistic Regression Model:')
    print('Average Log-Likelihood on Training Set:', avg_LL_train)
    print('Average Log-Likelihood on Test Set:', avg_LL_test)
    print('Training Accuracy:', accuracy_train)
    print('Testing Accuracy:', accuracy_test)
    
    
    
    