#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
STA414 Assignment 1 
Due Jan 29, 2019

@name: cross_validation.py
@author: zikunchen
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

### Part (b) ###

def shuffle_data(data):
    """Returns randomly uniformly permuted version of data along the samples.
    Note that y and X need to be permuted the same way preserving the target-feature pairs."""
    indices = np.arange(len(data['y']))
    np.random.seed(SEED)

    indices = np.random.permutation(indices)
    
    return {'y': np.array(data['y'])[indices].tolist(), \
            'X': np.array(data['X'])[indices].tolist()}

def split_data(data, num_folds, fold):
    """
    - Attributes -
    data:
        (y, X) pair in the form of dictionary
    num_folds:
        number of folds in total
    fold:
        specific fold number to be used as the training set
        
    - Returns - 
    data_fold:
        (y, X) pair used for validation
    data_rest:
        rest of the data used for training
    """

    N = len(data['y'])    
    fold_size = int(N//num_folds)
    
    fold_indices = np.arange((fold - 1) * fold_size, fold * fold_size)
    data_fold = {'y': np.array(data['y'])[fold_indices].tolist(), \
                     'X': np.array(data['X'])[fold_indices].tolist()}
    
    rest_indices = np.append(np.arange(0, (fold - 1) * fold_size), \
                             np.arange(fold * fold_size, N), axis = 0)    
    data_rest = {'y': np.array(data['y'])[rest_indices].tolist(), \
                     'X': np.array(data['X'])[rest_indices].tolist()}
    
    return data_fold, data_rest    


def train_model(data, lambd):
    """ Returns the coefficients of ridge regression with penalty level λ."""
    
    N = len(data['y'])
    
    X = np.array(data['X'])
    y = np.array(data['y']).reshape(N,1)
    
    M = X.shape[1]
    
    XTX = np.dot(X.T, X)
    
    model = np.dot(np.dot(np.linalg.inv(XTX + lambd*np.identity(M)), X.T), y)
    
    return np.squeeze(model).tolist()
    
    
def predict(data, model):
    """ Returns the predictions based on data and model."""
    
    X = np.array(data['X'])
    M = X.shape[1]
    beta = np.array(model).reshape(M,1)
    
    return np.squeeze(np.dot(X, beta)).tolist()

def loss(data, model):
    """Returns the average squared error loss based on model."""
    
    N = len(data['y'])
    y = np.array(data['y']).reshape(N,1)
    
    predictions = np.array(predict(data, model)).reshape(N, 1)
    
    error = np.sum(np.squeeze((y - predictions)**2)) / N
    return error
    
def cross_validation(data, num_folds, lambd_seq): 
    """Returns the cross validation error across all lambdas in lambd_seq"""
    data = shuffle_data(data)
    cv_error = [0] * len(lambd_seq)
    
    for i in range(len(lambd_seq)):
        lambd = lambd_seq[i]
        cv_loss_lmd = 0
        
        for fold in range(1, num_folds + 1):
            val_cv, train_cv = split_data(data, num_folds, fold)
            model = train_model(train_cv, lambd)
            cv_loss_lmd += loss(val_cv, model)
        cv_error[i] = cv_loss_lmd/num_folds
    
    return cv_error

### Part (c) ###

def lambd_train_test(train_data, test_data, lambd_seq): 
    """Returns the training and test error across all lambdas in lambd_seq"""

    train_error = [0] * len(lambd_seq)
    test_error = [0] * len(lambd_seq)

    for i in range(len(lambd_seq)):
        lambd = lambd_seq[i]
        model = train_model(train_data, lambd)
        train_error[i] = loss(train_data, model)
        test_error[i] = loss(test_data, model)
    
    return train_error, test_error
  
### Helper Function ###

def find_min_lambd(error_list):
    """Find the best lambda based on the minimum error achieved in error_list"""
    min_error, index =  min((val, idx) for (idx, val) in enumerate(five_fold_cv))
    return lambd_seq[index]


if __name__ == '__main__':
    
    ### Part (a) ###
    dataset = sio.loadmat("/Users/zikunchen/Desktop/STA414/A1/dataset.mat")
    data_train_X = dataset["data_train_X"]
    data_train_y = dataset["data_train_y"][0]
    data_test_X = dataset["data_test_X"]
    data_test_y = dataset["data_test_y"][0]

    # Set Random Seed to reproduce results
    SEED = 2019
    
    # Construct datasets
    train_data = {'y': data_train_y, 'X': data_train_X}
    test_data = {'y': data_test_y, 'X': data_test_X}
    
    # Construct Lambda values
    space = (1.5-0.02)/(50-1+2)
    lambd_seq = np.arange(0.02, 1.5, space).tolist() 
    
    # Compute training, validation and test errors
    train_error, test_error = lambd_train_test(train_data, test_data, lambd_seq)
    five_fold_cv = cross_validation(train_data, 5, lambd_seq)
    ten_fold_cv = cross_validation(train_data, 10, lambd_seq)
    
    #Find the lambda with the least error
    lambd_test = find_min_lambd(test_error)
    lambd_five_fold= find_min_lambd(five_fold_cv)
    lambd_ten_fold = find_min_lambd(ten_fold_cv)

    # Plot 'Training Error', 'Test Error', 
    # '5-fold Validation Error', and '10-fold Validation Error' 
    # curves against Lambda values
    fig, ax = plt.subplots()
    
    test_min = min(test_error)
    xpos = test_error.index(test_min)
    xmin = lambd_seq[xpos]

    ax.annotate('Best λ = %f' % lambd_test, xy=(xmin, test_min), \
                xytext=(xmin, test_min+1), \
                arrowprops=dict(facecolor='red', shrink=0.05),)

    plt.plot(lambd_seq, train_error, 'mo-') 
    plt.plot(lambd_seq, test_error, 'bo-') 
    plt.plot(lambd_seq, five_fold_cv, 'go-') 
    plt.plot(lambd_seq, ten_fold_cv, 'yo-') 
    
    plt.title('Errors vs. λ')
    plt.ylabel('Error')
    plt.xlabel('λ')
    plt.legend(['Training Error', 'Test Error', '5-fold Validation Error', \
                '10-fold Validation Error'], loc='upper right')

    plt.show()
    
    print("Test error is minimized when λ = %f" % lambd_test)
    print("5-fold validation error is minimized when λ = %f" % lambd_five_fold)
    print("10-fold validation error is minimized when λ = %f" % lambd_ten_fold)
    