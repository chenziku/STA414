#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 00:21:02 2019

@author: zikunchen
"""

import matplotlib.pyplot as plt
import numpy as np

# MU: mean matrix with mean vectors of each cluster
# COV: Covariance Matrices of clusters
# pi: mixing porportions vector of length K
# X: data matrix 400 x 2
# R: Responsibilities N x K

def log_ikelihood(X, pi, MU, COV):
    LL = 0
    for n in range(N):
        sum_k = 0
        for k in range(K):
            sum_k += pi[k] * normal_density(X[n], MU[k], COV[k])
        LL = np.log(sum_k)
    return LL
    
def normal_density(x, mu, cov):
    # x: D x 1
    K = len(mu)
    denom = np.sqrt((2 * np.pi)** K * np.linalg.det(cov))
    Cinv = np.linalg.inv(cov)
    numer = np.exp(-0.5 * np.matmul(np.matmul((x.T - mu.T), Cinv), (x - mu)))
    return numer/denom

def em_e_estep(X, MU, COV, pi):
    # MU: K x D
    # COV: K x D x D
    N = X.shape[0]
    K = len(pi)
    R = np.zeros((N, K))
    for n in range(N):
        piN = np.zeros(K)
        for k in range(K):
            piNk = pi[k] * normal_density(X[n], MU[k], COV[k])
            piN[k] = piNk
        R[n] = piN/np.sum(piN)    
    return R
    
def em_m_estep(X, R):
    # X: N x D
    # R: N x K (400 x 2)
    N = X.shape[0] # number of points
    K = R.shape[1] # number of cluster
    D = X.shape[1] # dimensionality of each data point
    MU = np.zeros((K, D))
    COV = np.zeros((K, D, D))
    pi = np.zeros(K)
    
    RX = np.matmul(R.T, X)
    Nk = R.sum(axis = 0)
    pi = Nk/N
    
    for k in range(K):
        MU[k] = RX[k]/Nk[k]
        cov_sum = np.zeros((D,D))
        for n in range(N):
            cov_sum += R[n,k] * np.matmul((X[n:n+1] - MU[k:k+1]).T, \
                            (X[n:n+1] - MU[k:k+1]))
        COV[k] = cov_sum/Nk[k]
    return MU, COV, pi                                                                                     

def accuracy(X, R):
    
    N_data = X.shape[0]
    cutoff = int(N_data/2)
    
    true_id1 = np.arange(cutoff)
    true_id2 = np.arange(cutoff, N_data)
    
    R_id1 = np.where(np.argmax(R, axis = 1) == 0)[0]
    R_id2 = np.where(np.argmax(R, axis = 1) == 1)[0]
    
    correct_count_1 = len(list(set(true_id1).intersection(R_id1)))
    correct_count_2 = len(list(set(true_id2).intersection(R_id2)))
    
    return (correct_count_1 + correct_count_2) / N_data

if __name__ == "__main__":
    

#    SEED = 2018
#    In 759 steps, EM converges to the maximized log-likelihood: -5.498959
#    The miss-classification error is: 0.08
#    
#    SEED = 2017 
#    In 4011 steps, EM converges to the maximized log-likelihood: -6.332573
#    The miss-classification error is: 0.49
    
# =============================================================================
#   Data Generation
# =============================================================================
    
    SEED = 2017
    
    m1 = [0.1, 0.1]
    m2 = [6.0, 0.1]
    cov = [[10, 7], [7, 10]]
    
    np.random.seed(SEED)
    
    x1, y1 = np.random.multivariate_normal(m1, cov, 200).T
    x2, y2 = np.random.multivariate_normal(m2, cov, 200).T
    plt.plot(x1, y1, 'x')
    plt.plot(m1[0], m1[1], 'x', color='red')

    plt.plot(x2, y2, 'o')
    plt.plot(m2[0], m2[1], 'o', color='red')
    plt.title('SEED %d' %(SEED))
    plt.axis('equal')
    plt.show()
    
# =============================================================================
#     4 c) Expectation Maximization
# =============================================================================
    
    K = 2
    D = 2
    N = 400
    
    X = np.vstack((np.stack((x1, y1)).T, np.stack((x2, y2)).T))
    MU = np.array([[0.0, 0.0], [1.0, 1.0]])
    I = np.eye(2)
    COV = np.vstack((I,I)).reshape(K, D, D)
    pi = np.array([0.5, 0.5])
        
    n_step = 0
    LL_old = -1
    LL = log_ikelihood(X, pi, MU, COV)
    LL_history = [LL]
    while (np.abs(LL - LL_old) != 0):
        # E Step
        R = em_e_estep(X, MU, COV, pi)
        # M Step
        MU, COV, pi = em_m_estep(X, R)
        # cost
        LL_old = LL
        LL = log_ikelihood(X, pi, MU, COV)
        print('epoch: %d | log-likelihood: %f'  % (n_step, LL))
        LL_history.append(LL)
        n_step += 1
        
### Cost Curve

    plt.title('log-likelihood vs. iteration')
    plt.ylabel('log-likelihood')
    plt.xlabel('iteration')
    plt.plot(np.arange(n_step + 1), LL_history, 'm-')
    plt.show()
    
    print('In %d steps, EM converges to the maximized log-likelihood: %f'  %(n_step, LL))
        
### Plot Result

    x = X[:,0]
    y = X[:,1]
    
    id1 = np.where(np.argmax(R, axis = 1) == 0)[0]
    id2 = np.where(np.argmax(R, axis = 1) == 1)[0]
    
    x1 = x[id1]
    y1 = y[id1]
    plt.plot(x1, y1, 'x')    
    plt.plot(MU[0,0], MU[0,1], 'x', color='red')
    
    x2 = x[id2]
    y2 = y[id2]
    plt.plot(x2, y2, 'o')    
    plt.plot(MU[1,0], MU[1,1], 'o', color='red')
    
    plt.title('SEED %d' %(SEED))
    plt.axis('equal')
    plt.show()

### Miss-Classification Error
    
    missed_rate = 1 - accuracy(X, R)
    print('The miss-classification error is: %.2f'  % missed_rate) # 93% aaccuracy
    
    

