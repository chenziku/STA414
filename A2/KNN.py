#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 14:15:39 2019

@author: zikunchen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 11:08:29 2019

@author: zikunchen
"""

import matplotlib.pyplot as plt
import numpy as np

def binarize(train_x):
    train_x[train_x >= 0.5] = 1
    train_x[train_x < 0.5] = 0
    return train_x

def cost(X, R, MU):
    J = 0
    for k in range(K):
        for n in range(N):
            J += R[n,k] * np.linalg.norm(X[n,] - MU[k,])**2
    return J      
    
def km_e_estep(X, MU):
    N = X.shape[0]
    K = MU.shape[0]
    R = np.zeros((N, K)) # N x K
    for k in range(K):
        for n in range(N):
            l = [np.linalg.norm(X[n,] - MU[k,])**2 for k in range(K)]
            k_min = np.argmin(l)
            R[n,k] = 1 if k == k_min else 0
    return R
    
def km_m_estep(X, R):                                                                                            
    # X: 400 x 2
    # R: 400 x 2
    K = R.shape[1]
    MU = np.zeros((K, D)) # K x D
    K = R.shape[1]
    for k in range(K):
        numer = np.matmul(X.T, R[:, k]) # 784 x 1
        denom = np.sum(R[:, k])
        MU[k] = numer/denom
    return MU

def accuracy(X, R):
    
    N_data = X.shape[0]
    cutoff = int(N_data/2)
    
    true_id1 = np.arange(cutoff)
    true_id2 = np.arange(cutoff, N_data)
    
    R_id1 = np.where(R[:,0] == 1)[0]
    R_id2 = np.where(R[:,0] == 0)[0]
    
    correct_count_1 = len(list(set(true_id1).intersection(R_id1)))
    correct_count_2 = len(list(set(true_id2).intersection(R_id2)))
    
    return (correct_count_1 + correct_count_2) / N_data

    

if __name__ == "__main__":
    
# =============================================================================
#     4 a) Data Generation
# =============================================================================
    
    m1 = [0.1, 0.1]
    m2 = [6.0, 0.1]
    cov = [[10, 7], [7, 10]]
    
    np.random.seed(2019)
    
    x1, y1 = np.random.multivariate_normal(m1, cov, 200).T
    x2, y2 = np.random.multivariate_normal(m2, cov, 200).T
    plt.plot(x1, y1, 'x')
    plt.plot(m1[0], m1[1], 'x', color='red')

    plt.plot(x2, y2, 'o')
    plt.plot(m2[0], m2[1], 'o', color='red')
    plt.axis('equal')
    
    plt.show()
    
# =============================================================================
#     4 b) KNN
# =============================================================================

### Training
    
    K = 2
    D = 2
    N = 400
    
    m1_init = [0.0, 0.0]
    m2_init = [1.0, 1.0]
        
    X = np.vstack((np.stack((x1, y1)).T, np.stack((x2, y2)).T))
    MU = np.array([m1_init, m2_init])
    R = km_e_estep(X, MU)
    
    n_step = 0
    J_old = 0
    J = cost(X, R, MU)
    cost_history = [J]
    
    while (J - J_old != 0):
        # E Step
        R = km_e_estep(X, MU)
        # M Step
        MU = km_m_estep(X, R)
        # cost
        J_old = J
        J = cost(X, R, MU)
        print('epoch: %d | cost: %f'  % (n_step, J))
        cost_history.append(J)
        n_step += 1

### Cost Curve

    plt.title('cost vs. iteration')
    plt.ylabel('cost')
    plt.xlabel('iteration')
    plt.plot(np.arange(n_step + 1), cost_history, 'mo-')
    plt.show()
        
    print('In %d steps, KNN converges to the minimized cost: %f'  %(n_step, J))

### Plot Result

    x = X[:,0]
    y = X[:,1]
    
    id1 = np.where(R[:,0] == 1)[0]
    id2 = np.where(R[:,0] == 0)[0]
    
    x1 = x[id1]
    y1 = y[id1]
    plt.plot(x1, y1, 'x')    
    plt.plot(MU[0,0], MU[0,1], 'x', color='red')
    
    x2 = x[id2]
    y2 = y[id2]
    plt.plot(x2, y2, 'o')    
    plt.plot(MU[1,0], MU[1,1], 'o', color='red')
    
    plt.axis('equal')
    plt.show()

### Miss-Classification Error
    
    missed_rate = 1 - accuracy(X, R)
    print('The miss-classification error is: %.2f'  % missed_rate)
    
    
    
    
    
    
    
