#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 17:06:20 2019

@author: zikunchen
"""

import loadMNIST as load
import matplotlib.pyplot as plt
import numpy as np


def binarize(train_x):
    train_x[train_x >= 0.5] = 1
    train_x[train_x < 0.5] = 0
    return train_x    

def fit(train_x, train_c):
    D = train_x.shape[1]
    K = train_c.shape[1]
    # Number of images that belongs to each class
    class_counts = np.sum(train_c, axis = 0)
    thetas = np.empty([K, D])
    for c in range(K):
        indices_c = np.where(np.argmax(train_c, axis =1) == c)[0]
        images_c = train_x[indices_c]
        pixel_sums_c = np.sum(images_c, axis = 0) # pixel sums of each class
        thetas[c] = (1 + pixel_sums_c) / (2 + class_counts[c]) 
    return thetas 
    
def image_LL(image, label, thetas, pis):
    D = len(image)
    marginal = 0
    for c in range(10):
        product = 1
        for d in range(D):
            product *= thetas[c,d]**image[d] * (1-thetas[c,d])**(1-image[d])
        if label == c:
            joint = pis[c] * product
        marginal += pis[c] * product 
    return joint/marginal

def avg_LL(images, labels, thetas, pis):
    sum_LL = 0
    N_data = images.shape[0]
    for n in range(N_data):
        image = images[n]
        label = np.argmax(labels[n])
        sum_LL += np.log(image_LL(image, label, thetas, pis))
    return (1/N_data) * sum_LL

def perdict(image, thetas, pis):
    D = len(image)
    K = len(pis)
    joint_dists = [0]*K
    for c in range(K):
        product = 1
        for d in range(D):
            product *= thetas[c,d]**image[d] * (1-thetas[c,d])**(1-image[d])
        joint_dists[c] = pis[c] * product
    return np.argmax(joint_dists)

def accuracy(images, labels, thetas, pis):
    correct_count = 0
    N = images.shape[0]
    labels = np.argmax(labels, axis =1)
    for n in range(N):
        if perdict(images[n], thetas, pis) == labels[n]:
            correct_count += 1
    return correct_count/N_data

def sample_image(thetas, pis):
    D = thetas.shape[1]
    K = len(pis)
    c = np.random.choice(K, 1, p=pis)
    theta_c = thetas[c].flatten()
    image = np.zeros(D)
    for d in range(D):
        image[d] = np.random.binomial(1, p=theta_c[d])
    return image

def sample_half(image, thetas, pis):
    D = thetas.shape[1]
    K = len(pis)
    class_sum = np.zeros(K)
    cut = int(D/2)
    top = range(cut)
    bottom = range(cut, D)
    # denominator
    for c in range(K):
        Theta_X = thetas[c, top]**image[top] \
            * (1 - thetas[c, top])**(1 - image[top])
        class_sum[c] = np.prod(Theta_X)
    denom = np.sum(class_sum)
    # numerator
    numer = np.zeros(cut)
    for d in bottom:
        for c in range(K):
            numer[d - cut] += class_sum[c] * thetas[c,d]
    image[bottom] = numer/denom
    return image


if __name__ == "__main__":
    
# =============================================================================
#     Load and Store Dataset
# =============================================================================
    
    training_cutoff = 10000
    debug_cutoff = 100
    image_size = 784
    
    N_data, train_images, train_labels, \
        test_images, test_labels = load.load_mnist()
    
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
#     1 c) Thetas
# =============================================================================

    thetas = fit(train_x, train_c)

    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)
    load.plot_images(thetas, ax)
    plt.show()
    
# =============================================================================
#     1 e) Average Likelihood and Accuracy
# =============================================================================

    pis = [0.1]*10
    avg_LL_train = avg_LL(train_x, train_c, thetas, pis)
    avg_LL_test = avg_LL(test_x, test_c, thetas, pis)
    accuracy_train = accuracy(train_x, train_c, thetas, pis)
    accuracy_test = accuracy(test_x, test_c, thetas, pis)
    
    print('Naive Bayes Model:')
    print('Average Log-Likelihood on Training Set:', avg_LL_train)
    print('Average Log-Likelihood on Test Set:', avg_LL_test)
    print('Training Accuracy:', accuracy_train)
    print('Testing Accuracy:', accuracy_test)
    
# =============================================================================
#     2 c) Sample 10 binary images
# =============================================================================
    
    print('Sample 10 Images from Marginal Distribution p(x)')
    
    ten_samples = [sample_image(thetas, pis) for _ in range(10)]
    ten_samples = np.array(np.stack(ten_samples, axis=0))
    
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)
    load.plot_images(ten_samples, ax)
    plt.show()

# =============================================================================
#     2 e) Generate 20 half images
# =============================================================================

    print('Sample 20 half images from Training Set with p(bottom | top)')

    images = train_x[np.random.choice(image_size,20),:]
    
    twenty_half_samples = [sample_half(image, thetas, pis) for image in images]
    twenty_half_samples = np.array(np.stack(twenty_half_samples, axis=0))
    
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)
    load.plot_images(twenty_half_samples, ax)
    plt.show()

    
    
    