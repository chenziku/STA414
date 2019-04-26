#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 11:41:31 2019

@author: zikunchen
"""
import numpy as np
import seaborn as sns
import scipy.stats as st
import scipy
import matplotlib.pyplot as plt

sns.set()

def pg(x, g, theta=0, sigma=4):
    result = scipy.sin(5*(x-theta))**2/(25*scipy.sin(x-theta)**2)
    return result if g == 1 else (1-result)

def px(x, theta=0, sigma=4):
    return st.norm.pdf(x, loc=theta, scale=sigma)

def smc(g, N, seed, theta=0, sigma=4):
    np.random.seed(SEED)
    X = np.random.normal(theta, sigma, N)
    return np.sum([pg(x, g) for x in X])/N

def rejection_sampler(M, N, seed, theta=0, sigma=4):
    np.random.seed(SEED)
    i = 0
    count = 0
    samples = np.empty(N)
    while i < N:
        x = np.random.normal(theta, sigma, 1)[0]
        u = np.random.uniform(0,1,1)[0]
        q_x = px(x, theta, sigma)
        p_x = pg(x, 1, theta, sigma)*q_x
        if u < p_x/(M*q_x):
            samples[i] = x
            i += 1
        count += 1
    acceptance = N / count
    return samples, acceptance

def importance_sampler(N, seed):
    return smc(0, N, seed)


def p_theta(theta, x=1.7, g=0, sigma=4):
    z = 1 + (theta/10)**2
    result = px(x, theta, sigma) * pg(x, g, theta, sigma)
    return result * 1 / (10 * np.pi * z)


def metro_hast(prop_sigma, N, seed, sigma=4, x=1.7, g=1):
    np.random.seed(seed)
    t = np.random.uniform(-20,20,1)[0]
    i = 0
    count = 0
    thetas = np.empty(N)
    while i < N:
        tnew = np.random.normal(t, prop_sigma, 1)[0]
        gtnew_t = st.norm.pdf(tnew, loc=t, scale=prop_sigma)
        gt_tnew = st.norm.pdf(t, loc=tnew, scale=prop_sigma)
        pt = p_theta(t, x, g)
        ptnew = p_theta(tnew, x, g)
        accept = min(1, (ptnew * gt_tnew)/(pt * gtnew_t))
        u = np.random.uniform(0,1,1)[0]
        if u < accept:
            t = tnew
            thetas[i] = t
            i += 1
        count += 1
    return thetas, acceptance
    

if __name__ == "__main__":
    theta = 0
    sigma = 4
    N = 10000
    SEED = 2019

# 1 a)
    # Quadrature (Direct Integration)
    x = np.linspace(start=-20, stop=20, num=N)    
    f = lambda x: px(x) * pg(x, 0)
    
    p_absorb = scipy.integrate.quadrature(f, -20, 20)[0]
    plt.plot(x, f(x))
    plt.show()
    
    print('Quadrature estimate of fraction of photons absorbed', \
          'average is %.3f' % (p_absorb))

# 1 b)
    samples, acceptance = rejection_sampler(1, 10000, SEED)
    
    sns.distplot(samples, kde = False)
    plt.show()
    print('Rejection Sampling: ')
    print('Fraction of accepted sample is %.3f' % (acceptance))

    
# 1 c)
    p_absorb_marginal = importance_sampler(1000, SEED)
    print('Importance Sampling:')
    print('Estimate of fraction of photons that get absorbed is %.3f'\
          % (p_absorb_marginal))

# 1 d)
    x = 1.7
    g = 1
    thetas = np.linspace(start=-20, stop=20, num=N)
    p_joint = p_theta(thetas, x, g)
    plt.plot(thetas, p_joint)
    plt.show()

# 1 e)
    sample_t, acceptance = metro_hast(3, 50000, SEED)
    sns.distplot(sample_t, kde = False)
    plt.show()

    print('Metropolis Hasting to sample theta:')
    print('Fraction of accepted sample is %.3f' % (acceptance))
    
# 1 f)
    p_abs_three =  len(sample_t[(sample_t<3) & (sample_t>-3)]) \
                                / len(sample_t)

    print('Posterior probability of theta between' \
          ' +3 and -3 is %.3f' % (p_abs_three))

    