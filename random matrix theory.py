# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 22:10:37 2015

@author: Anurag
"""
Random Matrix Theory
#
# Version: 0.1

import numpy as np
import scipy.linalg as LA
import matplotlib.pyplot as plt
import scipy as sp

n = 100  # size of matrices
t = 5000  # number of samples
v = np.empty((t, n))  # eigenvalue samples
v1 = np.empty(t)  # max eigenvalue samples
delta = 0.2  # histogram bin width


for i in range(t):

 
    # sample from GOE
    matrix = [np.random.normal(loc=0.0, scale=1.0, size=None)  for j in range(n*n)]
    a = np.array(matrix).reshape((n,n))
    
    
    s = (a + a.T) / 2

 
    # compute eigenvalues
    #evals = LA.eigvals(s)
    w = LA.eigvals(s.T)
    
    evals = w

    # store eigenvalues
    v[i, :] = evals

 
    # sample from GUE
    
    matrix1 = [np.random.normal(loc=0.0, scale=1.0, size=None)  for j in range(n*n)]
    a1 =  np.array(matrix1).reshape((n,n))
    matrix2 = [np.random.normal(loc=0.0, scale=1.0, size=None)  for j in range(n*n)]
    a2 =  np.array(matrix2).reshape((n,n))
    a = a1 + 1j*a2
    s = (a + a.conj().T) / 2
    #hermit_a =  a.conjugate()
    #s = (a + hermit_a)/2

  
    # compute eigenvalues
    w = LA.eigvals(s)
   
    evals = w
    # store maximum eigenvalue
    v1[i] = np.amax(evals)



# normalize v
v = v/np.sqrt(n / 2)


# set histogram bin values to a numpy array containing [-2, -2+delta,
# -2+2*delta, ..., 2]
# Note: both 2 and -2 are to be included
#bins = np.linspace(-2, 2, num=21, endpoint=True)
bins = np.arange(-2,2.2,0.2)


hist, bin_edges = np.histogram(v, bins=bins)
lower = sp.resize(bin_edges, len(bin_edges)-1)
tmid = lower + 0.5*sp.diff(bin_edges)

# plot bar chart
plt.bar(lower, hist/(t*n*delta), width=delta, facecolor='y')

# plot theoretical prediction, i.e., the semicircle law
plt.plot(bin_edges, np.sqrt(4-bin_edges**2)/(2*np.pi), linewidth=2)

# set axes and save to pdf
plt.ylim([0, .5])
plt.xlim([-2.5, 2.5])
plt.savefig('Semicircle.pdf')
plt.close()


# normalize v1
v1 = (v1 - 2*np.sqrt(n)) * n**(1./6)

# TASK 1.6.1
# set histogram bin values to a numpy array containing [-5, -5+delta,
# -5+2*delta, ..., 2]
# Note: both -5 and 2 are to be included

bins = np.arange(-5,2.2,0.2)

# compute histogram
hist, bin_edges = np.histogram(v1, bins=bins)


# plot bar chart
lower = sp.resize(bin_edges, len(bin_edges)-1)
plt.bar(bin_edges[:-1], hist/(t*delta), width=delta, facecolor='y')

# load theoretical prediction, i.e., the Tracy-Widom law, from file
prediction = np.loadtxt('tracy-widom.csv', delimiter=',')

# plot Tracy-Widom law
plt.plot(prediction[:, 0], prediction[:, 1], linewidth=2)

# set axes and save to pdf
plt.ylim([0, .5])
plt.xlim([-5, 2])
plt.savefig('Tracy-Widom.pdf')
plt.close()