#! /usr/bin/env python

# Copyright (C) 2014-2015 The BET Development Team

# import necessary modules
import numpy as np
import bet.sampling.adaptiveSampling as asam
import scipy.io as sio
from scipy.interpolate import griddata

sample_save_file = 'sandbox2d'

# Set minima and maxima
lam_domain = np.array([[.07, .15], [.1, .2]])
param_min = lam_domain[:, 0]
param_max = lam_domain[:, 1]

# Select only the stations I care about this will lead to better sampling
station_nums = [0, 5] # 1, 6

# Create Transition Kernel
transition_set = asam.transition_set(.5, .5**5, 1.0)

# Read in Q_ref and Q to create the appropriate rho_D 
mdat = sio.loadmat('Q_2D')
Q = mdat['Q']
Q = Q[:, station_nums]
Q_ref = mdat['Q_true']
Q_ref = Q_ref[15, station_nums] # 16th/20
bin_ratio = 0.15
bin_size = (np.max(Q, 0)-np.min(Q, 0))*bin_ratio

# Create experiment model
points = mdat['points']
def model(inputs):
    interp_values = np.empty((inputs.shape[0], Q.shape[1])) 
    for i in xrange(Q.shape[1]):
        interp_values[:, i] = griddata(points.transpose(), Q[:, i],
                inputs)
    return interp_values 

# Create kernel
maximum = 1/np.product(bin_size)
def rho_D(outputs):
    rho_left = np.repeat([Q_ref-.5*bin_size], outputs.shape[0], 0)
    rho_right = np.repeat([Q_ref+.5*bin_size], outputs.shape[0], 0)
    rho_left = np.all(np.greater_equal(outputs, rho_left), axis=1)
    rho_right = np.all(np.less_equal(outputs, rho_right), axis=1)
    inside = np.logical_and(rho_left, rho_right)
    max_values = np.repeat(maximum, outputs.shape[0], 0)
    return inside.astype('float64')*max_values

kernel_rD = asam.rhoD_kernel(maximum, rho_D)

# Create sampler
chain_length = 125
num_chains = 80
num_samples = chain_length*num_chains
sampler = asam.sampler(num_samples, chain_length, model)

# Get samples
inital_sample_type = "lhs"
(samples, data, all_step_ratios) = sampler.generalized_chains(param_min, param_max,
        transition_set, kernel_rD, sample_save_file, inital_sample_type)

# Read in points_ref and plot results
p_ref = mdat['points_true']
p_ref = p_ref[5:7, 15]

        
