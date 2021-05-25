# -*- coding: utf-8 -*-
"""
This script simulates a recurrent spiking neural network and uses the FORCE
training algorithm to generate a sine wave at the network output.

This is a Python port of a MATLAB implementation from Nicola & Clopath (2017):
https://doi.org/10.1038/s41467-017-01827-3
"""

import numpy as np
from numpy.random import rand, randn
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sys
from sklearn.preprocessing import StandardScaler
import numba as nb

np.random.seed(100)  # Seeding randomness for reproducibility

@nb.njit()
def normalize_weights(n, p, g):
    w = g * (randn(n, n)) * (rand(n, n) < p) / (np.sqrt(n) * p)
    for i in range(w.shape[0]):
        count = 0
        summed = 0
        for j in range(w.shape[1]):
            if np.abs(w[i,j]) > 0:
                count += 1
                summed += w[i, j]
        for j in range(w.shape[1]):
            if np.abs(w[i,j]) > 0:
                w[i, j] = w[i, j] - (summed / count)
    return w

@nb.njit()
def find_spikes(v, vpeak):
    n = v.shape[0]
    arr = np.empty(n)
    for idx in range(n):
        if v[idx] >= vpeak:
            arr[idx] = 1
        else:
            arr[idx] = 0
    return arr

@nb.njit()
def force_training(target, rlms_start, rlms_stop):
    """NETWORK PARAMETERS"""
    dt = 0.00005
    N = 2000  # Number of neurons
    tref = 0.002  # Refractory time constant in seconds
    tm = 0.01  # Membrane time constant
    vreset = -65  # Voltage reset
    vpeak = -40  # Voltage peak
    td = 0.02  # Synaptic decay time constant
    tr = 0.002  # Synaptic rise time constant
    p = 0.1  # Set the network sparsity
    BIAS = vpeak  # Set the BIAS current, can help decrease/increase firing rates.
    g = 0.04  # Factor of fixed weight matrix
    nt = target.shape[0]
    
    """RLMS PARAMETERS"""
    m = 2
    alpha = dt*0.1  # Sets the rate of weight change
    Pinv = np.eye(N)*alpha  # Initialize the correlation weight matrix for RLMS
    # RECB = np.zeros((nt, 5))  # Storage matrix for some synaptic weights 
    # The initial weight matrix with fixed random weights
    BPhi = np.zeros((N, m))  # The initial matrix that will be learned by FORCE method
    w = normalize_weights(N, p, g)

    step = 30  # Interval of RLMS in indices
    Q = 10
    E = (2*rand(N, m)-1)*Q
    
    """PREINITIALIZE STORAGE"""
    IPSC = np.zeros(N)  # Post synaptic current storage variable
    hm = np.zeros(N)  # Storage variable for filtered firing rates
    r = np.zeros(N)  # Second storage variable for filtered rates
    hr = np.zeros(N)  # Third variable for filtered rates
    JD = 0*IPSC  # Storage variable required for each spike time
    z = np.zeros(m)  # Initialize the approximant
    err = np.zeros(m)
    tlast = np.zeros((N))  # This vector is used to set  the refractory times
    v = vreset + rand(N)*(30-vreset)  # Initialize neuron voltage
    #REC2 = np.zeros((nt, 5))
    REC = np.zeros((nt, 5))
    current = np.zeros((nt, m))
    
    """START INTEGRATION LOOP"""
    for i in range(0, nt, 1):
        # print(i)
        inp = IPSC + E @ z + BIAS  # Total input current

        # Voltage equation with refractory period
        dv = (dt * i > tlast + tref) * (-v + inp) / tm
        v = v + dt * dv

        index = np.argwhere(find_spikes(v, vpeak))[:, 0]  # Find the neurons that have spiked

        # Store spike times, and get the weight matrix column sum of spikers
        if len(index) > 0:
            # Compute the increase in current due to spiking
            JD = w[:, index].sum(axis=1)
        else:
            JD = 0*IPSC

        # Used to set the refractory period of LIF neurons
        tlast = tlast + (dt * i - tlast) * find_spikes(v, vpeak)

        IPSC = IPSC * np.exp(-dt / tr) + hm * dt

        # Integrate the current
        hm = hm * np.exp(-dt / td) + JD * (len(index) > 0) / (tr * td)

        r = r * np.exp(-dt / tr) + hr * dt
        hr = hr * np.exp(-dt / td) + find_spikes(v, vpeak) / (tr * td)

        # Implement RLMS with the FORCE method
        z = BPhi.T @ r  # Approximant
        err = z - target[i]  # Error
        # Only adjust at an interval given by step
        if np.mod(i, step) == 1:
            # Only adjust between rlmst_start and rlms_stop
            if (i > rlms_start) and (i < rlms_stop):
                cd = Pinv @ r
                BPhi = BPhi - (cd.reshape(cd.shape[0], 1) @ err.reshape(1, err.shape[0]))
                Pinv = Pinv - (cd.reshape(cd.shape[0], 1) @ cd.reshape(1, cd.shape[0])) / (1 + (r @ cd))

        v = v + (30 - v) * find_spikes(v, vpeak)

        REC[i, :] = v[0:5]  # Record a random voltage

        # Reset with spike time interpolant implemented
        v = v + (vreset - v) * find_spikes(v, vpeak)
        current[i] = z
        #RECB[i, :] = BPhi[0:5]
        #REC2[i, :] = r[0:5]

    return current, REC

"""PLOTTING"""
if __name__ == '__main__':
    plot = True
    dt = 0.00005
    data = wavfile.read('Awful - josh pan (online-audio-converter.com).wav')
    data_subset = data[1][1000000:1000000+data[0]*3,:]
    scaler = StandardScaler()
    scaler_fit = scaler.fit(data_subset)
    repeats = 5
    upsampling = 1
    zx = scaler_fit.transform(data_subset)
    rlms_start = round(1/dt)  # When to start RLMS
    rlms_stop = rlms_start + zx.shape[0] * repeats  # When to stop RLMS
    zx = np.concatenate((np.zeros((rlms_start, 2)),np.tile(zx.repeat(upsampling, axis=0).T, repeats).T))
    kernel_size = 30
    kernel = np.ones(kernel_size) / kernel_size
    zx[:,0] = np.convolve(zx[:,0], kernel, mode='same')
    zx[:,1] = np.convolve(zx[:,1], kernel, mode='same')
    freq = 1
    target = np.sin(freq * np.pi*np.arange(0, 2 * (1 / dt) / freq, 1) * dt)
    target = target.reshape((target.shape[0]), 1)
    repeats = 10
    zx = np.tile(target.T, repeats).T
    current, REC = force_training(zx, rlms_start, zx.shape[0] - target.shape[0])
    
    if plot:
        fig, ax = plt.subplots(3)
        ax[0].plot(current[:,0])
        ax[0].plot(zx[:,0])
        ax[2].plot(REC[:, 0])
