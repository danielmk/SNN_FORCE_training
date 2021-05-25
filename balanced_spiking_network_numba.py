# -*- coding: utf-8 -*-
"""
Spyder Editor

Dies ist eine temporäre Skriptdatei.
"""

import numpy as np
from numpy.random import rand, randn
import matplotlib.pyplot as plt
import numba as nb

np.random.seed(100)

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

#@nb.njit()
def balanced_spiking_network(dt=0.00005,
                             T=2.0,
                             tref=0.002,
                             tm=0.01,
                             vreset=-65.0,
                             vpeak=-40.0,
                             n=2000,
                             td=0.02,
                             tr=0.002,
                             p=0.1,
                             offset=-40,
                             g=0.04,
                             nrec=10):

    # Seeding randomness for reproducibility
    nt = round(T/dt)  # Number of time steps
    # Set the row mean to zero
    w = normalize_weights(n, p, g)

    """Preinitialize recording"""
    rec = np.zeros((nt, nrec))

    """Initial conditions"""
    ipsc = np.zeros(n)  # Post synaptic current storage variable
    hm = np.zeros(n)  # Storage variable for filtered firing rates
    tlast = np.zeros((n))  # This vector is used to set  the refractory times
    v = vreset + rand(n)*(30-vreset)  # Initialize neuron voltage

    """Start integration loop"""
    for i in range(0, nt, 1):
        # inp = ipsc + offset  # Total input current

        # Voltage equation with refractory period
        # Only change if voltage outside of refractory time period
        dv = (dt * i > tlast + tref) * (-v + ipsc + offset) / tm
        v = v + dt * dv

        spikes = find_spikes(v, vpeak)  # Find the neurons that have spiked

        # Store spike times, and get the weight matrix column sum of spikers
        # Compute the increase in current due to spiking
        JD = w[:, np.argwhere(spikes)[:, 0]].sum(axis=1)
    
        # Used to set the refractory period of LIF neurons
        # tlast = tlast + (dt * i - tlast) * np.array(v >= vpeak, dtype=int)
        tlast = tlast + (dt * i - tlast) * spikes
    
        ipsc = ipsc * np.exp(-dt / tr) + hm * dt
    
        # Integrate the current
        hm = hm * np.exp(-dt / td) + JD * spikes.any() / (tr * td)
    
        v = v + (30 - v) * spikes
    
        rec[i, :] = v[0:nrec]  # Record a random voltage
        v = v + (vreset - v) * spikes
        
    return rec, w
rec, w = balanced_spiking_network()