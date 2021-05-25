# -*- coding: utf-8 -*-
"""
Spyder Editor

Dies ist eine temporäre Skriptdatei.
"""

import numpy as np
from numpy.random import rand, randn
import matplotlib.pyplot as plt

np.random.seed(100)  # Seeding randomness for reproducibility

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
    
    nt = round(T/dt)  # Number of time steps
    # Set the row mean to zero
    w = g * (randn(n, n)) * (rand(n, n) < p) / (np.sqrt(n) * p)
    row_means = np.mean(w, axis=1, where=np.abs(w)>0)[:, np.newaxis]
    row_means_shaped = np.repeat(row_means, w.shape[0], axis=1)
    w[np.abs(w) > 0] = w[np.abs(w) > 0] - row_means_shaped[np.abs(w) > 0]
    
    """Preinitialize recording"""
    rec = np.zeros((nt, nrec))
    
    """Initial conditions"""
    ipsc = np.zeros(n)  # Post synaptic current storage variable
    hm = np.zeros(n)  # Storage variable for filtered firing rates
    tlast = np.zeros((n))  # This vector is used to set  the refractory times
    v = vreset + rand(n)*(30-vreset)  # Initialize neuron voltage
    
    """Start integration loop"""
    for i in np.arange(0, nt, 1):
        inp = ipsc + offset  # Total input current
    
        # Voltage equation with refractory period
        # Only change if voltage outside of refractory time period
        dv = (dt * i > tlast + tref) * (-v + inp) / tm
        v = v + dt*dv
    
        index = np.argwhere(v >= vpeak)[:, 0]  # Find the neurons that have spiked
    
        # Store spike times, and get the weight matrix column sum of spikers
        if len(index) > 0:
            # Compute the increase in current due to spiking
            JD = w[:, index].sum(axis=1)
        else:
            JD = 0*ipsc
    
        # Used to set the refractory period of LIF neurons
        tlast = tlast + (dt * i - tlast) * (v >= vpeak)
    
        ipsc = ipsc * np.exp(-dt / tr) + hm * dt
    
        # Integrate the current
        hm = hm * np.exp(-dt / td) + JD * (len(index) > 0) / (tr * td)
    
        v = v + (30 - v) * (v >= vpeak)
    
        rec[i, :] = v[0:nrec]  # Record a random voltage
        v = v + (vreset - v) * (v >= vpeak)
        
    return rec
rec = balanced_spiking_network()