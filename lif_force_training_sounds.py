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

np.random.seed(100)  # Seeding randomness for reproducibility

"""Load .wav file"""
data = wavfile.read('Awful - josh pan (online-audio-converter.com).wav')
data_subset = data[1][1000000:1000000+data[0]*3,:]
scaler = StandardScaler()
scaler_fit = scaler.fit(data_subset)
data_scaled = scaler_fit.transform(data_subset)

"""SIMULATION PARAMETERS"""
plot = True
dt = 0.00005  # Sampling interval of the simulation
#T = 15
#nt = round(T/dt)

"""NETWORK PARAMETERS"""
N = 2000  # Number of neurons
tref = 0.002  # Refractory time constant in seconds
tm = 0.01  # Membrane time constant
vreset = -65  # Voltage reset
vpeak = -40  # Voltage peak
td = 0.02  # Synaptic decay time constant
tr = 0.002  # Synaptic rise time constant
p = 0.1  # Set the network sparsity
BIAS = vpeak  # Set the BIAS current, can help decrease/increase firing rates.
G = 0.04  # Factor of fixed weight matrix

"""Sound target"""
repeats = int(0.010 /dt)
zx = data_scaled
rlms_start = round(1/dt)  # When to start RLMS
rlms_stop = rlms_start + zx.shape[0] * 5  # When to stop RLMS
nt = rlms_start + zx.shape[0] * 5 + zx.shape[0]
zx = np.concatenate((np.zeros((rlms_start, 2)),np.tile(zx.T, 6).T))
# sys.exit()

"""RLMS PARAMETERS"""
m = 2
alpha = dt*0.1  # Sets the rate of weight change
Pinv = np.eye(N)*alpha  # Initialize the correlation weight matrix for RLMS
# RECB = np.zeros((nt, 5))  # Storage matrix for some synaptic weights 
# The initial weight matrix with fixed random weights
OMEGA = G * (randn(N, N)) * (rand(N, N) < p) / (np.sqrt(N) * p)
BPhi = np.zeros((N, m))  # The initial matrix that will be learned by FORCE method
# Set the row average weight to be zero, explicitly
for i in np.arange(0, N, 1): 
    QS = np.argwhere(abs(OMEGA[i, :])>0)
    OMEGA[i, QS] = OMEGA[i, QS] - sum(OMEGA[i, QS])/len(QS)

step = 50  # Interval of RLMS in indices
Q = 10
E = (2*rand(N, m)-1)*Q

"""PREINITIALIZE STORAGE"""
k = 1
IPSC = np.zeros(N)  # Post synaptic current storage variable
hm = np.zeros(N)  # Storage variable for filtered firing rates
r = np.zeros(N)  # Second storage variable for filtered rates
hr = np.zeros(N)  # Third variable for filtered rates
JD = 0*IPSC  # Storage variable required for each spike time
tspike = np.zeros((4*nt+1, 2))  # Storage variable for spike times
ns = 0  # Number of spikes, counts during simulation
z = np.zeros(m)  # Initialize the approximant
tlast = np.zeros((N))  # This vector is used to set  the refractory times
v = vreset + rand(N)*(30-vreset)  # Initialize neuron voltage
#REC2 = np.zeros((nt, 5))
REC = np.zeros((nt, 5))
current = np.zeros((nt, m))

"""START INTEGRATION LOOP"""
for i in np.arange(0, nt, 1):
    # print(i)
    inp = IPSC + E @ z + BIAS  # Total input current

    # Voltage equation with refractory period
    dv = (dt * i > tlast + tref) * (-v + inp) / tm
    v = v + dt*dv

    index = np.argwhere(v >= vpeak)[:, 0]  # Find the neurons that have spiked

    # Store spike times, and get the weight matrix column sum of spikers
    if len(index) > 0:
        # Compute the increase in current due to spiking
        JD = OMEGA[:, index].sum(axis=1)
        #tspike[ns:ns+len(index), :] = np.array([index, 0*index+dt*i]).T
        ns = ns + len(index)  # total number of psikes so far

    else:
        JD = 0*IPSC

    # Used to set the refractory period of LIF neurons
    tlast = tlast + (dt * i - tlast) * np.array(v >= vpeak, dtype=int)

    IPSC = IPSC * np.exp(-dt / tr) + hm * dt

    # Integrate the current
    hm = hm * np.exp(-dt / td) + JD * (int(len(index) > 0)) / (tr * td)

    r = r * np.exp(-dt / tr) + hr * dt
    hr = hr * np.exp(-dt / td) + np.array(v >= vpeak, dtype=int) / (tr * td)

    # Implement RLMS with the FORCE method
    z = BPhi.T @ r  # Approximant
    err = z - zx[i]  # Error
    # Only adjust at an interval given by step
    if np.mod(i, step) == 1:
        # Only adjust between rlmst_start and rlms_stop
        if (i > rlms_start) and i < rlms_stop:
            cd = np.matmul(Pinv, r)
            BPhi = BPhi - (cd[:, None] @ err[None, :])
            Pinv = Pinv - (cd[:, None] @ cd[None, :]) / (1 + (r @ cd))

    v = v + (30 - v) * (v >= vpeak)

    REC[i, :] = v[0:5]  # Record a random voltage

    # Reset with spike time interpolant implemented
    v = v + (vreset - v) * (v >= vpeak)
    current[i] = z
    #RECB[i, :] = BPhi[0:5]
    #REC2[i, :] = r[0:5]

"""PLOTTING"""
if plot:
    fig, ax = plt.subplots(3)
    ax[0].plot(current[:,0])
    ax[0].plot(zx[:,0])
    ax[1].plot(current[:,1])
    ax[1].plot(zx[:,1])
    ax[2].plot(REC[:, 0])
