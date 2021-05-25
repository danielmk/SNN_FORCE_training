# -*- coding: utf-8 -*-
"""
This is a Python port of a MATLAB implementation from Nicola & Clopath (2017):
https://doi.org/10.1038/s41467-017-01827-3
"""

import numpy as np
from numpy.random import rand, randn
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
import sys
from sklearn.preprocessing import StandardScaler
import numba as nb

  # Seeding randomness for reproducibility

@nb.njit()
def normalize_weights(n=2000, p=0.1, g=0.04):
    """Create a normalized weight matrix for n neurons"""
    weights = g * (randn(n, n)) * (rand(n, n) < p) / (np.sqrt(n) * p)
    for i in range(weights.shape[0]):
        count = 0
        summed = 0
        for j in range(weights.shape[1]):
            if np.abs(weights[i,j]) > 0:
                count += 1
                summed += weights[i, j]
        for j in range(weights.shape[1]):
            if np.abs(weights[i,j]) > 0:
                weights[i, j] = weights[i, j] - (summed / count)
    return weights

@nb.njit()
def find_spikes(v, vpeak):
    """Find voltages that exceed vpeak."""
    n = v.shape[0]
    arr = np.empty(n)
    for idx in range(n):
        if v[idx] >= vpeak:
            arr[idx] = 1
        else:
            arr[idx] = 0
    return arr

@nb.njit()
def rls_force_training(err, rd, cm, r):
    cd = cm @ r
    rd = rd - (cd.reshape(cd.shape[0], 1) @ err.reshape(1, err.shape[0]))
    cm = cm - (cd.reshape(cd.shape[0], 1) @ cd.reshape(1, cd.shape[0])) / (1 + (r @ cd))
    return rd, cm

@nb.njit()
def _force_train_lif(target, target_repeats=1, rls_step=30, rls_pre=1,
                     rls_post=1,
                     dt = 0.00005,
                     N = 2000, tref = 0.002, tm = 0.01, vreset = -65.0,
                     vpeak = -40.0, td = 0.02, tr = 0.002, p = 0.1,
                     BIAS = -40.0, g = 0.04, seed=100, hdts_states=100, initial_lr = 0.1):
    """FORCE train the target dynamic in a network of lif neurons.

    Supports high-dimensional temporal signals (HDTS).

    Parameters
    ----------
    target : ndarray (nt, m)
        The target dynamic that will be trained.
    target_repeats : int
        The number of times the target is repeated during the simulation.
    rls_step : float
        Interval of rls.
    rls_pre : float
        Time in seconds before target presentation.
    rls_post : float
        Time in seconds after target presentation.
    dt : float
        Sampling period of the simulation in seconds.
    N : int
        Number of neurons.
    tref : float
        Refractory period of the neuron.
    tm : float
        Membrane time constant of the neuron.
    vreset : float
        The reset voltage after a spike.
    vpeak : float
        The threshold voltage where a spike is registered and neuron reset.
    td : float
        Decay time constant of the synapse
    tr : float
        Rise time constant of the synapse
    p : float
        Connection probablity.
    BIAS : float
        Direct current input of all neurons.
    g : float
        Synaptic strength.
    seed : int
        Seed to make simulation deterministic.
    hdts_states : int
        The dimensionality of the HDTS.

    Returns
    -------
    tuple (current, REC, w, zx, target_out, rd)
        current : array 
    """

    np.random.seed(seed)
    tn = target.shape[0]
    m = target.shape[1]

    """Prepare rls"""
    rls_start = int(rls_pre / dt)
    rls_stop = int(rls_pre / dt) + target.shape[0] * target_repeats
    nt = int(rls_pre / dt) + target.shape[0] * target_repeats + int(rls_post / dt)
    alpha = dt * initial_lr # Initializes the rate of weight change
    cm = np.eye(N) * alpha  # Initialize the correlation weight matrix for RLMS
    rd = np.zeros((N, m))  # The initial matrix that will be learned by FORCE method
    weights = normalize_weights(N, p, g)

    Q = 10
    E1 = (2*rand(N, m)-1)*Q

    """Generate HDTS"""
    m2 = hdts_states
    tt = np.arange(0,tn,1)*dt  # Target time
    hdts_freq = int(m2 / (tn * dt))
    tb = np.abs(np.sin(hdts_freq*np.pi*tt))

    z2 = np.zeros((m2, target.shape[0]))
    for qw in range(m2):
        z2[qw,:] = tb * (tt >= (1/hdts_freq) * qw) * (tt < (1/hdts_freq) * (qw + 1))
        
    WE2 = 80
    E2 = (2*rand(N,m2)-1)*WE2 # HDTS input

    """PREINITIALIZE STORAGE"""
    IPSC = np.zeros(N)  # Post synaptic current storage variable
    hm = np.zeros(N)  # Storage variable for filtered firing rates
    r = np.zeros(N)  # Second storage variable for filtered rates
    hr = np.zeros(N)  # Third variable for filtered rates
    JD = 0*IPSC  # Storage variable required for each spike time
    z1 = np.zeros(m)  # Initialize the approximant
    err = np.zeros(m)
    tlast = np.zeros((N))  # This vector is used to set  the refractory times
    REC = np.zeros((nt, 5))
    current = np.zeros((nt, m))
    v = vreset + rand(N)*(30-vreset)  # Initialize neuron voltage
    it = 0  # The index for the target
    target_out = np.zeros(nt)

    """Integration loop"""
    for i in range(0, nt, 1):
        # z2it = z2[:,it % target.shape[0]]
        # inp = IPSC + E1 @ z1 + E2 @ z2[:,it] + BIAS  # Total input current
        inp = IPSC + E1 @ z1 + (E2 * z2[:,it % target.shape[0]]).sum(axis=1) + BIAS

        # Voltage equation with refractory period
        dv = (dt * i > tlast + tref) * (-v + inp) / tm
        v = v + dt * dv

        index = np.argwhere(find_spikes(v, vpeak))[:, 0]  # Find the neurons that have spiked

        # Store spike times, and get the weight matrix column sum of spikers
        JD = weights[:, index].sum(axis=1)

        # Used to set the refractory period of LIF neurons
        tlast = tlast + (dt * i - tlast) * find_spikes(v, vpeak)

        IPSC = IPSC * np.exp(-dt / tr) + hm * dt

        # Integrate the current
        hm = hm * np.exp(-dt / td) + JD * (len(index) > 0) / (tr * td)

        r = r * np.exp(-dt / tr) + hr * dt
        hr = hr * np.exp(-dt / td) + find_spikes(v, vpeak) / (tr * td)

        z1 = rd.T @ r  # Approximant
        err = z1 - target[it % target.shape[0]]  # Error
        target_out[i] = target[it % target.shape[0],0]
        # Only adjust at an interval given by step
        if (i > rls_start):
            if (np.mod(it, rls_step) == 0) and (i < rls_stop):
                rd, cm = rls_force_training(err, rd, cm, r)
            it += 1

        v = v + (30 - v) * find_spikes(v, vpeak)

        REC[i, :] = v[0:5]  # Record a random voltage

        # Reset with spike time interpolant implemented
        v = v + (vreset - v) * find_spikes(v, vpeak)
        current[i] = z1
        #break
    #print(locals().keys())

    return current, REC, weights, target_out, rd

"""PLOTTING"""
if __name__ == '__main__':
    plot = True
    dt = 0.00005
    nt = 100000
    rlms_start = nt / 10
    rlms_stop = nt - (nt / 10)
    freq = 5
    target = np.sin(freq * np.pi*np.arange(0, 2 * (1 / dt) / freq, 1) * dt)
    target = target.reshape((target.shape[0]), 1)
    # repeats = 20
    # zx = np.tile(target.T, repeats).T
    current, REC, weights, target_out, rd = _force_train_lif(target, 20, rls_step=50, rls_pre=1, rls_post=5)
    
    if plot:
        fig, ax = plt.subplots(3)
        ax[0].plot(current[:,0])
        ax[0].plot(target_out)
        # ax[1].plot(current[:,1])
        # ax[1].plot(zx[:,1])
        ax[2].plot(REC[:, 0])
