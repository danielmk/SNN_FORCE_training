# -*- coding: utf-8 -*-
"""
This is a Python port of a MATLAB implementation from Nicola & Clopath (2017):
https://doi.org/10.1038/s41467-017-01827-3
"""

import numpy as np
from numpy.random import rand, randn
import matplotlib.pyplot as plt
import numba as nb


@nb.njit()
def normalized_weights(n=2000, p=0.1, g=0.04):
    """Create a normalized weight matrix for n neurons"""
    weights = g * (randn(n, n)) * (rand(n, n) < p) / (np.sqrt(n) * p)
    for i in range(weights.shape[0]):
        count = 0
        summed = 0
        for j in range(weights.shape[1]):
            if np.abs(weights[i, j]) > 0:
                count += 1
                summed += weights[i, j]
        for j in range(weights.shape[1]):
            if np.abs(weights[i, j]) > 0:
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
def _force_train_lif(
        target,
        target_repeats=1,
        rls_step=30,
        rls_pre=1,
        rls_post=1,
        dt=0.00005,
        N=2000,
        tref=0.002,
        tm=0.01,
        vreset=-65.0,
        vpeak=-40.0,
        td=0.02,
        tr=0.002,
        p=0.1,
        BIAS=-40.0,
        g=0.04,
        seed=100,
        hdts_states=100,
        initial_lr=0.1,
        perturbation_factor=10,
        hdts_factor=80):

    """FORCE train the target dynamic in a network of lif neurons.

    Supports high-dimensional temporal signals (HDTS).

    Parameters
    ----------
    target : ndarray (time, target_dim)
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
    tuple (decoded_output,
           v_recording,
           weights_fixed,
           target_rec,
           rd,
           E1,
           E2,
           hdts)
    decoded_output : array (time, target_dim)
        The decoded output at each time point.
    v_recording : array (time, 5)
        Voltage recordings from the first five neurons.
    weights_fixed : array (N, N)
        The stable weight matrix.
    target_rec : array(time, target_dim)
        Recording of the target signal over time.
    rd : array (N, target.shape[1])
        The learned perturbation matrix after training
    E1 : array (N, target.shape[1])
        The stable decoder of the perturbation matrix.
    E2 : array (N, hdts_states)
        The stable decoder of the HDTS.
    hdts : array (hdts_states, target.shape[0])
        The HDTS.
        
    The outputs should be sufficient to reproduce the network and its dynamics
    without further training.
    
    Examples
    --------
    >>> dt = 0.00005
    >>> f = 5
    >>> target = np.sin(f * np.pi * np.arange(0, 2 * (1 / dt) / f, 1) * dt)
    >>> target = target.reshape((target.shape[0]), 1)
    >>> res = _force_train_lif(target, 20, dt=dt)
    >>> (decoded_output, v_recording,
         weights_fixed, target_rec, rd, E1, E2, hdts) = res
    """

    # Seed for reproducibility
    np.random.seed(seed)

    """Prepare rls"""
    rls_start = int(rls_pre / dt)
    rls_stop = rls_start + target.shape[0] * target_repeats
    t_len = rls_stop + int(rls_post / dt)
    alpha = dt * initial_lr  # Initializes the rate of weight change
    cm = np.eye(N) * alpha  # Initialize the correlation weight matrix for RLMS
    rd = np.zeros((N, target.shape[1]))  # The weights to train
    weights_fixed = normalized_weights(N, p, g)

    # Perturbation matrix
    E1 = (2 * rand(N, target.shape[1]) - 1) * perturbation_factor  

    """Generate HDTS"""
    target_time = np.arange(0, target.shape[0], 1) * dt
    hdts_freq = int(hdts_states / (target.shape[0] * dt))
    temporal_backbone = np.abs(np.sin(hdts_freq * np.pi * target_time))

    hdts = np.zeros((hdts_states, target.shape[0]))
    for qw in range(hdts_states):
        hdts[qw, :] = (
            temporal_backbone *
            (target_time >= (1 / hdts_freq) * qw) *
            (target_time < (1 / hdts_freq) * (qw + 1))
            )

    E2 = (2 * rand(N, hdts_states) - 1) * hdts_factor  # HDTS matrix

    """Preinitialize storage"""
    synaptic_currents = np.zeros(N)  # Post synaptic current storage variable
    hm = np.zeros(N)  # Storage variable for filtered firing rates
    r = np.zeros(N)  # Second storage variable for filtered rates
    hr = np.zeros(N)  # Third variable for filtered rates
    JD = 0 * synaptic_currents  # Storage variable required for each spike time
    z1 = np.zeros(target.shape[1])  # Initialize the approximant
    err = np.zeros(target.shape[1])
    tlast = np.zeros((N))  # This vector is used to set  the refractory times
    v_recording = np.zeros((t_len, 5))
    decoded_output = np.zeros((t_len, target.shape[1]))
    v = vreset + rand(N) * (30 - vreset)  # Initialize neuron voltage
    it = 0  # The index to cycle through the target
    target_rec = np.zeros(t_len)  # Record the target

    """Integration loop"""
    for i in range(0, t_len, 1):

        inp = (
            synaptic_currents +
            E1 @ z1 +
            (E2 * hdts[:, it % target.shape[0]]).sum(axis=1) +
            BIAS
            )

        # Voltage equation with refractory period
        dv = (dt * i > tlast + tref) * (-v + inp) / tm
        v = v + dt * dv

        spike_idc = np.argwhere(find_spikes(v, vpeak))[:, 0]

        # Store spike times, and get the weight matrix column sum of spikers
        JD = weights_fixed[:, spike_idc].sum(axis=1)

        # Used to set the refractory period of LIF neurons
        tlast = tlast + (dt * i - tlast) * find_spikes(v, vpeak)

        synaptic_currents = synaptic_currents * np.exp(-dt / tr) + hm * dt

        # Integrate the current
        hm = hm * np.exp(-dt / td) + JD * (len(spike_idc) > 0) / (tr * td)

        r = r * np.exp(-dt / tr) + hr * dt
        hr = hr * np.exp(-dt / td) + find_spikes(v, vpeak) / (tr * td)

        z1 = rd.T @ r  # Approximant
        err = z1 - target[it % target.shape[0]]  # Error
        target_rec[i] = target[it % target.shape[0], 0]
        # Only adjust at an interval given by step
        if i > rls_start:
            if (np.mod(it, rls_step) == 0) and (i < rls_stop):
                rd, cm = rls_force_training(err, rd, cm, r)
            it += 1

        v = v + (30 - v) * find_spikes(v, vpeak)

        v_recording[i, :] = v[0:5]

        # Reset with spike time interpolant implemented
        v = v + (vreset - v) * find_spikes(v, vpeak)
        decoded_output[i] = z1
        # break
    # print(locals().keys())

    return (decoded_output,
            v_recording,
            weights_fixed,
            target_rec,
            rd,
            E1,
            E2,
            hdts)


"""PLOTTING"""
if __name__ == "__main__":
    plot = True
    dt = 0.00005
    freq = 5
    target = np.sin(freq * np.pi * np.arange(0, 2 * (1 / dt) / freq, 1) * dt)
    target = target.reshape((target.shape[0]), 1)
    res = _force_train_lif(target, 20, dt=dt)
    (decoded_output, v_recording,
     weights_fixed, target_rec, rd, E1, E2, hdts) = res

    if plot:
        fig, ax = plt.subplots(3)
        ax[0].plot(decoded_output[:, 0])
        ax[0].plot(target_rec)
        ax[2].plot(v_recording[:, 0])
