# -*- coding: utf-8 -*-
"""
This script simulates a recurrent spiking neural network and uses the FORCE
training algorithm to generate a Hodgkin-Huxley system at the network output.

This is a Python port of a MATLAB implementation from Nicola & Clopath (2017):
https://doi.org/10.1038/s41467-017-01827-3
"""

import numpy as np
from numpy.random import rand, randn
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


np.random.seed(100)  # Seeding randomness for reproducibility

"""SIMULATION PARAMETERS"""
plot = False
output = True
dt = 0.00005  # Sampling interval of the simulation
T = 15
nt = round(T/dt)
store = 10

"""NETWORK PARAMETERS"""
N = 5000  # Number of neurons
tref = 0.002  # Refractory time constant in seconds
tm = 0.01  # Membrane time constant
vreset = -65  # Voltage reset
vpeak = -40  # Voltage peak
td = 0.02  # Synaptic decay time constant
tr = 0.002  # Synaptic rise time constant
p = 0.1  # Set the network sparsity
BIAS = vpeak  # Set the BIAS current, can help decrease/increase firing rates.
G = 0.04  # Factor of fixed weight matrix

"""RLMS PARAMETERS"""
m = 4
alpha = dt*0.01  # Sets the rate of weight change
Pinv = np.eye(N)*alpha  # Initialize the correlation weight matrix for RLMS
RECB = np.zeros((nt//store, 5))  # Storage matrix for some synaptic weights 
# The initial weight matrix with fixed random weights
OMEGA = G * (randn(N, N)) * (rand(N, N) < p) / (np.sqrt(N) * p)
BPhi = np.zeros((N, m))  # The initial matrix that will be learned by FORCE method
# Set the row average weight to be zero, explicitly
for i in np.arange(0, N, 1): 
    QS = np.argwhere(abs(OMEGA[i, :])>0)
    OMEGA[i, QS] = OMEGA[i, QS] - sum(OMEGA[i, QS])/len(QS)

rlms_start = round(5/dt)  # When to start RLMS
rlms_stop = round(10/dt)  # When to stop RLMS
step = 50  # Interval of RLMS in indices
Q = 10
E = (2*rand(N, m)-1)*Q

"""TARGET DYNAMICS - HODGKIN-HUXLEY"""
# Average potassium channel conductance per unit area (mS/cm^2)
gK = 35
# Average sodium channel conductance per unit area (mS/cm^2)
gNa = 40
# Average leak channel conductance per unit area (mS/cm^2)
gL = 0.3
# Membrane capacitance per unit area (uF/cm^2)
Cm = 1.0
# Potassium potential (mV)
VK = -77
# Sodium potential (mV)
VNa = 55
# Leak potential (mV)
Vl = -65

# Potassium ion-channel rate functions
def alpha_n(Vm):
    return (0.02 * (Vm - 25)) / (1 - np.exp((-1 * (Vm - 25)) / 9))

def beta_n(Vm):
    return (-0.002 * (Vm - 25)) / (1 - np.exp((Vm - 25) / 9))

# Sodium ion-channel rate functions
def alpha_m(Vm):
    return (0.182*(Vm + 35)) / (1 - np.exp((-1 * (Vm + 35)) / 9))

def beta_m(Vm):
    return (-0.124 * (Vm + 35)) / (1 - np.exp((Vm + 35) / 9))

def alpha_h(Vm):
    return 0.25 * np.exp((-1 * (Vm + 90)) / 12)

def beta_h(Vm):
    return (0.25 * np.exp((Vm + 62) / 6)) / np.exp((Vm + 90) / 12)
  
# n, m, and h steady-states
def n_inf(Vm=0.0):
    return alpha_n(Vm) / (alpha_n(Vm) + beta_n(Vm))

def m_inf(Vm=0.0):
    return alpha_m(Vm) / (alpha_m(Vm) + beta_m(Vm))

def h_inf(Vm=0.0):
    return alpha_h(Vm) / (alpha_h(Vm) + beta_h(Vm))


# Compute derivatives
def compute_derivatives(t0, y):

    dy = np.zeros((4,))

    Vm = y[0]
    m = y[1]
    n = y[2]
    h = y[3]
    
    # Input stimulus
    Ihh = 1.5
    
    # dVm/dt
    GK = (gK / Cm) * np.power(n, 4.0)
    GNa = (gNa / Cm) * np.power(m, 3.0) * h
    GL = gL / Cm
    
    dy[0] = (Ihh / Cm) - (GK * (Vm - VK)) - (GNa * (Vm - VNa)) - (GL * (Vm - Vl))
    
    # dm/dt
    dy[1] = (alpha_m(Vm) * (1.0 - m)) - (beta_m(Vm) * m)
        
    # dn/dt
    dy[2] = (alpha_n(Vm) * (1.0 - n)) - (beta_n(Vm) * n)
    
    # dh/dt
    dy[3] = (alpha_h(Vm) * (1.0 - h)) - (beta_h(Vm) * h)

    return dy
  
# Setup initial conditions
V0 = -65
Y = np.array([V0, n_inf(V0), m_inf(V0), h_inf(V0)])

# Solve ODE system
result = solve_ivp(compute_derivatives, (0,T*100), Y, t_eval=np.arange(0,nt)*dt*100, method='RK45')

xz = result.y.T
scaler = StandardScaler()
xz = scaler.fit_transform(xz)

"""PREINITIALIZE STORAGE"""
IPSC = np.zeros(N)  # Post synaptic current storage variable
hm = np.zeros(N)  # Storage variable for filtered firing rates
r = np.zeros(N)  # Second storage variable for filtered rates
hr = np.zeros(N)  # Third variable for filtered rates
JD = 0*IPSC  # Storage variable required for each spike time
tspike = np.zeros((8*nt+1, 2))  # Storage variable for spike times
ns = 0  # Number of spikes, counts during simulation
z = np.zeros(m)  # Initialize the approximant
tlast = np.zeros((N))  # This vector is used to set  the refractory times
v = vreset + rand(N)*(30-vreset)  # Initialize neuron voltage
v_ = v.copy()
REC2 = np.zeros((nt//store, 20))
REC = np.zeros((nt, 10))
current = np.zeros((nt//store, m))  # Storage variable for output current/approximant
store_time = np.zeros(round(nt/store));
j = 0

"""START INTEGRATION LOOP"""
for i in np.arange(0, nt, 1):
    print(i)
    inp = IPSC + E @ z + BIAS  # Total input current

    # Voltage equation with refractory period
    dv = (dt * i > tlast + tref) * (-v + inp) / tm
    v = v_ + dt*dv

    index = np.argwhere(v >= vpeak)[:, 0]  # Find the neurons that have spiked

    # Store spike times, and get the weight matrix column sum of spikers
    if len(index) > 0:
        # Compute the increase in current due to spiking
        JD = OMEGA[:, index].sum(axis=1)
        tspike[ns:ns+len(index), :] = np.array([index, 0*index+dt*i]).T
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

    v = v + (vreset - v) * (v >= vpeak)
    v_ = v
    # Implement RLMS with the FORCE method
    z = BPhi.T @ r  # Approximant
    err = z - xz[i]  # Error
    # Only adjust at an interval given by step
    if np.mod(i, step) == 1:
        # Only adjust between rlmst_start and rlms_stop
        if (i > rlms_start) and i < rlms_stop:
            cd = np.matmul(Pinv, r)
            BPhi = BPhi - (cd[:, None] @ err[None, :])
            Pinv = Pinv - (cd[:, None] @ cd[None, :]) / (1 + (r @ cd))

    v = v + (30 - v) * (v >= vpeak)

    REC[i, :] = v[0:10]  # Record a random voltage

    if np.mod(i, store) == 0:
        store_time[j] = dt*i
        # Reset with spike time interpolant implemented
        current[j] = z
        RECB[j, :] = BPhi[0:5, 1]
        REC2[j, :] = r[0:20]
        j += 1

"""WRANGLE OUTPUT"""
if output:
    sim_time = np.arange(0, nt, 1) * dt
    f_current = interp1d(store_time, current, axis=0, fill_value='extrapolate')
    f_RECB = interp1d(store_time, RECB, axis=0, fill_value='extrapolate')
    f_REC2 = interp1d(store_time, REC2, axis=0, fill_value='extrapolate')
    current_interp = f_current(sim_time)
    RECB_interp = f_RECB(sim_time)
    REC2_interp = f_REC2(sim_time)
    tspike = tspike[:ns,:]
    np.savez("lif_force_hh_output.npz",
             REC = REC,
             sim_time = sim_time,
             current = current_interp,
             RECB = RECB_interp,
             REC2 = REC2_interp,
             tspike = tspike,
             target=xz)


"""PLOTTING"""
if plot:
    fig, ax = plt.subplots(2)
    ax[0].plot(current)
    ax[0].plot(xz)
    ax[1].plot(REC[:, 0])
