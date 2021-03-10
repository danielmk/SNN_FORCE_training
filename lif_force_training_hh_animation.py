# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 20:03:50 2021

@author: Daniel
"""

import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib

dt = 0.00005
data = np.load("lif_force_hh_output.npz")
sim_time = data['sim_time']
current = data['current']
RECB = data['RECB']
REC2 = data['REC2']
target = data['target']
tspike = data['tspike']
target = data['target']
REC = data['REC']

for idx in range(REC.shape[1]):
    sts = tspike[tspike[:,0] == idx, 1]
    sti = np.array(sts / dt, dtype=int)
    REC[sti, idx] = 20
    
tspike_subset = tspike[tspike[:,0] < 50,:]

font = {'family' : 'Arial',
        'size'   : 8}
matplotlib.rc('font', **font)
# Create the figure and axes to animate
fig, ax = plt.subplots(3, constrained_layout=True)
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
ax[0].get_xaxis().set_ticks([])
ax[1].get_xaxis().set_ticks([])

# init_func() is called at the beginning of the animation
def init_func():
    for a in ax:
        a.clear()
 
# update_plot() is called between frames
def update_plot(i):
    for a in ax:
        a.clear()
        a.set_xlim((0, 15))
    t = i * dt
    ax[0].set_ylabel("# Neuron (Subset)")
    ax[1].set_ylabel("Example Neurons")
    ax[2].set_ylabel("Target (Orange)\nOutput (Blue)")
    ax[2].set_xlabel("Time (s)")
    ax[1].set_ylim((-70, 230))
    ax[2].set_ylim((-1.5, 5))
    tspike_curr = tspike_subset[tspike_subset[:,1] < t,:]
    ax[0].scatter(tspike_curr[:,1], tspike_curr[:,0], color='k', s=0.5)
    ax[1].plot(sim_time[:i], REC[:i,0], linewidth=0.5)
    ax[1].plot(sim_time[:i], REC[:i,1]+100, linewidth=0.5)
    ax[1].plot(sim_time[:i], REC[:i,2]+200, linewidth=0.5)
    ax[2].plot(sim_time[:i], current[:i,0], linewidth=0.5)
    ax[2].plot(sim_time[:i], target[:i,0], linewidth=0.5)
 
# Create animation
anim = FuncAnimation(fig,
                     update_plot,
                     frames=np.arange(0, len(sim_time), 1000),
                     init_func=init_func)
 
# Save animation
anim.save('lif_force_hh_animation.mp4',
          dpi=150,
          fps=30,
          writer='ffmpeg')
