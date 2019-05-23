#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import datetime
import math
import numpy as np
import ofmc.mechanics.solver as solver
import ofmc.util.velocity as velocity
import scipy.stats as stats
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

# Set path where results are saved.
resultpath = 'results/{0}'.format(
        datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
if not os.path.exists(resultpath):
    os.makedirs(resultpath)

# Create model and solver parameters.
mp = solver.ModelParams()
mp.t_cut = 0
mp.k_on = 0
mp.k_off = 0

sp = solver.SolverParams()
sp.n = 300
sp.m = 266
sp.T = 0.025
sp.dt = 1e-6


# Define initial values.
def ca_init(x: float) -> float:
    return stats.uniform.pdf(x, 0, 1) * 20 \
        - math.sin(40 * x + math.cos(40 * x)) / 5


def rho_init(x: float) -> float:
    return stats.uniform.pdf(x, 0, 1) \
        + (1 + math.sin(40 * x + math.cos(40 * x))) / 10


# Initialise tracers.
x = np.array(np.linspace(0, 1, num=25))

# Define parameters of artificial velocity field.
c0 = 0.05
v0 = 5
tau0 = 0.05
tau1 = 0.1


def vel(t: float, x: float) -> float:
    return velocity.vel(t, x - 0.5, c0, v0, tau0, tau1)


# Run solver.
rho, ca, v, sigma, x, idx = solver.solve(mp, sp, rho_init, ca_init, x, vel=vel)
#rho, ca, v, sigma, x, idx = solver.solve(mp, sp, rho_init, ca_init, x)



# Check pointwise error of continuity equation.
# Compute grid spacing.
dx = 1.0 / sp.n
dt = sp.T / sp.m

flux = np.array([solver.flux(a, b, sp.delta) for a, b in zip(v, rho)])
dtrho = np.diff(rho, n=1, axis=0) / dt
err = np.abs(dtrho + np.diff(flux[:-1, :], n=1, axis=1) / dx)

fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(err, cmap=cm.viridis)
# plt.plot(x * sp.n, np.linspace(0, sp.m, sp.m + 1))
ax.set_title('Error rho')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
fig.colorbar(im, cax=cax, orientation='vertical')
fig.tight_layout()
plt.show()
plt.close(fig)

# Plot image.
fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(rho, cmap=cm.viridis)
# plt.plot(x * sp.n, np.linspace(0, sp.m, sp.m + 1))
ax.set_title('rho')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
fig.colorbar(im, cax=cax, orientation='vertical')
fig.tight_layout()
plt.show()
plt.close(fig)

# Plot image.
fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(ca, cmap=cm.viridis)
# plt.plot(x * sp.n, np.linspace(0, sp.m, sp.m + 1))
ax.set_title('ca')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
fig.colorbar(im, cax=cax, orientation='vertical')
fig.tight_layout()
plt.show()
plt.close(fig)

# Plot image.
fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(v, cmap=cm.viridis)
ax.set_title('v')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
fig.colorbar(im, cax=cax, orientation='vertical')
fig.tight_layout()
plt.show()
plt.close(fig)

# Plot image.
fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(sigma, cmap=cm.viridis)
ax.set_title('sigma')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
fig.colorbar(im, cax=cax, orientation='vertical')
fig.tight_layout()
plt.show()
plt.close(fig)

# source = mp.k_on - mp.k_on * ca * rho
source = mp.k_on - mp.k_on * ca

# Plot image.
fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(source, cmap=cm.viridis)
ax.set_title('source')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
fig.colorbar(im, cax=cax, orientation='vertical')
fig.tight_layout()
plt.show()
plt.close(fig)

sig = mp.eta * np.diff(v, n=1, axis=1) / dx + mp.chi * ca
err = np.abs(sigma[:, 1:-1] - sig)

fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(err, cmap=cm.viridis)
ax.set_title('Error sigma')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
fig.colorbar(im, cax=cax, orientation='vertical')
fig.tight_layout()
plt.show()
plt.close(fig)

err = np.abs(np.diff(sigma, n=1, axis=1) / dx - mp.xi * v)

fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(err, cmap=cm.viridis)
ax.set_title('Error force balance')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
fig.colorbar(im, cax=cax, orientation='vertical')
fig.tight_layout()
plt.show()
plt.close(fig)
