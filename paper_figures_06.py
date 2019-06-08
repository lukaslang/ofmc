#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2017 Lukas Lang
#
# This file is part of OFMC.
#
#    OFMC is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    OFMC is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with OFMC.  If not, see <http://www.gnu.org/licenses/>.
#
# Figure 12: computes results for data created by the mechanical model.
import os
import datetime
import numpy as np
from dolfin import Point
from dolfin import RectangleMesh
from ofmc.model.cmscr import cmscr1d_img
import ofmc.util.pyplothelpers as ph
import math
import ofmc.mechanics.solver as solver
import scipy.stats as stats
import matplotlib.pyplot as plt

# Set font style.
font = {'family': 'sans-serif',
        'serif': ['DejaVu Sans'],
        'weight': 'normal',
        'size': 30}
plt.rc('font', **font)
plt.rc('text', usetex=True)

# Set output quality.
dpi = 100

# Set path where results are saved.
resultpath = 'results/{0}'.format(
        datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
if not os.path.exists(resultpath):
    os.makedirs(resultpath)

# Create model and solver parameters.
mp = solver.ModelParams()
mp.eta = 1
mp.xi = 0.1
mp.chi = 1
mp.t_cut = 0
mp.k_on = 200
mp.k_off = 10

sp = solver.SolverParams()
sp.n = 300
sp.m = 300
sp.T = 0.1
sp.dt = 2.5e-6
sp.delta = 1e-6


# Define initial values.
def ca_init(x):
    return stats.uniform.pdf(x, 0, 1) * 20 \
        - math.sin(40 * x + math.cos(40 * x)) / 5


def rho_init(x):
    return stats.uniform.pdf(x, 0, 1) \
        + (1 + math.sin(40 * x + math.cos(40 * x))) / 10


# Initialise tracers.
x = np.array(np.linspace(0, 1, num=25))

# Run solver.
rho, ca, v, sigma, x, idx = solver.solve(mp, sp, rho_init, ca_init, x)

# Compute mean from staggered grid.
v = (v[:, 0:-1] + v[:, 1:]) / 2

# Compute source.
source = mp.k_on - mp.k_off * ca * rho

# Set name and create folder.
name = 'mechanical_model'
resfolder = os.path.join(resultpath, name)
if not os.path.exists(resfolder):
    os.makedirs(resfolder)

# Plot and save figures.
ph.saveimage(resfolder, '{0}-rho'.format(name), rho)
ph.saveimage(resfolder, '{0}-ca'.format(name), ca)
ph.saveimage(resfolder, '{0}-v'.format(name), v)
ph.saveimage(resfolder, '{0}-sigma'.format(name), sigma)
ph.saveimage(resfolder, '{0}-k'.format(name), source)
ph.savevelocity(resfolder, '{0}-v'.format(name), ca, v, T=sp.T)

# Plot and save concentration profile.
fig, ax = plt.subplots(figsize=(10, 5))
for t in np.arange(0, sp.m, 59):
    plt.plot(ca[t, :], label='t={:.2f}'.format(sp.T * t / (sp.m - 1)))
ax.legend(loc='best')
fig.tight_layout()
plt.show()
# Save figure.
fig.savefig(os.path.join(resfolder, '{0}-ca-profile.png'.format(name)),
            dpi=dpi, bbox_inches='tight')
plt.close(fig)

# Perform linear regression on k = k_on + k_off * img * rho.
slope, intercept, r_value, p_value, std_err = \
    stats.linregress(ca.flatten() * rho.flatten(), source.flatten())

print(('Linear regression: k_on={0:.3f}, ' +
      'k_off={1:.3f}').format(intercept, -slope))

fig, ax = plt.subplots(figsize=(10, 5))
plt.scatter(ca.flatten() * rho.flatten(), source.flatten(), s=1)
plt.plot(np.linspace(0, 30, 10), slope * np.linspace(0, 30, 10) + intercept,
         color='red')
# ax.set_title('c vs k')
# plt.xlabel('c')
# plt.ylabel('k')
fig.tight_layout()
plt.show()
# Save figure.
fig.savefig(os.path.join(resfolder, '{0}-regress.png'.format(name)),
            dpi=dpi, bbox_inches='tight')
plt.close(fig)

# Set regularisation parameters for cmscr1d.
alpha0 = 1e-4
alpha1 = 1e-4
alpha2 = 5e-5
alpha3 = 5e-5
beta = 1e-6

# Define concentration and normalise image.
offset = idx
img = ca[offset:, :]
m, n = img.shape

# Create mesh.
mesh = RectangleMesh(Point(0.0, 0.0), Point(sp.T, 1.0), m - 1, n - 1)

# Set boundary conditions for velocity.
bc = 'zero'

# Compute velocity and source.
vel, k, res, fun, converged = cmscr1d_img(img, alpha0, alpha1, alpha2, alpha3,
                                          beta, 'mesh', mesh, bc)

# Set name and create folder.
name = 'cmscr1d'
resfolder = os.path.join(resultpath, name)
if not os.path.exists(resfolder):
    os.makedirs(resfolder)

# Plot and save figures.
ph.saveimage(resfolder, '{0}-v'.format(name), vel)
ph.saveimage(resfolder, '{0}-k'.format(name), k)
ph.savevelocity(resfolder, '{0}-v'.format(name), img, vel, T=sp.T)

# Compute and output errors.
err_v = np.abs(vel - v[offset:, :])
ph.saveimage(resfolder, '{0}-error_v'.format(name), err_v)
err_k = np.abs(k - source[offset:, :])
ph.saveimage(resfolder, '{0}-error_k'.format(name), err_k)

# Perform linear regression on k = k_on + k_off * img * rho.
slope, intercept, r_value, p_value, std_err = \
    stats.linregress(img.flatten() * rho[offset:, :].flatten(), k.flatten())

print(('Linear regression: k_on={0:.3f}, ' +
      'k_off={1:.3f}').format(intercept, -slope))

max_ca_rho = np.max(img * rho[offset:, :])
min_ca_rho = np.min(img * rho[offset:, :])
fig, ax = plt.subplots(figsize=(10, 5))
plt.scatter(img.flatten() * rho[offset:, :].flatten(), k.flatten(), s=1)
plt.plot(np.linspace(min_ca_rho, max_ca_rho, 10),
         slope * np.linspace(min_ca_rho, max_ca_rho, 10) + intercept,
         color='red')
fig.tight_layout()
plt.show()
# Save figure.
fig.savefig(os.path.join(resfolder, '{0}-regress.png'.format(name)),
            dpi=dpi, bbox_inches='tight')
plt.close(fig)
