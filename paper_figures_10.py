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
# This script computes results for data created by the mechanical model.
import os
import datetime
import numpy as np
from ofmc.model.cmscr import cmscr1d_img
import ofmc.util.pyplothelpers as ph
import math
import ofmc.mechanics.solver as solver
import scipy.stats as stats
import matplotlib.pyplot as plt

# Set path where results are saved.
resultpath = 'results/{0}'.format(
        datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
if not os.path.exists(resultpath):
    os.makedirs(resultpath)

# Set artificial velocity.
artvel = False

# Create model and solver parameters.
mp = solver.ModelParams()
mp.t_cut = 0
mp.k_on = 0
mp.k_off = 0

sp = solver.SolverParams()
sp.n = 200
sp.m = 200
sp.T = 0.025
sp.dt = 1e-5


# Define initial values.
def ca_init(x):
    return stats.uniform.pdf(x, 0, 1) * 20 \
        - math.sin(40 * x + math.cos(40 * x)) / 2


def rho_init(x):
    return stats.uniform.pdf(x, 0, 1) \
        + (1 + math.sin(50 * x + math.cos(50 * x))) / 10


# Initialise tracers.
x = np.array(np.linspace(0, 1, num=25))

# Run solver.
rho, ca, v, sigma, x, idx = solver.solve(mp, sp, rho_init, ca_init, x)

plt.figure()
for t in np.arange(0, 200, 25):
    plt.plot(ca[t, :])
plt.show()
plt.close()

# Compute mean from staggered grid.
v = (v[:, 0:-1] + v[:, 1:]) / 2

# Scale velocities to time interval [0, 1].
v = v * sp.T

# Compute source.
source = mp.k_on - mp.k_off * ca

# Scale source to time interval [0, 1].
source = source * sp.T

# Set name and create folder.
name = 'mechanical_model_artvel_{0}_simulated'.format(str(artvel).lower())
resfolder = os.path.join(resultpath, name)
if not os.path.exists(resfolder):
    os.makedirs(resfolder)

# Plot and save figures.
ph.saveimage(resfolder, '{0}-ca'.format(name), ca)
ph.saveimage(resfolder, '{0}-v'.format(name), v)
ph.saveimage(resfolder, '{0}-k'.format(name), source)
ph.savevelocity(resfolder, '{0}-v'.format(name), ca, v)

# Set regularisation parameter.
alpha0 = 5e-2
alpha1 = 5e-3
alpha2 = 5e-3
alpha3 = 5e-3
beta = 1e-3

# Define concentration.
img = ca[idx + 1:, :]

# Compute velocity and source.
vel, k, res, fun, converged = cmscr1d_img(img, alpha0, alpha1, alpha2, alpha3,
                                          beta, 'mesh')

# Set name and create folder.
name = 'mechanical_model_artvel_{0}'.format(str(artvel).lower())
resfolder = os.path.join(resultpath, name)
if not os.path.exists(resfolder):
    os.makedirs(resfolder)

# Plot and save figures.
ph.saveimage(resfolder, name, img)
ph.savevelocity(resfolder, name, img, vel)
ph.savesource(resfolder, name, k)
ph.savestrainrate(resfolder, name, img, vel)

# Compute and output errors.
err_v = np.abs(vel - v[idx+1:, :])
ph.saveimage(resfolder, '{0}-error_v'.format(name), err_v)
err_k = np.abs(k - source[idx+1:, :])
ph.saveimage(resfolder, '{0}-error_k'.format(name), err_k)
