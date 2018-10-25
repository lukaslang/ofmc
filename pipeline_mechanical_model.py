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
import os
import datetime
import numpy as np
from ofmc.model.cmscr import cmscr1d_img
import ofmc.util.pyplothelpers as ph
import sys
sys.path.append('../cuts-octave')
from timestepping import timestepping

# Set path where results are saved.
resultpath = 'results/{0}'.format(
        datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
if not os.path.exists(resultpath):
    os.makedirs(resultpath)

# Set artificial velocity.
artvel = False

# Run time stepping algorithm.
N, ii, TimeS, ca_sav, cd_sav, a_sav, v_sav = timestepping(artvel)
print('Done!\n')

# Plot results.
rng = range(ii)

# Set name.
name = 'mechanical_model_artvel_{0}'.format(str(artvel).lower())

resfolder = os.path.join(resultpath, name)
if not os.path.exists(resfolder):
    os.makedirs(resfolder)

# Plot and save figures.
ph.saveimage(resfolder, '{0}-a_sav'.format(name),
             a_sav[:, rng].transpose(), 'a\\_sav')
ph.saveimage(resfolder, '{0}-ca_sav'.format(name),
             ca_sav[:, rng].transpose(), 'ca\\_sav')
ph.saveimage(resfolder, '{0}-cd_sav'.format(name),
             cd_sav[:, rng].transpose(), 'cd\\_sav')
ph.saveimage(resfolder, '{0}-ca_sav+a_sav'.format(name),
             (cd_sav[:, rng] + a_sav[:, rng]).transpose(), 'ca\\_sav + a\\_sav')
ph.saveimage(resfolder, '{0}-v_sav'.format(name),
             v_sav[:, rng].transpose(), 'v\\_sav')

# Set regularisation parameter.
alpha0 = 5e0
alpha1 = 1e-1
alpha2 = 1e-2
alpha3 = 1e-2
beta = 5e-4

# Set start time for analysis (eg. 11 or 12)
start = 11

# Define concentration.
# img = (ca_sav[:, start:ii] + a_sav[:, start:ii]).transpose()
img = ca_sav[:, start:ii].transpose()
# img = ca_sav[:, 11:ii].transpose()

# Compute velocity and source.
vel, k, res, fun, converged = cmscr1d_img(img, alpha0, alpha1, alpha2, alpha3,
                                          beta, 'mesh')

# Plot and save figures.
ph.saveimage(resfolder, name, img)
ph.savevelocity(resfolder, name, img, vel)
ph.savesource(resfolder, name, k)
ph.savestrainrate(resfolder, name, img, vel)

# TODO: Fix scaling issue.
# Compute and output errors.
err_v = np.abs(vel - v_sav[0:-1, start:ii].transpose())
ph.saveimage(resfolder, '{0}-error_v'.format(name),
             err_v, 'Absolute difference in v.')
# TODO: Compute error in source term.

# Set artificial velocity.
artvel = True

# Run time stepping algorithm.
N, ii, TimeS, ca_sav, cd_sav, a_sav, v_sav = timestepping(artvel)
print('Done!\n')

# Plot results.
rng = range(ii)

# Set name.
name = 'mechanical_model_artvel_{0}'.format(str(artvel).lower())

resfolder = os.path.join(resultpath, name)
if not os.path.exists(resfolder):
    os.makedirs(resfolder)

# Plot and save figures.
ph.saveimage(resfolder, '{0}-a_sav'.format(name),
             a_sav[:, rng].transpose(), 'a\\_sav')
ph.saveimage(resfolder, '{0}-ca_sav'.format(name),
             ca_sav[:, rng].transpose(), 'ca\\_sav')
ph.saveimage(resfolder, '{0}-cd_sav'.format(name),
             cd_sav[:, rng].transpose(), 'cd\\_sav')
ph.saveimage(resfolder, '{0}-ca_sav+a_sav'.format(name),
             (cd_sav[:, rng] + a_sav[:, rng]).transpose(), 'ca\\_sav + a\\_sav')
ph.saveimage(resfolder, '{0}-v_sav'.format(name),
             v_sav[:, rng].transpose(), 'v\\_sav')

# Set regularisation parameter.
alpha0 = 5e0
alpha1 = 1e-1
alpha2 = 1e-2
alpha3 = 1e-2
beta = 5e-4

# Set start time for analysis (eg. 11 or 12)
start = 1

# Define concentration.
# img = (ca_sav[:, start:ii] + a_sav[:, start:ii]).transpose()
img = ca_sav[:, start:ii].transpose()
# img = ca_sav[:, 11:ii].transpose()

# Compute velocity and source.
vel, k, res, fun, converged = cmscr1d_img(img, alpha0, alpha1, alpha2, alpha3,
                                          beta, 'mesh')

# Plot and save figures.
ph.saveimage(resfolder, name, img)
ph.savevelocity(resfolder, name, img, vel)
ph.savesource(resfolder, name, k)
ph.savestrainrate(resfolder, name, img, vel)

# TODO: Fix scaling issue.
# Compute and output errors.
err_v = np.abs(vel - v_sav[0:-1, start:ii].transpose())
ph.saveimage(resfolder, '{0}-error_v'.format(name),
             err_v, 'Absolute difference in v.')
# TODO: Compute error in source term.
