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
import datetime
import glob
import imageio
import logging
import numpy as np
import os
import re
import warnings
import ofmc.util.pyplothelpers as ph
from scipy import ndimage
from ofmc.model.of import of1d_img
from ofmc.model.cms import cms1dl2_img
from ofmc.model.cms import cms1d_img
from ofmc.model.cmscr import cmscr1d_img

ffc_logger = logging.getLogger('FFC')
ffc_logger.setLevel(logging.WARNING)
ufl_logger = logging.getLogger('UFL')
ufl_logger.setLevel(logging.WARNING)

# Set path with data.
datapath = ('/Users/lukaslang/'
            'Dropbox (Cambridge University)/Drosophila/Data from Elena')

# Set path where results are saved.
resultpath = 'results/{0}'.format(
        datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
if not os.path.exists(resultpath):
    os.makedirs(resultpath)

print('Processing {0}'.format(datapath))


def load_kymo(datfolder, dat):
    # Identify Kymograph and do sanity check.
    kymos = glob.glob('{0}/SUM_Reslice of {1}*.tif'.format(datfolder, dat))
    if len(kymos) != 1:
        warnings.warn("No Kymograph found!")

    # Extract name of kymograph and replace whitespaces.
    name = os.path.splitext(os.path.basename(kymos[0]))[0]
    name = re.sub(' ', '_', name)

    # Load kymograph.
    return imageio.imread(kymos[0]), name


def prepareimage(img: np.array) -> np.array:
    # Remove cut.
    img = np.vstack((img[0:5, :], img[4, :], img[6:, :]))

    # Filter image.
    img = ndimage.gaussian_filter(img, sigma=1.0)

    # Normalise to [0, 1].
    img = np.array(img, dtype=float)
    img = (img - img.min()) / (img.max() - img.min())
    return img


# Select dataset.
gen = 'SqAX3_SqhGFP42_GAP43_TM6B'
# dat = '190216E8PSB1'
dat = '190216E5PSB2'

# Load kymograph.
datfolder = os.path.join(datapath, os.path.join(gen, dat))
img, name = load_kymo(datfolder, dat)

# Prepare image.
imgp = prepareimage(img)

# Figure 5: different models.

# Set regularisation parameters for of1d.
alpha0 = 5e-3
alpha1 = 5e-3

# Compute velocity.
vel, res, fun = of1d_img(imgp, alpha0, alpha1, 'mesh')

# Plot and save figures.
path = os.path.join(*[resultpath, 'of1d', gen, dat])
if not os.path.exists(resultpath):
    os.makedirs(resultpath)

ph.saveimage(path, name, imgp)
ph.savevelocity(path, name, img, vel)

# Set regularisation parameters for cms1dl2.
alpha0 = 5e-3
alpha1 = 5e-3
gamma = 1e-1

# Compute velocity and source.
vel, k, res, fun = cms1dl2_img(imgp, alpha0, alpha1, gamma, 'mesh')

# Plot and save figures.
path = os.path.join(*[resultpath, 'cms1dl2', gen, dat])
if not os.path.exists(resultpath):
    os.makedirs(resultpath)

ph.saveimage(path, name, imgp)
ph.savevelocity(path, name, img, vel)
ph.savesource(path, name, k)

# Set regularisation parameters for cms1d.
alpha0 = 5e-3
alpha1 = 5e-3
alpha2 = 1e-4
alpha3 = 1e-4

# Compute velocity and source.
vel, k, res, fun = cms1d_img(imgp, alpha0, alpha1, alpha2, alpha3, 'mesh')

# Plot and save figures.
path = os.path.join(*[resultpath, 'cms1d', gen, dat])
if not os.path.exists(resultpath):
    os.makedirs(resultpath)

ph.saveimage(path, name, imgp)
ph.savevelocity(path, name, img, vel)
ph.savesource(path, name, k)

# Set regularisation parameters for cmscr1d.
alpha0 = 5e-3
alpha1 = 5e-3
alpha2 = 1e-4
alpha3 = 1e-4
beta = 2.5e-3

# Compute velocity and source.
vel, k, res, fun, converged = cmscr1d_img(imgp, alpha0, alpha1,
                                          alpha2, alpha3,
                                          beta, 'mesh')

# Plot and save figures.
path = os.path.join(*[resultpath, 'cmscr1d', gen, dat])
if not os.path.exists(resultpath):
    os.makedirs(resultpath)

ph.saveimage(path, name, imgp)
ph.savevelocity(path, name, img, vel)
ph.savesource(path, name, k)

# Figure 5: increasing parameter beta.
alpha0 = 5e-3
alpha1 = 5e-3
alpha2 = 1e-4
alpha3 = 1e-4
beta = 1e-4

# Compute velocity and source.
vel, k, res, fun, converged = cmscr1d_img(imgp, alpha0, alpha1,
                                          alpha2, alpha3,
                                          beta, 'mesh')

# Plot and save figures.
path = os.path.join(*[resultpath, 'cmscr1d', '0', gen, dat])
if not os.path.exists(resultpath):
    os.makedirs(resultpath)

ph.savevelocity(path, name, img, vel)
ph.savesource(path, name, k)

alpha0 = 5e-3
alpha1 = 5e-3
alpha2 = 1e-4
alpha3 = 1e-4
beta = 1e-3

# Compute velocity and source.
vel, k, res, fun, converged = cmscr1d_img(imgp, alpha0, alpha1,
                                          alpha2, alpha3,
                                          beta, 'mesh')

# Plot and save figures.
path = os.path.join(*[resultpath, 'cmscr1d', '1', gen, dat])
if not os.path.exists(resultpath):
    os.makedirs(resultpath)

ph.savevelocity(path, name, img, vel)
ph.savesource(path, name, k)

alpha0 = 5e-3
alpha1 = 5e-3
alpha2 = 1e-4
alpha3 = 1e-4
beta = 1e-2

# Compute velocity and source.
vel, k, res, fun, converged = cmscr1d_img(imgp, alpha0, alpha1,
                                          alpha2, alpha3,
                                          beta, 'mesh')

# Plot and save figures.
path = os.path.join(*[resultpath, 'cmscr1d', '2', gen, dat])
if not os.path.exists(resultpath):
    os.makedirs(resultpath)

ph.savevelocity(path, name, img, vel)
ph.savesource(path, name, k)
