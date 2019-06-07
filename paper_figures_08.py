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
# This script estimates a source and plots concentration versus
# k and fits a linear model.
import datetime
import glob
import imageio
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import scipy.stats as stats
import warnings
import ofmc.util.pyplothelpers as ph
from scipy import ndimage
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

# Use only small seciton.
img = img[:, 40:125]

# Prepare image.
imgp = prepareimage(img)

# Set regularisation parameters for cmscr1d.
alpha0 = 5e-3
alpha1 = 1e-3
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

# Perform linear regression on k = k_on + k_off * imgp.
slope, intercept, r_value, p_value, std_err = stats.linregress(imgp.flatten(),
                                                               k.flatten())

print(('Linear regression: k_on={0:.3f}, ' +
      'k_off={1:.3f}').format(intercept, -slope))

# Set font style.
font = {'family': 'sans-serif',
        'serif': ['DejaVu Sans'],
        'weight': 'normal',
        'size': 20}
plt.rc('font', **font)
plt.rc('text', usetex=True)

# Set output quality.
dpi = 100

fig, ax = plt.subplots(figsize=(10, 5))
plt.scatter(imgp.flatten(), k.flatten(), s=1)
plt.plot(np.linspace(0, 1, 10), slope * np.linspace(0, 1, 10) + intercept,
         color='red')
# ax.set_title('c vs k')
# plt.xlabel('c')
# plt.ylabel('k')
fig.tight_layout()
plt.show()
# Save figure.
fig.savefig(os.path.join(path, '{0}-regress.png'.format(name)),
            dpi=dpi, bbox_inches='tight')
plt.close(fig)
