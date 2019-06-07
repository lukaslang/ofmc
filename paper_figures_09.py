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
# k and fits a linear model for the MATLAB created concentration file.
import datetime
import imageio
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats as stats
import ofmc.util.pyplothelpers as ph
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ofmc.model.cmscr import cmscr1d_img

ffc_logger = logging.getLogger('FFC')
ffc_logger.setLevel(logging.WARNING)
ufl_logger = logging.getLogger('UFL')
ufl_logger.setLevel(logging.WARNING)

# Set output quality.
dpi = 100

# Set path with data.
datapath = ('data')

# Set path where results are saved.
resultpath = 'results/{0}'.format(
        datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
if not os.path.exists(resultpath):
    os.makedirs(resultpath)

print('Processing {0}'.format(datapath))


def prepareimage(img: np.array, idx: int) -> np.array:
    # Remove first frame.
    img = img[idx:, :]

    # Filter image.
    # img = ndimage.gaussian_filter(img, sigma=1.0)

    # Normalise to [0, 1].
    img = np.array(img, dtype=float)
    img = (img - img.min()) / (img.max() - img.min())
    return img


# Select dataset.
name = 'myo_int'
# name = 'myo_int_jocelyn'

# Load kymograph.
file = os.path.join(datapath, '{0}.png'.format(name))
img = imageio.imread(file)

# Set starting index.
idx = 1

# Prepare image.
imgp = prepareimage(img, idx)

# Estimate derivatives for plotting purpose.
dtf = np.diff(imgp, n=1, axis=0)
dxf = np.diff(imgp, n=1, axis=1)

# Plot image.
fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(dtf, cmap=cm.viridis)

# Create colourbar.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
fig.colorbar(im, cax=cax, orientation='vertical')

# Save figure.
fig.savefig(os.path.join(resultpath, '{0}-dt.png'.format(name)),
            dpi=dpi, bbox_inches='tight')
plt.close(fig)

# Plot image.
fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(dxf, cmap=cm.viridis)

# Create colourbar.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
fig.colorbar(im, cax=cax, orientation='vertical')

# Save figure.
fig.savefig(os.path.join(resultpath, '{0}-dx.png'.format(name)),
            dpi=dpi, bbox_inches='tight')
plt.close(fig)

# Set regularisation parameters for cmscr1d.
alpha0 = 1e-2
alpha1 = 1e-3
alpha2 = 1e-4
alpha3 = 1e-4
beta = 2.5e-3

# Compute velocity and source.
vel, k, res, fun, converged = cmscr1d_img(imgp, alpha0, alpha1,
                                          alpha2, alpha3,
                                          beta, 'mesh')

# Plot and save figures.
path = os.path.join(*[resultpath, 'cmscr1d'])
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
