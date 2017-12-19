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
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy import misc
from scipy import ndimage
import ofmc.model.of as of
import os

# Set regularisation parameter.
alpha = 1e-2
beta = 1e-3

# Load image.
name = ['E2PSB1PMT-560-C1', 'E5PSB1PMT-1', 'E8-PSB1PMT', 'PSB1_E1PMT', 'PSB4PMT', 'PSB5PMT', 'epidermalejonction', 'artificialcut_small']
ending = ['png', 'tif', 'tif', 'tif', 'tif', 'tif', 'png', 'png']

# Select file.
idx = 0
name = name[idx]

# Read image.
img = misc.imread('data/{0}.{1}'.format(name, ending[idx]))

# Remove cut.
img = np.vstack((img[0:4, :], img[6:, :]))

# Filter image.
img = ndimage.gaussian_filter(img, sigma=1)

# Normalise to [0, 1].
img = np.array(img, dtype=float)
img = (img - img.min()) / (img.max() - img.min())

# Compute optical flow.
vel = of.of1d(img, alpha, beta)

# Plot image.
plt.figure(figsize=(10, 10))
plt.imshow(img, cmap=cm.gray)

# Plot velocity.
fig, ax = plt.subplots()
cax = ax.imshow(vel, interpolation='nearest', cmap=cm.coolwarm)
ax.set_title('Velocity')

maxvel = abs(vel).max()
# Add colorbar, make sure to specify tick locations to match desired ticklabels
cbar = fig.colorbar(cax, orientation='horizontal')

if not os.path.exists('results/of'):
    os.makedirs('results/of')    
fig.savefig('results/of/{0}-vel.png'.format(name))

m, n = img.shape
hx, hy = 1./(m-1), 1./(n-1)

# Create grid for streamlines.
Y, X = np.mgrid[0:m:1, 0:n:1]
V = np.ones_like(X)*hy

# Plot streamlines.
fig, ax = plt.subplots(figsize=(10, 10))
plt.imshow(img, cmap=cm.gray)
strm = ax.streamplot(X, Y, vel*hx, V, density=3, color=vel, linewidth=1, cmap=cm.coolwarm)
fig.colorbar(strm.lines, orientation='horizontal')
plt.show()
fig.savefig('results/of/{0}-img.png'.format(name))