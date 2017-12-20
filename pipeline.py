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
import glob
import warnings
import numpy as np
from scipy import misc
from scipy import ndimage
from matplotlib import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
from ofmc.model.cm import cm1d

# Set path with data.
datapath = ('/Users/lukaslang/'
            'Dropbox (Cambridge University)/Drosophila/Data from Elena')

# Set path where results are saved.
resultpath = 'results'

# Set regularisation parameter.
alpha = 1e-2
beta = 1e-3


def loadimage(filename: str) -> np.array:
    # Read image.
    img = misc.imread(filename)

    # Remove cut.
    img = np.vstack((img[0:4, :], img[6:, :]))

    # Filter image.
    img = ndimage.gaussian_filter(img, sigma=1)

    # Normalise to [0, 1].
    img = np.array(img, dtype=float)
    img = (img - img.min()) / (img.max() - img.min())
    return img


def saveresults(path: str, name: str, vel: np.array):
    if not os.path.exists(path):
        os.makedirs(path)

    # Plot image.
    # plt.figure(figsize=(10, 10))
    # plt.imshow(img, cmap=cm.gray)

    maxvel = abs(vel).max()
    normi = mpl.colors.Normalize(vmin=-maxvel, vmax=maxvel)

    # Plot velocity.
    fig, ax = plt.subplots(figsize=(10, 5))
    cax = ax.imshow(vel, interpolation='nearest', norm=normi, cmap=cm.coolwarm)
    ax.set_title('Velocity')
    fig.colorbar(cax, orientation='horizontal')

    # Save figure.
    fig.savefig(os.path.join(path, '{0}-vel.png'.format(name)))
    plt.close(fig)

    m, n = vel.shape
    hx, hy = 1./(m-1), 1./(n-1)

    # Create grid for streamlines.
    Y, X = np.mgrid[0:m:1, 0:n:1]
    V = np.ones_like(X)*hy

    # Plot streamlines.
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.imshow(img, cmap=cm.gray)
    strm = ax.streamplot(X, Y, vel*hx, V, density=2,
                         color=vel, linewidth=1, norm=normi, cmap=cm.coolwarm)
    fig.colorbar(strm.lines, orientation='horizontal')

    fig.savefig(os.path.join(path, '{0}-streamlines.png'.format(name)))
    plt.close(fig)


print('Processing {0}'.format(datapath))

# Get folders with genotypes.
genotypes = [d for d in os.listdir(datapath)
             if os.path.isdir(os.path.join(datapath, d))]

# Run through genotypes.
for gen in genotypes:
    # Get folders with datasets.
    datasets = [d for d in os.listdir(os.path.join(datapath, gen))
                if os.path.isdir(os.path.join(datapath, os.path.join(gen, d)))]
    # Run through datasets.
    for dat in datasets:
        datfolder = os.path.join(datapath, os.path.join(gen, dat))
        print("Dataset {0}/{1}".format(gen, dat))

        # Identify Kymograph and do sanity check.
        kymos = glob.glob('{0}/SUM_Reslice of {1}*.tif'.format(datfolder, dat))
        if len(kymos) != 1:
            warnings.warn("No Kymograph found!")

        name = os.path.splitext(os.path.basename(kymos[0]))[0]
        print("Computing velocities for file '{0}'".format(name))

        # Load and preprocess Kymograph.
        img = loadimage(kymos[0])

        # Compute velocities.
        # vel = of1d(img, alpha, beta)
        vel = cm1d(img, alpha, beta)

        # Plot and save figures.
        saveresults(os.path.join(os.path.join(resultpath, gen), dat),
                    name, vel)
