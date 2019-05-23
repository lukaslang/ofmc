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
# This file computes two-channel optical flow with a source for the second
# channel.
import glob
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
import ofmc.external.tifffile as tiff
from matplotlib import cm
from ofmc.model.of import of2dmcs
from scipy import ndimage

# Set path with data.
# datapath = ('/home/ll542/store/'
#            'Dropbox (Cambridge University)/Drosophila/Data from Guy/3LU')
datapath = ('/Users/lukaslang/'
            'Dropbox (Cambridge University)/Drosophila/Data from Guy/')

# Set path where results are saved.
resultpath = 'results/of2dmcs'

# Set regularisation parameter.
alpha0 = 1e-2
alpha1 = 1e-3
beta0 = 1e-3
beta1 = 1e-5


def loadimage(filename: str) -> np.array:
    # Read image.
    img = tiff.imread(filename)

    # Crop images.
    # img = img[0:30, 80:-100, 80:-100]
    img = img[0:3, 100:200, 100:200]

    # Filter each frame.
    for k in range(img.shape[0]):
        img[k] = ndimage.gaussian_filter(img[k], sigma=1)

    # Normalise to [0, 1].
    img = np.array(img, dtype=float)
    img = (img - img.min()) / (img.max() - img.min())
    return img


def saveimage(path: str, name: str, img: np.array):
    if not os.path.exists(path):
        os.makedirs(path)

    # Plot image.
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, cmap=cm.gray)
    ax.set_title('Image')

    # Save figure.
    fig.savefig(os.path.join(path, name))
    plt.close(fig)


def savesource(path: str, name: str, img: np.array):
    if not os.path.exists(path):
        os.makedirs(path)

    # Plot image.
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.imshow(img, cmap=cm.coolwarm)
    ax.set_title('Source')
    fig.colorbar(cax, orientation='vertical')

    # Save figure.
    fig.savefig(os.path.join(path, name))
    plt.close(fig)


def savevelocity(path: str, name: str, img: np.array, vel: np.array):
    if not os.path.exists(path):
        os.makedirs(path)
    # Set spacing for vectors.
    s = 2

    # Define grid and velocities.
    m, n = img.shape
    Y, X = np.mgrid[0:m:1, 0:n:1]
    V, W = vel[:, :, 0], vel[:, :, 1]
    col = np.sqrt(V**2 + W**2)

    # Plot image and velocities.
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plt.imshow(img, cmap=cm.gray)
    ax.quiver(X[::s, ::s], Y[::s, ::s],
              W[::s, ::s], -V[::s, ::s], col[::s, ::s], cmap=cm.coolwarm)
    ax.set_title('Velocities')

    # Save figure.
    fig.savefig(os.path.join(path, name))
    plt.close(fig)


def savestreamline(path: str, name: str, img: np.array, vel: np.array):
    if not os.path.exists(path):
        os.makedirs(path)

    m, n = img.shape
    Y, X = np.mgrid[0:m:1, 0:n:1]
    V, W = vel[:, :, 0], vel[:, :, 1]
    col = np.sqrt(V**2 + W**2)

    # Plot image and velocities.
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plt.imshow(img, cmap=cm.gray)
    ax.streamplot(X, Y, W, V, density=2, color=col,
                  linewidth=1, cmap=cm.coolwarm)
    ax.set_title('Streamlines')

    # Save figure.
    fig.savefig(os.path.join(path, name))
    plt.close(fig)


print('Processing {0}'.format(datapath))

# Get folders with datasets.
datasets = [d for d in os.listdir(datapath)
            if os.path.isdir(os.path.join(datapath, d))]

# Run through datasets.
for dat in datasets:
    datfolder = os.path.join(datapath, dat)
    print("Dataset {0}".format(dat))

    # Identify Kymograph and do sanity check.
    cadherinfile = glob.glob('{0}/CadherinFrames.tif'.format(datfolder, dat))
    myosinfile = glob.glob('{0}/MyosinFrames.tif'.format(datfolder, dat))
    if len(cadherinfile) != 1 or len(myosinfile) != 1:
        warnings.warn("At least one of the channels is missing!")

    print("Computing velocities for file '{0}'".format(dat))

    # Load and preprocess Kymograph.
    imgc = loadimage(cadherinfile[0])
    imgm = loadimage(myosinfile[0])

    # Compute velocities and source.
    vel, source = of2dmcs(imgc, imgm, alpha0, alpha1, beta0, beta1)

    # Plot and save images.
    for k in range(imgc.shape[0]):
        saveimage(os.path.join(resultpath, dat),
                  'cadherin-{0:03d}.png'.format(k), imgc[k])
        saveimage(os.path.join(resultpath, dat),
                  'myosin-{0:03d}.png'.format(k), imgm[k])

    # Plot and save velocities.
    for k in range(imgc.shape[0]-1):
        savevelocity(os.path.join(resultpath, dat),
                     'cadherin-velocity-{0:03d}.png'.format(k),
                     imgc[k], vel[k])
        savevelocity(os.path.join(resultpath, dat),
                     'myosin-velocity-{0:03d}.png'.format(k), imgm[k], vel[k])
        savestreamline(os.path.join(resultpath, dat),
                       'cadherin-streamlines-{0:03d}.png'.format(k),
                       imgc[k], vel[k])
        savestreamline(os.path.join(resultpath, dat),
                       'myosin-streamlines-{0:03d}.png'.format(k),
                       imgm[k], vel[k])
        savesource(os.path.join(resultpath, dat),
                   'source-{0:03d}.png'.format(k), source[k])
