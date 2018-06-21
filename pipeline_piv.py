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
import glob
import imageio
import re
import numpy as np
import warnings
import ofmc.external.tifffile as tiff
from scipy import ndimage
from scipy import interpolate
from read_roi import read_roi_zip
from matplotlib import cm
from matplotlib.collections import LineCollection
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import ofmc.util.roihelpers as rh
from openpiv import tools
from openpiv import process
from openpiv import scaling
from openpiv import validation
from openpiv import filters
from ofmc.model.of import of2dmcs

# Set path with data.
datapath = ('/Users/lukaslang/'
            'Dropbox (Cambridge University)/Drosophila/Data from Elena')

# Set path where results are saved.
resultpath = 'results/{0}'.format(
        datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
if not os.path.exists(resultpath):
    os.makedirs(resultpath)

# Set regularisation parameter.
alpha0 = 5e-3
alpha1 = 1e-3
alpha2 = 1e-4
alpha3 = 1e-4
beta = 5e-3

# Set font style.
font = {'weight': 'normal',
        'size': 20}
plt.rc('font', **font)

# Set colormap.
cmap = cm.viridis

# Streamlines.
density = 2
linewidth = 2


def loadimage(filename: str) -> np.array:
    return imageio.imread(filename)


def prepareimage(img: np.array) -> np.array:
    # Remove cut.
    img = np.vstack((img[0:5, :], img[4, :], img[6:, :]))

    # Filter image.
    img = ndimage.gaussian_filter(img, sigma=1)

    # Normalise to [0, 1].
    img = np.array(img, dtype=float)
    img = (img - img.min()) / (img.max() - img.min())
    return img


def saveimage(path: str, name: str, img: np.array):
    if not os.path.exists(path):
        os.makedirs(path)

    # Plot image.
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(img, cmap=cm.gray)
    ax.set_title('Fluorescence intensity')

    # Create colourbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # Save figure.
    fig.savefig(os.path.join(path, '{0}.png'.format(name)),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def savesource(path: str, name: str, img: np.array):
    if not os.path.exists(path):
        os.makedirs(path)

    # Plot image.
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(img, cmap=cmap)
    ax.set_title('Source')

    # Create colourbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # Save figure.
    fig.tight_layout()
    fig.savefig(os.path.join(path, '{0}-source.png'.format(name)),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def savevelocity(path: str, name: str, img: np.array, vel: np.array):
    if not os.path.exists(path):
        os.makedirs(path)

    maxvel = abs(vel).max()
    normi = mpl.colors.Normalize(vmin=-maxvel, vmax=maxvel)

    # Plot velocity.
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(vel, interpolation='nearest', norm=normi, cmap=cmap)
    ax.set_title('Velocity')

    # Create colourbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # Save figure.
    fig.tight_layout()
    fig.savefig(os.path.join(path, '{0}-velocity.png'.format(name)),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    m, n = vel.shape
    hx, hy = 1.0 / (m - 1), 1.0 / (n - 1)

    # Create grid for streamlines.
    Y, X = np.mgrid[0:m, 0:n]
    V = np.ones_like(X)

    # Plot streamlines.
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.imshow(img, cmap=cm.gray)
    ax.set_title('Streamlines')
    strm = ax.streamplot(X, Y, vel * hx / hy, V, density=density,
                         color=vel, linewidth=linewidth, norm=normi, cmap=cmap)
#    fig.colorbar(strm.lines, orientation='vertical')

    # Create colourbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(strm.lines, cax=cax, orientation='vertical')

    fig.tight_layout()
    fig.savefig(os.path.join(path, '{0}-streamlines.png'.format(name)),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Save velocity profile after cut.
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.plot(vel[5])
    ax.set_title('Velocity profile right after the cut')

    fig.tight_layout()
    fig.savefig(os.path.join(path, '{0}-profile.png'.format(name)),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def savestrainrate(path: str, name: str, img: np.array, vel: np.array):
    if not os.path.exists(path):
        os.makedirs(path)

    m, n = img.shape
    hy = 1.0 / (n - 1)
    sr = np.gradient(vel, hy, axis=1)

    maxsr = abs(sr).max()
    normi = mpl.colors.Normalize(vmin=-maxsr, vmax=maxsr)

    # Plot velocity.
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(sr, interpolation='nearest', norm=normi, cmap=cmap)
    ax.set_title('Strain rate')

    # Create colourbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # Save figure.
    fig.tight_layout()
    fig.savefig(os.path.join(path, '{0}-strainrate.png'.format(name)),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def saveroi(path: str, name: str, img: np.array, roi):

    # Plot image.
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(img, cmap=cm.gray)
    ax.set_title('Manual tracks')

    for v in roi:
        plt.plot(roi[v]['x'], roi[v]['y'], 'C3', lw=2)

    # Create colourbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # Save figure.
    fig.tight_layout()
    fig.savefig(os.path.join(path, '{0}-roi.png'.format(name)),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def savespl(path: str, name: str, img: np.array, roi, spl):

    m, n = img.shape

    # Determine min/max velocity.
    maxvel = -np.inf
    for v in roi:
        y = roi[v]['y']
        derivspl = spl[v].derivative()
        maxvel = max(maxvel, max(abs(derivspl(y) * m / n)))

    # Determine colour coding.
    normi = mpl.colors.Normalize(vmin=-maxvel, vmax=maxvel)

    # Plot image.
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.imshow(img, cmap=cm.gray)
    ax.set_title('Velocity of spline')

    # Plot splines.
    for v in roi:
        # Compute derivative of spline.
        derivspl = spl[v].derivative()

        y = roi[v]['y']
        y = np.arange(y[0], y[-1] + 0.5, 0.5)

        points = np.array([spl[v](y), y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap=cmap, norm=normi)
        lc.set_array(derivspl(y) * m / n)
        lc.set_linewidth(2)
        plt.gca().add_collection(lc)

    # Create colourbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=normi)

    # Save figure.
    fig.tight_layout()
    fig.savefig(os.path.join(path, '{0}-spline.png'.format(name)),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def saveerror(path: str, name: str, img: np.array, spl):

    # Interpolate velocity.
    m, n = vel.shape
    gridx, gridy = np.mgrid[0:m, 0:n]
    gridpoints = np.hstack([gridx.reshape(m * n, 1), gridy.reshape(m * n, 1)])

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.imshow(img, cmap=cm.gray)
    ax.set_title('Error along tacks')

    # Evaluate velocity for each spline.
    error = dict()
    maxerror = 0
    for v in roi:
        y = roi[v]['y']

        # Interpolate velocity.
        y = np.arange(y[0], y[-1] + 0.5, 0.5)
        x = np.array(spl[v](y))
        veval = interpolate.griddata(gridpoints, vel.flatten(), (y, x),
                                     method='cubic')

        # Compute derivative of spline.
        derivspl = spl[v].derivative()

        # Compute error in velocity.
        error[v] = abs(derivspl(y) * m / n - veval)

        # Update maximum error.
        maxerror = max(maxerror, max(abs(error[v])))

    # Determine colour coding.
    normi = mpl.colors.Normalize(vmin=0, vmax=maxerror)

    # Evaluate velocity for each spline.
    for v in roi:
        y = roi[v]['y']

        # Interpolate velocity.
        y = np.arange(y[0], y[-1] + 0.5, 0.5)
        x = np.array(spl[v](y))

        points = np.array([spl[v](y), y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap=cmap, norm=normi)

        lc.set_array(error[v])
        lc.set_linewidth(2)
        plt.gca().add_collection(lc)

    # Create colourbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=normi)

    # Save figure.
    fig.tight_layout()
    fig.savefig(os.path.join(path, '{0}-error.png'.format(name)),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_spl_streamlines_(path: str, name: str, img: np.array, vel: np.array,
                          spl):

    m, n = img.shape

    # Determine max. velocity of splines.
    maxvel = -np.inf
    for v in roi:
        y = roi[v]['y']
        derivspl = spl[v].derivative()
        maxvel = max(maxvel, max(abs(derivspl(y) * m / n)))

    # Determine max. of splines and computed velocities.
    maxvel = max(maxvel, abs(vel).max())
    normi = mpl.colors.Normalize(vmin=-maxvel, vmax=maxvel)

    m, n = vel.shape
    hx, hy = 1.0 / (m - 1), 1.0 / (n - 1)

    # Create grid for streamlines.
    Y, X = np.mgrid[0:m, 0:n]
    V = np.ones_like(X)

    # Plot streamlines.
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.imshow(img, cmap=cm.gray)

    # Plot splines and velocity.
    for v in roi:
        y = roi[v]['y']
        # Compute derivative of spline.
        derivspl = spl[v].derivative()

        points = np.array([spl[v](y), y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap=cmap, norm=normi)
        lc.set_array(derivspl(y) * m / n)
        lc.set_linewidth(2)
        plt.gca().add_collection(lc)

    # Plot streamlines.
    ax.streamplot(X, Y, vel * hx / hy, V, density=density,
                  color=vel, linewidth=linewidth, norm=normi, cmap=cmap)

    # Create colourbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=normi)

    fig.tight_layout()
    fig.savefig(os.path.join(path, '{0}-spline-streamlines.png'.format(name)),
                dpi=300)
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
            print("No Kymograph found for {0}!".format(dat))

        # Extract name of kymograph and replace whitespaces.
        name = os.path.splitext(os.path.basename(kymos[0]))[0]
        name = re.sub(' ', '_', name)
        print("Computing velocities for file '{0}'".format(name))

        # Load and preprocess Kymograph.
        img = loadimage(kymos[0])

        # Output first frames of image sequence.
        seq = glob.glob('{0}/{1}*.tif'.format(datfolder, dat))
        if len(seq) != 1:
            warnings.warn("No sequence found!")
        img = tiff.imread(seq)

        frames = img.shape[0] if len(img.shape) is 3 else img.shape[1]
        frames = 10

        for k in range(6, frames - 1):
            if len(img.shape) is 4:
                frame = img[0, k]
                nextframe = img[0, k + 1]
            else:
                frame = img[k]
                nextframe = img[k + 1]

            frame = ndimage.gaussian_filter(frame, sigma=0.5)
            nextframe = ndimage.gaussian_filter(nextframe, sigma=0.5)

            u, v, sig2noise = \
                process.extended_search_area_piv(frame.astype('int32'),
                                                 nextframe.astype('int32'),
                                                 window_size=20,
                                                 overlap=15, dt=0.1,
                                                 search_area_size=20,
                                                 sig2noise_method='peak2peak')
            x, y = process.get_coordinates(image_size=frame.shape,
                                           window_size=20,
                                           overlap=15)
            # u, v, mask = validation.sig2noise_val(u, v, sig2noise, threshold=1)
            # u, v = filters.replace_outliers(u, v, method='localmean', max_iter=5, kernel_size=2)
            # x, y, u, v = openpiv.scaling.uniform(x, y, u, v,
            #                                     scaling_factor=96.52)

            plt.figure()
            plt.imshow(frame)
            plt.quiver(x, y, np.flipud(u), np.flipud(v),
                       color='red', angles='xy', scale_units='xy', scale=1)
            plt.axis('equal')
            plt.show()

            filepath = os.path.join(os.path.join(resultpath, gen), dat)
            # saveimage(filepath, '{0}-{1}'.format(dat, k), frame)
            # saveimage_no_legend(filepath, '{0}-{1}'.format(dat, k), frame)