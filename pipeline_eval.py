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
import imageio
import numpy as np
from scipy import ndimage
from scipy import interpolate
from read_roi import read_roi_zip
from matplotlib import cm
from matplotlib.collections import LineCollection
import matplotlib as mpl
import matplotlib.pyplot as plt
import ofmc.util.roihelpers as rh
from ofmc.model.cmscr import cmscr1d_img

# Set path with data.
datapath = ('/Users/lukaslang/'
            'Dropbox (Cambridge University)/Drosophila/Data from Elena')

# Set path where results are saved.
resultpath = 'results'
if not os.path.exists(resultpath):
    os.makedirs(resultpath)

# Set regularisation parameter.
alpha0 = 5e-3
alpha1 = 1e-2
alpha2 = 1e-4
alpha3 = 1e-4
beta = 1e-3


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
        kymos = glob.glob('{0}/MAX_Reslice of {1}*.tif'.format(datfolder, dat))
        if len(kymos) != 1:
            print("No Kymograph found for {0}!".format(dat))

        # Load and preprocess Kymograph.
        img = loadimage(kymos[0])

        # Sanity check.
        roifile = 'manual_ROIs.zip'
        if roifile not in os.listdir(datfolder):
            print("No Kymograph found for {0}!".format(dat))
            continue

        # Load roi zip.
        roi = read_roi_zip(os.path.join(datfolder, roifile))

        # Plot image.
        fig = plt.figure()
        plt.imshow(img, cmap=cm.gray)

        for v in roi:
            plt.plot(roi[v]['x'], roi[v]['y'], 'C3', lw=2)

        # Save figure.
        fig.savefig(os.path.join(resultpath, '{0}-roi.png'.format(dat)))
        plt.close(fig)

        # Fit splines.
        spl = rh.roi2splines(roi)

        # Plot image.
        fig = plt.figure()
        plt.imshow(img, cmap=cm.gray)

        # Plot splines.
        for v in roi:
            y = roi[v]['y']
            # Compute derivative of spline.
            derivspl = spl[v].derivative()

            points = np.array([spl[v](y), y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(segments, cmap=cm.coolwarm,
                                norm=plt.Normalize(-2, 2))
            lc.set_array(derivspl(y))
            lc.set_linewidth(2)
            plt.gca().add_collection(lc)

        # Save figure.
        fig.savefig(os.path.join(resultpath, '{0}-spline.png'.format(dat)))
        plt.close(fig)

        # Compute velocity and source.
        imgp = prepareimage(img)
        vel, k = cmscr1d_img(imgp, alpha0, alpha1, alpha2, alpha3,
                             beta, 'fd')

        # Interpolate velocity.
        m, n = vel.shape
        gridx, gridy = np.mgrid[0:m, 0:n]
        gridpoints = np.hstack([gridx.reshape(m*n, 1), gridy.reshape(m*n, 1)])

        # Plot image.
        # fig = plt.figure()
        # plt.imshow(imgp, cmap=cm.gray)

        fig, ax = plt.subplots(figsize=(10, 5))
        plt.imshow(imgp, cmap=cm.gray)

        maxvel = abs(vel).max()
        normi = mpl.colors.Normalize(vmin=-maxvel, vmax=maxvel)

        # Plot velocity.
        # fig, ax = plt.subplots(figsize=(10, 5))
        # cax = ax.imshow(vel, interpolation='nearest', norm=normi,
        #                cmap=cm.coolwarm)
        # ax.set_title('Velocity')
        # fig.colorbar(cax, orientation='horizontal')

        # Evaluate velocity for each spline.
        for v in roi:
            y = roi[v]['y']

            # Interpolate velocity.
            # y = np.array(range(y[0], y[-1] + 1))
            y = np.arange(y[0], y[-1] + 1, 0.5)
            x = np.array(spl[v](y))
            veval = interpolate.griddata(gridpoints, vel.flatten(), (y, x),
                                         method='cubic')

            points = np.array([spl[v](y), y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(segments, cmap=cm.coolwarm, norm=normi)

            lc.set_array(veval)
            lc.set_linewidth(2)
            plt.gca().add_collection(lc)

        m, n = vel.shape
        hx, hy = 1./(m-1), 1./(n-1)

        # Create grid for streamlines.
        Y, X = np.mgrid[0:m, 0:n]
        V = np.ones_like(X)*hy

        # Plot streamlines.
        strm = ax.streamplot(X, Y, vel*hx, V, density=2,
                             color=vel, linewidth=1, norm=normi,
                             cmap=cm.coolwarm)

        # Save figure.
        fig.savefig(os.path.join(resultpath, '{0}-velocity.png'.format(dat)))
        plt.close(fig)
