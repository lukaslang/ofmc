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
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import warnings
import ofmc.external.tifffile as tiff
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Set font style.
font = {'weight': 'normal',
        'size': 20}
plt.rc('font', **font)


# Set path with data.
datapath = ('/Users/lukaslang/'
            'Dropbox (Cambridge University)/Drosophila/Data from Elena')

# Set path where results are saved.
resultpath = 'results/figures'
if not os.path.exists(resultpath):
    os.makedirs(resultpath)


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
    fig.tight_layout()
    fig.savefig(os.path.join(path, '{0}.png'.format(name)),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def saveimage_no_legend(path: str, name: str, img: np.array):
    if not os.path.exists(path):
        os.makedirs(path)

    # Plot image.
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(img, cmap=cm.gray)

    # Save figure.
    fig.tight_layout()
    fig.savefig(os.path.join(path, '{0}.png'.format(name)),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def savekymo(path: str, name: str, img: np.array):
    if not os.path.exists(path):
        os.makedirs(path)

    # Plot image.
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(img, cmap=cm.gray)
    ax.set_xlabel('Space')
    ax.set_ylabel('Time')
    ax.set_title('Fluorescence intensity')

    # Create colourbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # Save figure.
    fig.tight_layout()
    fig.savefig(os.path.join(path, '{0}.png'.format(name)),
                dpi=300, bbox_inches='tight')
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

        # Extract name of kymograph and replace whitespaces.
        name = os.path.splitext(os.path.basename(kymos[0]))[0]
        name = re.sub(' ', '_', name)
        print("Outputting file '{0}'".format(name))

        # Load and preprocess Kymograph.
        img = imageio.imread(kymos[0])

        # Plot and save figures.
        # savekymo(os.path.join(os.path.join(resultpath, gen), dat), name, img)
        savekymo(os.path.join(resultpath, gen), name, img)

        # Output first frames of image sequence.
        seq = glob.glob('{0}/{1}*.tif'.format(datfolder, dat))
        if len(seq) != 1:
            warnings.warn("No sequence found!")
        img = tiff.imread(seq)

        frames = img.shape[0] if len(img.shape) is 3 else img.shape[1]

        for k in range(frames):
            if len(img.shape) is 4:
                frame = img[0, k]
            else:
                frame = img[k]

            filepath = os.path.join(os.path.join(resultpath, gen), dat)
            # saveimage(filepath, '{0}-{1}'.format(dat, k), frame)
            saveimage_no_legend(filepath, '{0}-{1}'.format(dat, k), frame)