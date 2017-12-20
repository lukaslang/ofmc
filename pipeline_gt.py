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
import numpy as np
from scipy import misc
from read_roi import read_roi_zip
from matplotlib import cm
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import ofmc.util.roihelpers as rh

# Set path with data.
datapath = ('/Users/lukaslang/'
            'Dropbox (Cambridge University)/Drosophila/Data from Elena')

# Set path where results are saved.
resultpath = 'results'


def loadimage(filename: str) -> np.array:
    return misc.imread(filename)


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
        plt.show()

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
        plt.show()

        # Save figure.
        fig.savefig(os.path.join(resultpath, '{0}-spline.png'.format(dat)))
        plt.close(fig)
