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
import logging
import numpy as np
from scipy import ndimage
from read_roi import read_roi_zip
import ofmc.util.roihelpers as rh
import ofmc.util.pyplothelpers as ph
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

# Set regularisation parameter.
alpha0 = 5e-3
alpha1 = 5e-3
alpha2 = 1e-4
alpha3 = 1e-4
beta = 2.5e-3
gamma = 1e-1

# Flat to save images.
savefigs = True


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
        kymos = glob.glob('{0}/SUM_Reslice of {1}*.tif'.format(datfolder, dat))
        if len(kymos) != 1:
            print("No kymograph found for {0}!".format(dat))

        # Extract name of kymograph and replace whitespaces.
        name = os.path.splitext(os.path.basename(kymos[0]))[0]
        name = re.sub(' ', '_', name)
        print("Computing velocities for file '{0}'".format(name))

        # Load and preprocess kymograph.
        img = loadimage(kymos[0])

        # Compute velocity and source.
        imgp = prepareimage(img)
        vel, k, res, fun, converged = cmscr1d_img(imgp, alpha0, alpha1,
                                                  alpha2, alpha3,
                                                  beta, 'mesh')
        # vel, k, res, fun = cms1dl2_img(imgp, alpha0, alpha1, gamma, 'mesh')
        # vel, k, res, fun = cms1d_img(imgp, alpha0, alpha1, alpha2, alpha3,
        #                              'mesh')
        # vel, res, fun = of1d_img(imgp, alpha0, alpha1, 'mesh')

        resfolder = os.path.join(os.path.join(resultpath, gen), dat)
        if not os.path.exists(resfolder):
            os.makedirs(resfolder)

        # Plot and save figures.
        if savefigs:
            ph.saveimage(os.path.join(os.path.join(resultpath, gen), dat),
                         name, img)
            ph.savevelocity(os.path.join(os.path.join(resultpath, gen), dat),
                            name, img, vel)
            ph.savesource(os.path.join(os.path.join(resultpath, gen), dat),
                          name, k)
            ph.savestrainrate(os.path.join(os.path.join(resultpath, gen), dat),
                              name, img, vel)

        # Sanity check.
        roifile = 'manual_ROIs.zip'
        if roifile not in os.listdir(datfolder):
            print("No ROI file found for {0}!".format(dat))
            continue

        # Load roi zip.
        roi = read_roi_zip(os.path.join(datfolder, roifile))

        # Fit splines.
        spl = rh.roi2splines(roi)

        # Print error.
        error = rh.compute_error(vel, roi, spl)
        totalerr = 0
        for v in roi:
            err = sum(error[v]) / len(error[v])
            # print("Error for {0}: {1}".format(v, err))
            totalerr += err
        print("Total error: {0}".format(totalerr))

        # Save tracks and fitted splines.
        if savefigs:
            ph.saveroi(resfolder, name, img, roi)
            ph.savespl(resfolder, name, img, roi, spl)
            ph.saveerror(os.path.join(os.path.join(resultpath, gen), dat),
                         name, img, vel, roi, spl)
            ph.save_spl_streamlines(os.path.join(os.path.join(resultpath, gen),
                                                 dat),
                                    name, img, vel, roi, spl)
            ph.save_roi_streamlines(os.path.join(os.path.join(resultpath, gen),
                                                 dat), name, img, vel, roi)
