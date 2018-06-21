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
import itertools
import re
import numpy as np
from scipy import ndimage
from read_roi import read_roi_zip
import ofmc.util.roihelpers as rh
import ofmc.util.pyplothelpers as ph
from ofmc.model.of import of1d_img
from ofmc.model.cms import cms1dl2_img
from ofmc.model.cms import cms1d_img
from ofmc.model.cmscr import cmscr1d_img

# Set path with data.
datapath = ('/Users/lukaslang/'
            'Dropbox (Cambridge University)/Drosophila/Data from Elena')

# Set path where results are saved.
resultpath = 'results/{0}'.format(
        datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
if not os.path.exists(resultpath):
    os.makedirs(resultpath)


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


def error(vel, roi, spl) -> float:
    # Print error.
    error = rh.compute_error(vel, roi, spl)
    totalerr = 0
    for v in roi:
        err = sum(error[v]) / len(error[v])
        totalerr += err
    print("Total error: {0}".format(totalerr))
    return totalerr


print('Processing {0}'.format(datapath))

# Get folders with genotypes.
genotypes = [d for d in os.listdir(datapath)
             if os.path.isdir(os.path.join(datapath, d))]

# Paramters for of1d.
alpha0_of1d = [1e-3, 1e-2, 1e-1]
alpha1_of1d = [1e-3, 1e-2, 1e-1]
prod_of1d = itertools.product(alpha0_of1d, alpha1_of1d)
prod_of1d_len = len(alpha0_of1d) * len(alpha1_of1d)

# Paramters for cms1dl2.
alpha0_cms1dl2 = [1e-3, 1e-2, 1e-1]
alpha1_cms1dl2 = [1e-3, 1e-2, 1e-1]
gamma_cms1dl2 = [1e-3, 1e-2, 1e-1]
prod_cms1dl2 = itertools.product(alpha0_cms1dl2, alpha1_cms1dl2, gamma_cms1dl2)
prod_cms1dl2_len = len(alpha0_cms1dl2) * len(alpha1_cms1dl2) \
    * len(gamma_cms1dl2)

# Paramters for cms1d.
alpha0_cms1d = [1e-3, 1e-2, 1e-1]
alpha1_cms1d = [1e-3, 1e-2, 1e-1]
alpha2_cms1d = [1e-3, 1e-2, 1e-1]
alpha3_cms1d = [1e-3, 1e-2, 1e-1]
prod_cms1d = itertools.product(alpha0_cms1d,
                               alpha1_cms1d,
                               alpha2_cms1d,
                               alpha3_cms1d)
prod_cms1d_len = len(alpha0_cms1d) * len(alpha1_cms1d) \
    * len(alpha2_cms1d) * len(alpha3_cms1d)

# Paramters for cms1dcr.
alpha0_cmscr1d = [1e-3, 1e-2, 1e-1]
alpha1_cmscr1d = [1e-3, 1e-2, 1e-1]
alpha2_cmscr1d = [1e-3, 1e-2, 1e-1]
alpha3_cmscr1d = [1e-3, 1e-2, 1e-1]
beta_cmscr1d = [1e-3, 1e-2, 1e-1]
prod_cmscr1d = itertools.product(alpha0_cmscr1d,
                                 alpha1_cmscr1d,
                                 alpha2_cmscr1d,
                                 alpha3_cmscr1d,
                                 beta_cmscr1d)
prod_cmscr1d_len = len(alpha0_cmscr1d) * len(alpha1_cmscr1d) \
    * len(alpha2_cmscr1d) * len(alpha3_cmscr1d) * len(beta_cmscr1d)

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
        imgp = prepareimage(img)

        # Sanity check.
        roifile = 'manual_ROIs.zip'
        if roifile not in os.listdir(datfolder):
            print("No ROI file found for {0}!".format(dat))
            continue

        # Load roi zip.
        roi = read_roi_zip(os.path.join(datfolder, roifile))

        # Fit splines.
        spl = rh.roi2splines(roi)

        # Compute velocity and source for all parameter pairs.
        err = np.zeros(prod_of1d_len)
        for idx, p in enumerate(prod_of1d):
            vel = of1d_img(imgp, p[0], p[1], 'mesh')
            err[idx] = error(vel, roi, spl)
        print(err)

        # Compute velocity and source for all parameter pairs.
        err = np.zeros(prod_cms1dl2_len)
        for idx, p in enumerate(prod_cms1dl2):
            vel, k = cms1dl2_img(imgp, p[0], p[1], p[2], 'mesh')
            err[idx] = error(vel, roi, spl)
        print(err)

        # Compute velocity and source for all parameter pairs.
        err = np.zeros(prod_cms1d_len)
        for idx, p in enumerate(prod_cms1d):
            vel, k = cms1d_img(imgp, p[0], p[1], p[2], p[3], 'mesh')
            err[idx] = error(vel, roi, spl)
        print(err)

        # Compute velocity and source for all parameter pairs.
        err = np.zeros(prod_cmscr1d_len)
        for idx, p in enumerate(prod_cmscr1d):
            vel, k = cmscr1d_img(imgp, p[0], p[1], p[2], p[3], p[4], 'mesh')
            err[idx] = error(vel, roi, spl)
        print(err)
