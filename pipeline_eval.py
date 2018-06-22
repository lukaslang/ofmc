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
import collections
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

# Set projection method.
proj = 'SUM'


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


# Paramters for of1d.
alpha0_of1d = [1e-2]
alpha1_of1d = [1e-2, 1e-1]
prod_of1d = itertools.product(alpha0_of1d, alpha1_of1d)
prod_of1d_len = len(alpha0_of1d) * len(alpha1_of1d)

# Paramters for cms1dl2.
alpha0_cms1dl2 = [1e-2]
alpha1_cms1dl2 = [1e-2]
gamma_cms1dl2 = [1e-2, 1e-1]
prod_cms1dl2 = itertools.product(alpha0_cms1dl2, alpha1_cms1dl2, gamma_cms1dl2)
prod_cms1dl2_len = len(alpha0_cms1dl2) * len(alpha1_cms1dl2) \
    * len(gamma_cms1dl2)

# Paramters for cms1d.
alpha0_cms1d = [1e-2]
alpha1_cms1d = [1e-2]
alpha2_cms1d = [1e-2]
alpha3_cms1d = [1e-2, 1e-1]
prod_cms1d = itertools.product(alpha0_cms1d,
                               alpha1_cms1d,
                               alpha2_cms1d,
                               alpha3_cms1d)
prod_cms1d_len = len(alpha0_cms1d) * len(alpha1_cms1d) \
    * len(alpha2_cms1d) * len(alpha3_cms1d)

# Paramters for cms1dcr.
alpha0_cmscr1d = [1e-2]
alpha1_cmscr1d = [1e-2]
alpha2_cmscr1d = [1e-2]
alpha3_cmscr1d = [1e-2]
beta_cmscr1d = [1e-2, 1e-1]
prod_cmscr1d = itertools.product(alpha0_cmscr1d,
                                 alpha1_cmscr1d,
                                 alpha2_cmscr1d,
                                 alpha3_cmscr1d,
                                 beta_cmscr1d)
prod_cmscr1d_len = len(alpha0_cmscr1d) * len(alpha1_cmscr1d) \
    * len(alpha2_cmscr1d) * len(alpha3_cmscr1d) * len(beta_cmscr1d)


print('Processing {0}'.format(datapath))

# Initialise dicts.
name = collections.defaultdict(dict)
img = collections.defaultdict(dict)
imgp = collections.defaultdict(dict)
roi = collections.defaultdict(dict)
spl = collections.defaultdict(dict)

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
        kymos = glob.glob('{0}/{1}_Reslice of {2}*.tif'.format(datfolder,
                                                               proj, dat))
        if len(kymos) != 1:
            print("No Kymograph found for {0}!".format(dat))

        # Extract name of kymograph and replace whitespaces.
        filename = os.path.splitext(os.path.basename(kymos[0]))[0]
        name[gen][dat] = re.sub(' ', '_', filename)
        print("Loading file '{0}'".format(name[gen][dat]))

        # Load and preprocess Kymograph.
        img[gen][dat] = loadimage(kymos[0])
        imgp[gen][dat] = prepareimage(img[gen][dat])

        # Sanity check.
        roifile = 'manual_ROIs.zip'
        if roifile not in os.listdir(datfolder):
            print("No ROI file found for {0}!".format(dat))
            continue

        # Load roi zip.
        roi[gen][dat] = read_roi_zip(os.path.join(datfolder, roifile))

        # Fit splines.
        spl[gen][dat] = rh.roi2splines(roi[gen][dat])

# Compute velocity and source for all parameter pairs.
vel_of1d = [collections.defaultdict(dict) for x in range(prod_of1d_len)]
err_of1d = [collections.defaultdict(dict) for x in range(prod_of1d_len)]
for idx, p in enumerate(prod_of1d):
    # Run through datasets.
    for gen in name.keys():
        for dat in name[gen].keys():
            vel_of1d[idx][gen][dat] = of1d_img(imgp[gen][dat],
                                               p[0], p[1], 'mesh')
            err_of1d[idx][gen][dat] = error(vel_of1d[idx][gen][dat],
                                            roi[gen][dat], spl[gen][dat])


# Compute velocity and source for all parameter pairs.
vel_cms1dl2 = [collections.defaultdict(dict) for x in range(prod_cms1dl2_len)]
k_cms1dl2 = [collections.defaultdict(dict) for x in range(prod_cms1dl2_len)]
err_cms1dl2 = [collections.defaultdict(dict) for x in range(prod_cms1dl2_len)]
for idx, p in enumerate(prod_cms1dl2):
    # Run through datasets.
    for gen in name.keys():
        for dat in name[gen].keys():
            vel_cms1dl2[idx][gen][dat], k_cms1dl2[idx][gen][dat] = \
                cms1dl2_img(imgp[gen][dat], p[0], p[1], p[2], 'mesh')
            err_cms1dl2[idx][gen][dat] = error(vel_cms1dl2[idx][gen][dat],
                                               roi[gen][dat], spl[gen][dat])

# Compute velocity and source for all parameter pairs.
vel_cms1d = [collections.defaultdict(dict) for x in range(prod_cms1d_len)]
k_cms1d = [collections.defaultdict(dict) for x in range(prod_cms1d_len)]
err_cms1d = [collections.defaultdict(dict) for x in range(prod_cms1d_len)]
for idx, p in enumerate(prod_cms1d):
    # Run through datasets.
    for gen in name.keys():
        for dat in name[gen].keys():
            vel_cms1d[idx][gen][dat], k_cms1d[idx][gen][dat] = \
                cms1d_img(imgp[gen][dat], p[0], p[1], p[2], p[3], 'mesh')
            err_cms1d[idx][gen][dat] = error(vel_cms1d[idx][gen][dat],
                                             roi[gen][dat], spl[gen][dat])

# Compute velocity and source for all parameter pairs.
vel_cmscr1d = [collections.defaultdict(dict) for x in range(prod_cmscr1d_len)]
k_cmscr1d = [collections.defaultdict(dict) for x in range(prod_cmscr1d_len)]
err_cmscr1d = [collections.defaultdict(dict) for x in range(prod_cmscr1d_len)]
for idx, p in enumerate(prod_cmscr1d):
    # Run through datasets.
    for gen in name.keys():
        for dat in name[gen].keys():
            vel_cmscr1d[idx][gen][dat], k_cmscr1d[idx][gen][dat] = \
                cmscr1d_img(imgp[gen][dat],
                            p[0], p[1], p[2], p[3], p[4], 'mesh')
            err_cmscr1d[idx][gen][dat] = error(vel_cmscr1d[idx][gen][dat],
                                               roi[gen][dat], spl[gen][dat])


# Print errors.
for gen in name.keys():
    for dat in name[gen].keys():
        print("Dataset {0}/{1}".format(gen, dat))
        err = [x[gen][dat] for x in err_of1d]
        print("of1d:    " + ", ".join('{0:.3f}'.format(x) for x in err))
        err = [x[gen][dat] for x in err_cms1dl2]
        print("cms1dl2: " + ", ".join('{0:.3f}'.format(x) for x in err))
        err = [x[gen][dat] for x in err_cms1d]
        print("cms1d:   " + ", ".join('{0:.3f}'.format(x) for x in err))
        err = [x[gen][dat] for x in err_cmscr1d]
        print("cmscr1d: " + ", ".join('{0:.3f}'.format(x) for x in err))
