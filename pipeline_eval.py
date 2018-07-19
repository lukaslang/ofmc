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
import itertools as it
import logging
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

ffc_logger = logging.getLogger('FFC')
ffc_logger.setLevel(logging.WARNING)
ufl_logger = logging.getLogger('UFL')
ufl_logger.setLevel(logging.WARNING)

# Set path with data.
datapath = ('/Users/lukaslang/'
            'Dropbox (Cambridge University)/Drosophila/Data from Elena test')

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


def error(vel, roi, spl) -> (float, float):
    # Compute accumulated error in velocity for each spline.
    error = rh.compute_error(vel, roi, spl)
    totalerr = 0
    maxerror = 0
    for v in roi:
        err = sum(error[v]) / len(error[v])
        totalerr += err
        maxerror = max(maxerror, max(error[v]))
    return (totalerr, maxerror)


# Paramters for of1d.
alpha0_of1d = [1e-1]
alpha1_of1d = [1e-1]
prod_of1d = it.product(alpha0_of1d, alpha1_of1d)
prod_of1d_len = len(alpha0_of1d) * len(alpha1_of1d)

# Paramters for cms1dl2.
alpha0_cms1dl2 = [1e-1]
alpha1_cms1dl2 = [1e-1]
gamma_cms1dl2 = [1e-1]
prod_cms1dl2 = it.product(alpha0_cms1dl2, alpha1_cms1dl2, gamma_cms1dl2)
prod_cms1dl2_len = len(alpha0_cms1dl2) * len(alpha1_cms1dl2) \
    * len(gamma_cms1dl2)

# Paramters for cms1d.
alpha0_cms1d = [1e-1]
alpha1_cms1d = [1e-1]
alpha2_cms1d = [1e-1]
alpha3_cms1d = [1e-1]
prod_cms1d = it.product(alpha0_cms1d,
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
beta_cmscr1d = [1e-3]
prod_cmscr1d_1, prod_cmscr1d_2 = it.tee(it.product(alpha0_cmscr1d,
                                                   alpha1_cmscr1d,
                                                   alpha2_cmscr1d,
                                                   alpha3_cmscr1d,
                                                   beta_cmscr1d), 2)
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
num_datasets = 0
for gen in genotypes:
    # Get folders with datasets.
    datasets = [d for d in os.listdir(os.path.join(datapath, gen))
                if os.path.isdir(os.path.join(datapath,
                                              os.path.join(gen, d)))]
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

        # Increase counter.
        num_datasets += 1

# Compute velocity and source for all parameter pairs.
print("Running of1d on {0} datasets ".format(num_datasets) +
      "and {0} parameter combinations.".format(prod_of1d_len))
vel_of1d = [collections.defaultdict(dict) for x in range(prod_of1d_len)]
err_of1d = [collections.defaultdict(dict) for x in range(prod_of1d_len)]
max_err_of1d = [collections.defaultdict(dict) for x in range(prod_of1d_len)]
count = 1
for idx, p in enumerate(prod_of1d):
    # Run through datasets.
    for gen in name.keys():
        for dat in name[gen].keys():
            print("{0}/{1}".format(count, num_datasets * prod_of1d_len))
            vel_of1d[idx][gen][dat] = of1d_img(imgp[gen][dat],
                                               p[0], p[1], 'mesh')
            err_of1d[idx][gen][dat], max_err_of1d[idx][gen][dat] = \
                error(vel_of1d[idx][gen][dat], roi[gen][dat], spl[gen][dat])
            count += 1


# Compute velocity and source for all parameter pairs.
print("Running cms1dl2 on {0} datasets ".format(num_datasets) +
      "and {0} parameter combinations.".format(prod_cms1dl2_len))
vel_cms1dl2 = [collections.defaultdict(dict) for x in range(prod_cms1dl2_len)]
k_cms1dl2 = [collections.defaultdict(dict) for x in range(prod_cms1dl2_len)]
err_cms1dl2 = [collections.defaultdict(dict) for x in range(prod_cms1dl2_len)]
max_err_cms1dl2 = [collections.defaultdict(dict) for
                   x in range(prod_cms1dl2_len)]
count = 1
for idx, p in enumerate(prod_cms1dl2):
    # Run through datasets.
    for gen in name.keys():
        for dat in name[gen].keys():
            print("{0}/{1}".format(count, num_datasets * prod_cms1dl2_len))
            vel_cms1dl2[idx][gen][dat], k_cms1dl2[idx][gen][dat] = \
                cms1dl2_img(imgp[gen][dat], p[0], p[1], p[2], 'mesh')
            err_cms1dl2[idx][gen][dat], max_err_cms1dl2[idx][gen][dat] = \
                error(vel_cms1dl2[idx][gen][dat], roi[gen][dat], spl[gen][dat])
            count += 1

# Compute velocity and source for all parameter pairs.
print("Running cms1d on {0} datasets ".format(num_datasets) +
      "and {0} parameter combinations.".format(prod_cms1d_len))
vel_cms1d = [collections.defaultdict(dict) for x in range(prod_cms1d_len)]
k_cms1d = [collections.defaultdict(dict) for x in range(prod_cms1d_len)]
err_cms1d = [collections.defaultdict(dict) for x in range(prod_cms1d_len)]
max_err_cms1d = [collections.defaultdict(dict) for x in range(prod_cms1d_len)]
count = 1
for idx, p in enumerate(prod_cms1d):
    # Run through datasets.
    for gen in name.keys():
        for dat in name[gen].keys():
            print("{0}/{1}".format(count, num_datasets * prod_cms1d_len))
            vel_cms1d[idx][gen][dat], k_cms1d[idx][gen][dat] = \
                cms1d_img(imgp[gen][dat], p[0], p[1], p[2], p[3], 'mesh')
            err_cms1d[idx][gen][dat], max_err_cms1d[idx][gen][dat] = \
                error(vel_cms1d[idx][gen][dat], roi[gen][dat], spl[gen][dat])
            count += 1

# Compute velocity and source for all parameter pairs.
print("Running cmscr1d on {0} datasets ".format(num_datasets) +
      "and {0} parameter combinations.".format(prod_cmscr1d_len))
vel_cmscr1d = [collections.defaultdict(dict) for x in range(prod_cmscr1d_len)]
k_cmscr1d = [collections.defaultdict(dict) for x in range(prod_cmscr1d_len)]
err_cmscr1d = [collections.defaultdict(dict) for x in range(prod_cmscr1d_len)]
max_err_cmscr1d = [collections.defaultdict(dict) for
                   x in range(prod_cmscr1d_len)]
count = 1
for idx, p in enumerate(prod_cmscr1d_1):
    # Run through datasets.
    for gen in name.keys():
        for dat in name[gen].keys():
            print("{0}/{1}".format(count, num_datasets * prod_cmscr1d_len))
            vel_cmscr1d[idx][gen][dat], k_cmscr1d[idx][gen][dat] = \
                cmscr1d_img(imgp[gen][dat],
                            p[0], p[1], p[2], p[3], p[4], 'mesh')
            err_cmscr1d[idx][gen][dat], max_err_cmscr1d[idx][gen][dat] = \
                error(vel_cmscr1d[idx][gen][dat], roi[gen][dat], spl[gen][dat])
            count += 1


# Print errors.
print('Cumulative absolute error:')
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

# Print max. errors.
print('Maximum absolute error:')
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

# Output best result for each method and each dataset.
for gen in name.keys():
    for dat in name[gen].keys():
        # Find indices of best results (not necessarily unique).
        idx_of1d = np.argmin([x[gen][dat] for x in err_of1d])
        idx_cms1dl2 = np.argmin([x[gen][dat] for x in err_cms1dl2])
        idx_cms1d = np.argmin([x[gen][dat] for x in err_cms1d])
        idx_cmscr1d = np.argmin([x[gen][dat] for x in err_cmscr1d])

        tmpimg = img[gen][dat]
        tmpname = name[gen][dat]
        tmproi = roi[gen][dat]
        tmpspl = spl[gen][dat]

        # of1d
        tmpfolder = [resultpath, 'of1d', gen, dat]
        resfolder = os.path.join(*tmpfolder)
        if not os.path.exists(resfolder):
            os.makedirs(resfolder)
        tmpvel = vel_of1d[idx_of1d][gen][dat]
        ph.savevelocity(resfolder, tmpname, tmpimg, tmpvel)
        ph.saveroi(resfolder, tmpname, tmpimg, tmproi)
        ph.savespl(resfolder, tmpname, tmpimg, tmproi, tmpspl)
        ph.saveerror(resfolder, tmpname, tmpimg, tmpvel, tmproi, tmpspl)
        ph.save_spl_streamlines(resfolder, tmpname, tmpimg,
                                tmpvel, tmproi, tmpspl)
        ph.save_roi_streamlines(resfolder, tmpname, tmpimg, tmpvel, tmproi)

        # cms1d
        tmpfolder = [resultpath, 'cms1d', gen, dat]
        resfolder = os.path.join(*tmpfolder)
        if not os.path.exists(resfolder):
            os.makedirs(resfolder)
        tmpvel = vel_cms1d[idx_cms1d][gen][dat]
        tmpk = k_cms1d[idx_cms1d][gen][dat]
        ph.savevelocity(resfolder, tmpname, tmpimg, tmpvel)
        ph.savesource(resfolder, tmpname, tmpk)
        ph.saveroi(resfolder, tmpname, tmpimg, tmproi)
        ph.savespl(resfolder, tmpname, tmpimg, tmproi, tmpspl)
        ph.saveerror(resfolder, tmpname, tmpimg, tmpvel, tmproi, tmpspl)
        ph.save_spl_streamlines(resfolder, tmpname, tmpimg,
                                tmpvel, tmproi, tmpspl)
        ph.save_roi_streamlines(resfolder, tmpname, tmpimg, tmpvel, tmproi)

        # cms1dl2
        tmpfolder = [resultpath, 'cms1dl2', gen, dat]
        resfolder = os.path.join(*tmpfolder)
        if not os.path.exists(resfolder):
            os.makedirs(resfolder)
        tmpvel = vel_cms1dl2[idx_cms1dl2][gen][dat]
        tmpk = k_cms1dl2[idx_cms1dl2][gen][dat]
        ph.savevelocity(resfolder, tmpname, tmpimg, tmpvel)
        ph.savesource(resfolder, tmpname, tmpk)
        ph.saveroi(resfolder, tmpname, tmpimg, tmproi)
        ph.savespl(resfolder, tmpname, tmpimg, tmproi, tmpspl)
        ph.saveerror(resfolder, tmpname, tmpimg, tmpvel, tmproi, tmpspl)
        ph.save_spl_streamlines(resfolder, tmpname, tmpimg,
                                tmpvel, tmproi, tmpspl)
        ph.save_roi_streamlines(resfolder, tmpname, tmpimg, tmpvel, tmproi)

        # cmscr1d
        tmpfolder = [resultpath, 'cmscr1d', gen, dat]
        resfolder = os.path.join(*tmpfolder)
        if not os.path.exists(resfolder):
            os.makedirs(resfolder)
        tmpvel = vel_cmscr1d[idx_cmscr1d][gen][dat]
        tmpk = k_cmscr1d[idx_cmscr1d][gen][dat]
        ph.savevelocity(resfolder, tmpname, tmpimg, tmpvel)
        ph.savesource(resfolder, tmpname, tmpk)
        ph.saveroi(resfolder, tmpname, tmpimg, tmproi)
        ph.savespl(resfolder, tmpname, tmpimg, tmproi, tmpspl)
        ph.saveerror(resfolder, tmpname, tmpimg, tmpvel, tmproi, tmpspl)
        ph.save_spl_streamlines(resfolder, tmpname, tmpimg,
                                tmpvel, tmproi, tmpspl)
        ph.save_roi_streamlines(resfolder, tmpname, tmpimg, tmpvel, tmproi)


# Output LaTeX table in sorte order.
print('LaTeX table with results:')
for gen in sorted(name.keys()):
    for dat in sorted(name[gen].keys()):
        # Find indices of best results (not necessarily unique).
        idx_of1d = np.argmin([x[gen][dat] for x in err_of1d])
        idx_cms1dl2 = np.argmin([x[gen][dat] for x in err_cms1dl2])
        idx_cms1d = np.argmin([x[gen][dat] for x in err_cms1d])
        idx_cmscr1d = np.argmin([x[gen][dat] for x in err_cmscr1d])

        formatstr = '{0}/{1} & {2:.2f} & {3:.2f} & {4:.2f} & {5:.2f} \\\\'
        print(formatstr.format(re.sub('_', '\\_', gen),
                               re.sub('_', '\\_', dat),
                               err_of1d[idx_of1d][gen][dat],
                               err_cms1dl2[idx_cms1dl2][gen][dat],
                               err_cms1d[idx_cms1d][gen][dat],
                               err_cmscr1d[idx_cmscr1d][gen][dat]))
print('\\hline')

# Output average over all datasets.
# Output LaTeX table in sorte order.
sum_of1d = 0.0
sum_cms1dl2 = 0.0
sum_cms1d = 0.0
sum_cmscr1d = 0.0
count = 0.0
for gen in sorted(name.keys()):
    for dat in sorted(name[gen].keys()):
        # Find indices of best results (not necessarily unique).
        idx_of1d = np.argmin([x[gen][dat] for x in err_of1d])
        idx_cms1dl2 = np.argmin([x[gen][dat] for x in err_cms1dl2])
        idx_cms1d = np.argmin([x[gen][dat] for x in err_cms1d])
        idx_cmscr1d = np.argmin([x[gen][dat] for x in err_cmscr1d])

        sum_of1d += err_of1d[idx_of1d][gen][dat]
        sum_cms1dl2 += err_cms1dl2[idx_cms1dl2][gen][dat]
        sum_cms1d += err_cms1d[idx_cms1d][gen][dat]
        sum_cmscr1d += err_cmscr1d[idx_cmscr1d][gen][dat]
        count += 1.0

formatstr = 'Average & {0:.2f} & {1:.2f} & {2:.2f} & {3:.2f} \\\\'
print(formatstr.format(sum_of1d / count,
                       sum_cms1dl2 / count,
                       sum_cms1d / count,
                       sum_cmscr1d / count))
