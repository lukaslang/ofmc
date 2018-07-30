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
import pickle
import re
import numpy as np
from scipy import ndimage
from read_roi import read_roi_zip
import ofmc.util.roihelpers as rh
from ofmc.model.of import of1d_img
from ofmc.model.cms import cms1dl2_img
from ofmc.model.cms import cms1d_img
from ofmc.model.cmscr import cmscr1d_img

ffc_logger = logging.getLogger('FFC')
ffc_logger.setLevel(logging.WARNING)
ufl_logger = logging.getLogger('UFL')
ufl_logger.setLevel(logging.WARNING)

# Set path with data.
# datapath = ('/Users/lukaslang/'
#             'Dropbox (Cambridge University)/Drosophila/Data from Elena')
datapath = ('/home/ll542/store/'
            'Dropbox (Cambridge University)/Drosophila/Data from Elena')

# Set path where results are saved.
resultpath = 'results/{0}'.format(
        datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
if not os.path.exists(resultpath):
    os.makedirs(resultpath)

# Create path for output files.
if not os.path.exists(os.path.join(resultpath, 'pkl')):
    os.makedirs(os.path.join(resultpath, 'pkl'))

# Set projection method (SUM, MAX, AVG).
proj = 'SUM'

# Set evalution mode for regularisation parameters (SAME, DIFFERENT).
mode = 'SAME'


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


if mode is 'DIFFERENT':
    # Paramters for of1d.
    alpha0_of1d = [1e-3, 1e-2, 1e-1]
    alpha1_of1d = [1e-3, 1e-2, 1e-1]
    prod_of1d = list(it.product(alpha0_of1d, alpha1_of1d))

    # Paramters for cms1dl2.
    alpha0_cms1dl2 = [1e-3, 1e-2, 1e-1]
    alpha1_cms1dl2 = [1e-3, 1e-2, 1e-1]
    gamma_cms1dl2 = [1e-3, 1e-2, 1e-1]
    prod_cms1dl2 = list(it.product(alpha0_cms1dl2,
                                   alpha1_cms1dl2,
                                   gamma_cms1dl2))

    # Paramters for cms1d.
    alpha0_cms1d = [1e-3, 1e-2, 1e-1]
    alpha1_cms1d = [1e-3, 1e-2, 1e-1]
    alpha2_cms1d = [1e-3, 1e-2, 1e-1]
    alpha3_cms1d = [1e-3, 1e-2, 1e-1]
    prod_cms1d = list(it.product(alpha0_cms1d,
                                 alpha1_cms1d,
                                 alpha2_cms1d,
                                 alpha3_cms1d))

    # Paramters for cms1dcr.
    alpha0_cmscr1d = [1e-3, 1e-2, 1e-1]
    alpha1_cmscr1d = [1e-3, 1e-2, 1e-1]
    alpha2_cmscr1d = [1e-3, 1e-2, 1e-1]
    alpha3_cmscr1d = [1e-3, 1e-2, 1e-1]
    beta_cmscr1d = [1e-3, 1e-2, 1e-1]
    prod_cmscr1d = list(it.product(alpha0_cmscr1d,
                                   alpha1_cmscr1d,
                                   alpha2_cmscr1d,
                                   alpha3_cmscr1d,
                                   beta_cmscr1d))

elif mode is 'SAME':
    # Paramters for of1d.
    alpha_v_of1d = [1e-3, 1e-2, 1e-1]
    prod_of1d = [(alpha, alpha) for alpha in alpha_v_of1d]

    # Paramters for cms1dl2.
    alpha_v_cms1dl2 = [1e-3, 1e-2, 1e-1]
    gamma_cms1dl2 = [1e-3, 1e-2, 1e-1]
    prod_cms1dl2 = [(alpha, alpha, gamma) for (alpha, gamma)
                    in it.product(alpha_v_cms1dl2, gamma_cms1dl2)]

    # Paramters for cms1d.
    alpha_v_cms1d = [1e-3, 1e-2, 1e-1]
    alpha_k_cms1d = [1e-3, 1e-2, 1e-1]
    prod_cms1d = [(alpha_v, alpha_v, alpha_k, alpha_k) for (alpha_v, alpha_k)
                  in it.product(alpha_v_cms1d, alpha_k_cms1d)]

    # Paramters for cms1dcr.
    alpha_v_cmscr1d = [1e-3, 1e-2, 1e-1]
    alpha_k_cmscr1d = [1e-3, 1e-2, 1e-1]
    beta_cmscr1d = [1e-3, 1e-2, 1e-1]
    prod_cmscr1d = [(alpha_v, alpha_v, alpha_k, alpha_k, beta)
                    for (alpha_v, alpha_k, beta)
                    in it.product(alpha_v_cmscr1d,
                                  alpha_k_cmscr1d,
                                  beta_cmscr1d)]

# Write parameter settings to file.
with open(os.path.join(resultpath, 'pkl', 'prod_of1d.pkl'), 'wb') as f:
    pickle.dump(prod_of1d, f, pickle.HIGHEST_PROTOCOL)
with open(os.path.join(resultpath, 'pkl', 'prod_cms1dl2.pkl'), 'wb') as f:
    pickle.dump(prod_of1d, f, pickle.HIGHEST_PROTOCOL)
with open(os.path.join(resultpath, 'pkl', 'prod_cms1d.pkl'), 'wb') as f:
    pickle.dump(prod_of1d, f, pickle.HIGHEST_PROTOCOL)
with open(os.path.join(resultpath, 'pkl', 'prod_cmscr1d.pkl'), 'wb') as f:
    pickle.dump(prod_of1d, f, pickle.HIGHEST_PROTOCOL)

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
datasets = dict()
for gen in genotypes:
    # Get folders with datasets.
    datasets[gen] = [d for d in os.listdir(os.path.join(datapath, gen))
                     if os.path.isdir(os.path.join(datapath,
                                                   os.path.join(gen, d)))]
    # Run through datasets.
    for dat in datasets[gen]:
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

# Write data to result folder.
with open(os.path.join(resultpath, 'pkl', 'genotypes.pkl'), 'wb') as f:
    pickle.dump(genotypes, f, pickle.HIGHEST_PROTOCOL)
with open(os.path.join(resultpath, 'pkl', 'datasets.pkl'), 'wb') as f:
    pickle.dump(datasets, f, pickle.HIGHEST_PROTOCOL)
with open(os.path.join(resultpath, 'pkl', 'name.pkl'), 'wb') as f:
    pickle.dump(name, f, pickle.HIGHEST_PROTOCOL)
with open(os.path.join(resultpath, 'pkl', 'img.pkl'), 'wb') as f:
    pickle.dump(img, f, pickle.HIGHEST_PROTOCOL)
with open(os.path.join(resultpath, 'pkl', 'imgp.pkl'), 'wb') as f:
    pickle.dump(imgp, f, pickle.HIGHEST_PROTOCOL)
with open(os.path.join(resultpath, 'pkl', 'roi.pkl'), 'wb') as f:
    pickle.dump(roi, f, pickle.HIGHEST_PROTOCOL)
with open(os.path.join(resultpath, 'pkl', 'spl.pkl'), 'wb') as f:
    pickle.dump(spl, f, pickle.HIGHEST_PROTOCOL)

# Compute velocity and source for all parameter pairs.
print("Running of1d on {0} datasets ".format(num_datasets) +
      "and {0} parameter combinations.".format(len(prod_of1d)))
vel_of1d = [collections.defaultdict(dict) for x in range(len(prod_of1d))]
res_of1d = [collections.defaultdict(dict) for x in range(len(prod_of1d))]
fun_of1d = [collections.defaultdict(dict) for x in range(len(prod_of1d))]
count = 1
for idx, p in enumerate(prod_of1d):
    # Run through datasets.
    for gen in genotypes:
        for dat in datasets[gen]:
            print("{0}/{1}".format(count, num_datasets * len(prod_of1d)))
            vel_of1d[idx][gen][dat], \
                res_of1d[idx][gen][dat], \
                fun_of1d[idx][gen][dat] = of1d_img(imgp[gen][dat],
                                                   p[0], p[1], 'mesh')
            count += 1

# Store results.
with open(os.path.join(resultpath, 'pkl', 'vel_of1d.pkl'), 'wb') as f:
    pickle.dump(vel_of1d, f, pickle.HIGHEST_PROTOCOL)
with open(os.path.join(resultpath, 'pkl', 'res_of1d.pkl'), 'wb') as f:
    pickle.dump(res_of1d, f, pickle.HIGHEST_PROTOCOL)
with open(os.path.join(resultpath, 'pkl', 'fun_of1d.pkl'), 'wb') as f:
    pickle.dump(fun_of1d, f, pickle.HIGHEST_PROTOCOL)

# Clear memory.
del vel_of1d

# Compute velocity and source for all parameter pairs.
print("Running cms1dl2 on {0} datasets ".format(num_datasets) +
      "and {0} parameter combinations.".format(len(prod_cms1dl2)))
vel_cms1dl2 = [collections.defaultdict(dict) for x in range(len(prod_cms1dl2))]
k_cms1dl2 = [collections.defaultdict(dict) for x in range(len(prod_cms1dl2))]
res_cms1dl2 = [collections.defaultdict(dict) for x in range(len(prod_cms1dl2))]
fun_cms1dl2 = [collections.defaultdict(dict) for x in range(len(prod_cms1dl2))]
count = 1
for idx, p in enumerate(prod_cms1dl2):
    # Run through datasets.
    for gen in genotypes:
        for dat in datasets[gen]:
            print("{0}/{1}".format(count, num_datasets * len(prod_cms1dl2)))
            vel_cms1dl2[idx][gen][dat], k_cms1dl2[idx][gen][dat], \
                res_cms1dl2[idx][gen][dat], fun_cms1dl2[idx][gen][dat] = \
                cms1dl2_img(imgp[gen][dat], p[0], p[1], p[2], 'mesh')
            count += 1

# Store results.
with open(os.path.join(resultpath, 'pkl', 'vel_cms1dl2.pkl'), 'wb') as f:
    pickle.dump(vel_cms1dl2, f, pickle.HIGHEST_PROTOCOL)
with open(os.path.join(resultpath, 'pkl', 'k_cms1dl2.pkl'), 'wb') as f:
    pickle.dump(k_cms1dl2, f, pickle.HIGHEST_PROTOCOL)
with open(os.path.join(resultpath, 'pkl', 'res_cms1dl2.pkl'), 'wb') as f:
    pickle.dump(res_cms1dl2, f, pickle.HIGHEST_PROTOCOL)
with open(os.path.join(resultpath, 'pkl', 'fun_cms1dl2.pkl'), 'wb') as f:
    pickle.dump(fun_cms1dl2, f, pickle.HIGHEST_PROTOCOL)

# Clear memory.
del vel_cms1dl2, k_cms1dl2

# Compute velocity and source for all parameter pairs.
print("Running cms1d on {0} datasets ".format(num_datasets) +
      "and {0} parameter combinations.".format(len(prod_cms1d)))
vel_cms1d = [collections.defaultdict(dict) for x in range(len(prod_cms1d))]
k_cms1d = [collections.defaultdict(dict) for x in range(len(prod_cms1d))]
res_cms1d = [collections.defaultdict(dict) for x in range(len(prod_cms1d))]
fun_cms1d = [collections.defaultdict(dict) for x in range(len(prod_cms1d))]
count = 1
for idx, p in enumerate(prod_cms1d):
    # Run through datasets.
    for gen in genotypes:
        for dat in datasets[gen]:
            print("{0}/{1}".format(count, num_datasets * len(prod_cms1d)))
            vel_cms1d[idx][gen][dat], k_cms1d[idx][gen][dat], \
                res_cms1d[idx][gen][dat], fun_cms1d[idx][gen][dat] = \
                cms1d_img(imgp[gen][dat], p[0], p[1], p[2], p[3], 'mesh')
            count += 1

# Store results.
with open(os.path.join(resultpath, 'pkl', 'vel_cms1d.pkl'), 'wb') as f:
    pickle.dump(vel_cms1d, f, pickle.HIGHEST_PROTOCOL)
with open(os.path.join(resultpath, 'pkl', 'k_cms1d.pkl'), 'wb') as f:
    pickle.dump(k_cms1d, f, pickle.HIGHEST_PROTOCOL)
with open(os.path.join(resultpath, 'pkl', 'res_cms1d.pkl'), 'wb') as f:
    pickle.dump(res_cms1d, f, pickle.HIGHEST_PROTOCOL)
with open(os.path.join(resultpath, 'pkl', 'fun_cms1d.pkl'), 'wb') as f:
    pickle.dump(fun_cms1d, f, pickle.HIGHEST_PROTOCOL)

# Clear memory.
del vel_cms1d, k_cms1d

# Compute velocity and source for all parameter pairs.
print("Running cmscr1d on {0} datasets ".format(num_datasets) +
      "and {0} parameter combinations.".format(len(prod_cmscr1d)))
vel_cmscr1d = [collections.defaultdict(dict) for x in range(len(prod_cmscr1d))]
k_cmscr1d = [collections.defaultdict(dict) for x in range(len(prod_cmscr1d))]
res_cmscr1d = [collections.defaultdict(dict) for x in range(len(prod_cmscr1d))]
fun_cmscr1d = [collections.defaultdict(dict) for x in range(len(prod_cmscr1d))]
converged_cmscr1d = [collections.defaultdict(dict)
                     for x in range(len(prod_cmscr1d))]
count = 1
for idx, p in enumerate(prod_cmscr1d):
    # Run through datasets.
    for gen in genotypes:
        for dat in datasets[gen]:
            print("{0}/{1}".format(count, num_datasets * len(prod_cmscr1d)))
            vel_cmscr1d[idx][gen][dat], \
                k_cmscr1d[idx][gen][dat], \
                res_cmscr1d[idx][gen][dat], \
                fun_cmscr1d[idx][gen][dat], \
                converged_cmscr1d[idx][gen][dat] = \
                cmscr1d_img(imgp[gen][dat],
                            p[0], p[1], p[2], p[3], p[4], 'mesh')
            count += 1

# Store results.
with open(os.path.join(resultpath, 'pkl', 'vel_cmscr1d.pkl'), 'wb') as f:
    pickle.dump(vel_cmscr1d, f, pickle.HIGHEST_PROTOCOL)
with open(os.path.join(resultpath, 'pkl', 'k_cmscr1d.pkl'), 'wb') as f:
    pickle.dump(k_cmscr1d, f, pickle.HIGHEST_PROTOCOL)
with open(os.path.join(resultpath, 'pkl', 'res_cmscr1d.pkl'), 'wb') as f:
    pickle.dump(res_cmscr1d, f, pickle.HIGHEST_PROTOCOL)
with open(os.path.join(resultpath, 'pkl', 'fun_cmscr1d.pkl'), 'wb') as f:
    pickle.dump(fun_cmscr1d, f, pickle.HIGHEST_PROTOCOL)
with open(os.path.join(resultpath, 'pkl', 'converged_cmscr1d.pkl'), 'wb') as f:
    pickle.dump(converged_cmscr1d, f, pickle.HIGHEST_PROTOCOL)

# Clear memory.
del vel_cmscr1d, k_cmscr1d, converged_cmscr1d
