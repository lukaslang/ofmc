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
# Figure 1 and 2: Plots a dataset and a kymograph.
import datetime
import glob
import imageio
import logging
import numpy as np
import os
import re
import warnings
import tifffile.tifffile as tiff
import ofmc.util.pyplothelpers as ph
import ofmc.util.roihelpers as rh
from read_roi import read_roi_zip
from scipy import ndimage
import datapath as dp

# Get path where data is located.
datapath = dp.path

ffc_logger = logging.getLogger('FFC')
ffc_logger.setLevel(logging.WARNING)
ufl_logger = logging.getLogger('UFL')
ufl_logger.setLevel(logging.WARNING)

# Set path where results are saved.
resultpath = 'results/{0}'.format(
        datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
if not os.path.exists(resultpath):
    os.makedirs(resultpath)

print('Processing {0}'.format(datapath))


def load_kymo(datfolder, dat):
    # Identify Kymograph and do sanity check.
    kymos = glob.glob('{0}/SUM_Reslice of {1}*.tif'.format(datfolder, dat))
    if len(kymos) != 1:
        warnings.warn("No Kymograph found!")

    # Extract name of kymograph and replace whitespaces.
    name = os.path.splitext(os.path.basename(kymos[0]))[0]
    name = re.sub(' ', '_', name)

    # Load kymograph.
    return imageio.imread(kymos[0]), name


def prepareimage(img: np.array) -> np.array:
    # Remove cut.
    img = np.vstack((img[0:5, :], img[4, :], img[6:, :]))

    # Filter image.
    img = ndimage.gaussian_filter(img, sigma=1.0)

    # Normalise to [0, 1].
    img = np.array(img, dtype=float)
    img = (img - img.min()) / (img.max() - img.min())
    return img


# Figure 1: output frames for one dataset.
gen = 'SqAX3_SqhGFP42_GAP43_TM6B'
dat = 'E2PSB1'
frames = [4, 6, 10, 20, 40, 60, 80, 90]

# Output frames.
datfolder = os.path.join(datapath, os.path.join(gen, dat))
seq = glob.glob('{0}/{1}*.tif'.format(datfolder, dat))
if len(seq) != 1:
    warnings.warn("No sequence found!")
img = tiff.imread(seq)

# Output each frame.
for k in frames:
    if len(img.shape) is 4:
        frame = img[0, k]
    else:
        frame = img[k]
    filepath = os.path.join(os.path.join(resultpath, gen), dat)
    ph.saveimage_nolegend(filepath, '{0}-{1}'.format(dat, k), frame)


# Figure 2: load and output kymograph.
img, name = load_kymo(datfolder, dat)

# Plot and save figures.
ph.saveimage(os.path.join(*[resultpath, gen, dat]), name, img)

# Figure 7: save kymograph with tracks and spline fits.
gen = 'SqAX3_SqhGFP42_GAP43_TM6B'
# dat = '190216E8PSB1'
dat = '190216E5PSB2'

# Load kymograph.
datfolder = os.path.join(datapath, os.path.join(gen, dat))
img, name = load_kymo(datfolder, dat)

# Prepare image.
imgp = prepareimage(img)

# Sanity check.
roifile = 'manual_ROIs.zip'
if roifile not in os.listdir(datfolder):
    print("No ROI file found for {0}!".format(dat))

# Load roi zip.
roi = read_roi_zip(os.path.join(datfolder, roifile))

# Fit splines.
spl = rh.roi2splines(roi)

# Save kymo, tracks, and spline fit.
ph.saveroi(os.path.join(*[resultpath, gen, dat]), name, img, roi)
ph.savespl(os.path.join(*[resultpath, gen, dat]), name, img, roi, spl)
