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
import re
import numpy as np
import ofmc.util.pyplothelpers as ph
import pickle

# Set path where results are saved.
resultpath = 'results/2018-07-19-17-19-13/'

# Load dataset.
with open(os.path.join(resultpath, 'pkl', 'name.pkl'), 'rb') as f:
        name = pickle.load(f)
with open(os.path.join(resultpath, 'pkl', 'img.pkl'), 'rb') as f:
        img = pickle.load(f)
with open(os.path.join(resultpath, 'pkl', 'roi.pkl'), 'rb') as f:
        roi = pickle.load(f)
with open(os.path.join(resultpath, 'pkl', 'spl.pkl'), 'rb') as f:
        spl = pickle.load(f)

with open(os.path.join(resultpath, 'pkl', 'err_of1d.pkl'), 'rb') as f:
        err_of1d = pickle.load(f)
with open(os.path.join(resultpath, 'pkl', 'err_cms1dl2.pkl'), 'rb') as f:
        err_cms1dl2 = pickle.load(f)
with open(os.path.join(resultpath, 'pkl', 'err_cms1d.pkl'), 'rb') as f:
        err_cms1d = pickle.load(f)
with open(os.path.join(resultpath, 'pkl', 'err_cmscr1d.pkl'), 'rb') as f:
        err_cmscr1d = pickle.load(f)

with open(os.path.join(resultpath, 'pkl', 'max_err_of1d.pkl'), 'rb') as f:
        max_err_of1d = pickle.load(f)
with open(os.path.join(resultpath, 'pkl', 'max_err_cms1dl2.pkl'), 'rb') as f:
        max_err_cms1dl2 = pickle.load(f)
with open(os.path.join(resultpath, 'pkl', 'max_err_cms1d.pkl'), 'rb') as f:
        max_err_cms1d = pickle.load(f)
with open(os.path.join(resultpath, 'pkl', 'max_err_cmscr1d.pkl'), 'rb') as f:
        max_err_cmscr1d = pickle.load(f)

with open(os.path.join(resultpath, 'pkl', 'vel_of1d.pkl'), 'rb') as f:
        vel_of1d = pickle.load(f)
with open(os.path.join(resultpath, 'pkl', 'vel_cms1dl2.pkl'), 'rb') as f:
        vel_cms1dl2 = pickle.load(f)
with open(os.path.join(resultpath, 'pkl', 'vel_cms1d.pkl'), 'rb') as f:
        vel_cms1d = pickle.load(f)
with open(os.path.join(resultpath, 'pkl', 'vel_cmscr1d.pkl'), 'rb') as f:
        vel_cmscr1d = pickle.load(f)

with open(os.path.join(resultpath, 'pkl', 'k_cms1dl2.pkl'), 'rb') as f:
        k_cms1dl2 = pickle.load(f)
with open(os.path.join(resultpath, 'pkl', 'k_cms1d.pkl'), 'rb') as f:
        k_cms1d = pickle.load(f)
with open(os.path.join(resultpath, 'pkl', 'k_cmscr1d.pkl'), 'rb') as f:
        k_cmscr1d = pickle.load(f)

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
        err = [x[gen][dat] for x in max_err_of1d]
        print("of1d:    " + ", ".join('{0:.3f}'.format(x) for x in err))
        err = [x[gen][dat] for x in max_err_cms1dl2]
        print("cms1dl2: " + ", ".join('{0:.3f}'.format(x) for x in err))
        err = [x[gen][dat] for x in max_err_cms1d]
        print("cms1d:   " + ", ".join('{0:.3f}'.format(x) for x in err))
        err = [x[gen][dat] for x in max_err_cmscr1d]
        print("cmscr1d: " + ", ".join('{0:.3f}'.format(x) for x in err))


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
# Output LaTeX table in sorted order.
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
