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
import re
import numpy as np
import ofmc.util.pyplothelpers as ph
import ofmc.util.roihelpers as rh
import pickle

# Set path where results are saved.
resultpath = 'results/2018-07-23-14-14-11/'


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


def endpoint_error(vel, roi, spl) -> (float, float):
    # Compute accumulated error in position for each spline.
    error, curve = rh.compute_endpoint_error(vel, roi, spl)
    totalerr = 0
    maxerror = 0
    for v in roi:
        err = sum(error[v]) / len(error[v])
        totalerr += err
        maxerror = max(maxerror, max(error[v]))
    return (totalerr, maxerror)


# Load dataset.
with open(os.path.join(resultpath, 'pkl', 'genotypes.pkl'), 'rb') as f:
        genotypes = pickle.load(f)
with open(os.path.join(resultpath, 'pkl', 'datasets.pkl'), 'rb') as f:
        datasets = pickle.load(f)
with open(os.path.join(resultpath, 'pkl', 'name.pkl'), 'rb') as f:
        name = pickle.load(f)
with open(os.path.join(resultpath, 'pkl', 'img.pkl'), 'rb') as f:
        img = pickle.load(f)
with open(os.path.join(resultpath, 'pkl', 'roi.pkl'), 'rb') as f:
        roi = pickle.load(f)
with open(os.path.join(resultpath, 'pkl', 'spl.pkl'), 'rb') as f:
        spl = pickle.load(f)

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

with open(os.path.join(resultpath, 'pkl', 'converged_cmscr1d.pkl'), 'rb') as f:
        converged_cmscr1d = pickle.load(f)

# Compute errors.
print('Computing error for of1d.')
err_of1d = [collections.defaultdict(dict) for x in range(len(vel_of1d))]
max_err_of1d = [collections.defaultdict(dict) for x in range(len(vel_of1d))]
for idx in range(len(vel_of1d)):
    print("{0}/{1}".format(idx, len(vel_of1d)))
    # Run through datasets.
    for gen in genotypes:
        for dat in datasets[gen]:
            err_of1d[idx][gen][dat], max_err_of1d[idx][gen][dat] = \
                error(vel_of1d[idx][gen][dat], roi[gen][dat], spl[gen][dat])

print('Computing error for cms1dl2.')
err_cms1dl2 = [collections.defaultdict(dict) for x in range(len(vel_cms1dl2))]
max_err_cms1dl2 = [collections.defaultdict(dict)
                   for x in range(len(vel_cms1dl2))]
for idx in range(len(vel_cms1dl2)):
    print("{0}/{1}".format(idx, len(vel_cms1dl2)))
    # Run through datasets.
    for gen in genotypes:
        for dat in datasets[gen]:
            err_cms1dl2[idx][gen][dat], max_err_cms1dl2[idx][gen][dat] = \
                error(vel_cms1dl2[idx][gen][dat], roi[gen][dat], spl[gen][dat])

print('Computing error for cms1d.')
err_cms1d = [collections.defaultdict(dict) for x in range(len(vel_cms1d))]
max_err_cms1d = [collections.defaultdict(dict) for x in range(len(vel_cms1d))]
for idx in range(len(vel_cms1d)):
    print("{0}/{1}".format(idx, len(vel_cms1d)))
    # Run through datasets.
    for gen in genotypes:
        for dat in datasets[gen]:
            err_cms1d[idx][gen][dat], max_err_cms1d[idx][gen][dat] = \
                error(vel_cms1d[idx][gen][dat], roi[gen][dat], spl[gen][dat])

print('Computing error for cmscr1d.')
err_cmscr1d = [collections.defaultdict(dict) for x in range(len(vel_cmscr1d))]
max_err_cmscr1d = [collections.defaultdict(dict)
                   for x in range(len(vel_cmscr1d))]
for idx in range(len(vel_cmscr1d)):
    print("{0}/{1}".format(idx, len(vel_cmscr1d)))
    # Run through datasets.
    for gen in genotypes:
        for dat in datasets[gen]:
            if converged_cmscr1d[idx][gen][dat]:
                err_cmscr1d[idx][gen][dat], max_err_cmscr1d[idx][gen][dat] = \
                    error(vel_cmscr1d[idx][gen][dat],
                          roi[gen][dat], spl[gen][dat])
            else:
                err_cmscr1d[idx][gen][dat] = np.inf
                max_err_cmscr1d[idx][gen][dat] = np.inf

# Open file.
f = open(os.path.join(resultpath, 'results.txt'), 'w')

# Print errors.
f.write('Cumulative absolute error:\n')
for gen in genotypes:
    for dat in datasets[gen]:
        f.write("Dataset {0}/{1}\n".format(gen, dat))
        err = [x[gen][dat] for x in err_of1d]
        f.write("of1d:    " +
                ", ".join('{0:.3f}'.format(x) for x in err) + "\n")
        err = [x[gen][dat] for x in err_cms1dl2]
        f.write("cms1dl2: " +
                ", ".join('{0:.3f}'.format(x) for x in err) + "\n")
        err = [x[gen][dat] for x in err_cms1d]
        f.write("cms1d:   " +
                ", ".join('{0:.3f}'.format(x) for x in err) + "\n")
        err = [x[gen][dat] for x in err_cmscr1d]
        f.write("cmscr1d: " +
                ", ".join('{0:.3f}'.format(x) for x in err) + "\n")

# Print max. errors.
f.write('Maximum absolute error:\n')
for gen in genotypes:
    for dat in datasets[gen]:
        f.write("Dataset {0}/{1}".format(gen, dat))
        err = [x[gen][dat] for x in max_err_of1d]
        f.write("of1d:    " +
                ", ".join('{0:.3f}'.format(x) for x in err) + "\n")
        err = [x[gen][dat] for x in max_err_cms1dl2]
        f.write("cms1dl2: " +
                ", ".join('{0:.3f}'.format(x) for x in err) + "\n")
        err = [x[gen][dat] for x in max_err_cms1d]
        f.write("cms1d:   " +
                ", ".join('{0:.3f}'.format(x) for x in err) + "\n")
        err = [x[gen][dat] for x in max_err_cmscr1d]
        f.write("cmscr1d: " +
                ", ".join('{0:.3f}'.format(x) for x in err) + "\n")


# Output LaTeX table in sorte order.
f.write('LaTeX table with results:\n')
for gen in sorted(name.keys()):
    for dat in sorted(name[gen].keys()):
        # Find indices of best results (not necessarily unique).
        idx_of1d = np.argmin([x[gen][dat] for x in err_of1d])
        idx_cms1dl2 = np.argmin([x[gen][dat] for x in err_cms1dl2])
        idx_cms1d = np.argmin([x[gen][dat] for x in err_cms1d])
        idx_cmscr1d = np.argmin([x[gen][dat] for x in err_cmscr1d])

        formatstr = '{0}/{1} & {2:.2f} & {3:.2f} & {4:.2f} & {5:.2f} \\\\\n'
        f.write(formatstr.format(re.sub('_', '\\_', gen),
                                 re.sub('_', '\\_', dat),
                                 err_of1d[idx_of1d][gen][dat],
                                 err_cms1dl2[idx_cms1dl2][gen][dat],
                                 err_cms1d[idx_cms1d][gen][dat],
                                 err_cmscr1d[idx_cmscr1d][gen][dat]))
f.write('\\hline\n')

# Output average over all datasets.
# Output LaTeX table in sorted order.
sum_of1d = 0.0
sum_cms1dl2 = 0.0
sum_cms1d = 0.0
sum_cmscr1d = 0.0
count = 0.0
count_converged = 0.0
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

        if converged_cmscr1d[idx_cmscr1d][gen][dat]:
            sum_cmscr1d += err_cmscr1d[idx_cmscr1d][gen][dat]
            count_converged += 1.0

        count += 1.0

formatstr = 'Average & {0:.2f} & {1:.2f} & {2:.2f} & {3:.2f} \\\\\n'
f.write(formatstr.format(sum_of1d / count,
                         sum_cms1dl2 / count,
                         sum_cms1d / count,
                         sum_cmscr1d / count_converged))

# Close file.
f.close()

# Output best result for each method and each dataset.
for gen in genotypes:
    for dat in datasets[gen]:
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

print("Done.")
