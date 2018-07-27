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
import matplotlib
matplotlib.use('agg')
import collections
import os
import re
import numpy as np
import ofmc.util.pyplothelpers as ph
import ofmc.util.roihelpers as rh
import pickle

# Set path where results are saved.
resultpath = 'results/2018-07-23-17-50-01/'


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
print('Loading results from {0}.'.format(resultpath))
with open(os.path.join(resultpath, 'pkl', 'genotypes.pkl'), 'rb') as f:
    genotypes = pickle.load(f)
with open(os.path.join(resultpath, 'pkl', 'datasets.pkl'), 'rb') as f:
    datasets = pickle.load(f)
with open(os.path.join(resultpath, 'pkl', 'name.pkl'), 'rb') as f:
    name = pickle.load(f)
with open(os.path.join(resultpath, 'pkl', 'img.pkl'), 'rb') as f:
    img = pickle.load(f)
with open(os.path.join(resultpath, 'pkl', 'imgp.pkl'), 'rb') as f:
    imgp = pickle.load(f)
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
def compute_error(idx: int, count: int, vel: dict):
    err = collections.defaultdict(dict)
    max_err = collections.defaultdict(dict)
    print("Result {0}/{1}".format(idx + 1, len(vel_of1d)))
    # Run through datasets.
    for gen in genotypes:
        for dat in datasets[gen]:
            print("Computing error for {0}/{1}".format(gen, dat))
            err[gen][dat], max_err[gen][dat] = \
                error(vel[idx][gen][dat], roi[gen][dat], spl[gen][dat])
    return err, max_err


# Check if error evaluation is present, otherwise compute.
def load_or_compute_error(model: str, vel: dict):
    err_file = os.path.join(resultpath, 'pkl', 'err_{0}.pkl'.format(model))
    max_err_file = os.path.join(resultpath,
                                'pkl', 'max_err_{0}.pkl'.format(model))
    if os.path.isfile(err_file) and \
            os.path.isfile(max_err_file):
        print('Loading error for {0}.'.format(model))
        # Load existing results.
        with open(err_file, 'rb') as f:
            err = pickle.load(f)
        with open(max_err_file, 'rb') as f:
            max_err = pickle.load(f)
    else:
        print('Computing error for {0}.'.format(model))
        num = len(vel)
        results = [compute_error(idx, num, vel) for idx in range(num)]
        err, max_err = zip(*results)
        # Store results.
        with open(err_file, 'wb') as f:
            pickle.dump(err, f, pickle.HIGHEST_PROTOCOL)
        with open(max_err_file, 'wb') as f:
            pickle.dump(max_err, f, pickle.HIGHEST_PROTOCOL)
    return err, max_err


# Load or compute errors.
err_of1d, max_err_of1d = load_or_compute_error('of1d', vel_of1d)
err_cms1dl2, max_err_cms1dl2 = load_or_compute_error('cms1dl2', vel_cms1dl2)
err_cms1d, max_err_cms1d = load_or_compute_error('cms1d', vel_cms1d)
err_cmscr1d, max_err_cmscr1d = load_or_compute_error('cmscr1d', vel_cmscr1d)

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
        f.write("Dataset {0}/{1}\n".format(gen, dat))
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


# Output LaTeX table in sorted order.
f.write('LaTeX table with average error:\n')
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
f.write('\\hline\n')

# Output LaTeX table in sorted order.
f.write('LaTeX table with maximum error:\n')
for gen in sorted(name.keys()):
    for dat in sorted(name[gen].keys()):
        # Find indices of best results (not necessarily unique).
        idx_of1d = np.argmin([x[gen][dat] for x in max_err_of1d])
        idx_cms1dl2 = np.argmin([x[gen][dat] for x in max_err_cms1dl2])
        idx_cms1d = np.argmin([x[gen][dat] for x in max_err_cms1d])
        idx_cmscr1d = np.argmin([x[gen][dat] for x in max_err_cmscr1d])

        formatstr = '{0}/{1} & {2:.2f} & {3:.2f} & {4:.2f} & {5:.2f} \\\\\n'
        f.write(formatstr.format(re.sub('_', '\\_', gen),
                                 re.sub('_', '\\_', dat),
                                 max_err_of1d[idx_of1d][gen][dat],
                                 max_err_cms1dl2[idx_cms1dl2][gen][dat],
                                 max_err_cms1d[idx_cms1d][gen][dat],
                                 max_err_cmscr1d[idx_cmscr1d][gen][dat]))
f.write('\\hline\n')

# Output maximum over all datasets.
max_of1d = -np.inf
max_cms1dl2 = -np.inf
max_cms1d = -np.inf
max_cmscr1d = -np.inf
for gen in sorted(name.keys()):
    for dat in sorted(name[gen].keys()):
        # Find indices of best results (not necessarily unique).
        idx_of1d = np.argmin([x[gen][dat] for x in max_err_of1d])
        idx_cms1dl2 = np.argmin([x[gen][dat] for x in max_err_cms1dl2])
        idx_cms1d = np.argmin([x[gen][dat] for x in max_err_cms1d])
        idx_cmscr1d = np.argmin([x[gen][dat] for x in max_err_cmscr1d])

        max_of1d = max(max_of1d, max_err_of1d[idx_of1d][gen][dat])
        max_cms1dl2 = max(max_cms1dl2, max_err_cms1dl2[idx_cms1dl2][gen][dat])
        max_cms1d = max(max_cms1d, max_err_cms1d[idx_cms1d][gen][dat])

        if converged_cmscr1d[idx_cmscr1d][gen][dat]:
            max_cmscr1d = max(max_cmscr1d,
                              max_err_cmscr1d[idx_cmscr1d][gen][dat])

formatstr = 'Maximum & {0:.2f} & {1:.2f} & {2:.2f} & {3:.2f} \\\\\n'
f.write(formatstr.format(sum_of1d / count,
                         sum_cms1dl2 / count,
                         sum_cms1d / count,
                         sum_cmscr1d / count_converged))
f.write('\\hline\n')

# Close file.
f.close()

# Output datasets.
print("Plotting datasets.")
for gen in genotypes:
    for dat in datasets[gen]:
        print("Plotting dataset {0}/{1}".format(gen, dat))

        tmpimg = img[gen][dat]
        tmpimgp = imgp[gen][dat]
        tmpname = name[gen][dat]
        tmproi = roi[gen][dat]
        tmpspl = spl[gen][dat]

        # Save images.
        tmpfolder = [resultpath, gen, dat]
        resfolder = os.path.join(*tmpfolder)
        ph.saveimage(resfolder, tmpname, tmpimg)
        ph.saveimage(resfolder, '{0}-filtered'.format(tmpname), tmpimgp)

        # Save manual tracks.
        ph.saveroi(resfolder, tmpname, tmpimg, tmproi)
        ph.savespl(resfolder, tmpname, tmpimg, tmproi, tmpspl)


# Output best result for each dataset.
def output_best_result(err_name: str,
                       model: str, err: list, vel: dict, k=None):
    print("Plotting {0} results for {1}".format(err_name, model))
    for gen in genotypes:
        for dat in datasets[gen]:
            print("Plotting results for {0}/{1}".format(gen, dat))

            # Find index of best results (not necessarily unique).
            idx = np.argmin([x[gen][dat] for x in err])

            # Get data.
            tmpimg = img[gen][dat]
            tmpname = name[gen][dat]
            tmproi = roi[gen][dat]
            tmpspl = spl[gen][dat]

            # of1d
            tmpfolder = [resultpath, model, err_name, gen, dat]
            resfolder = os.path.join(*tmpfolder)
            if not os.path.exists(resfolder):
                os.makedirs(resfolder)
            tmpvel = vel[idx][gen][dat]
            ph.savevelocity(resfolder, tmpname, tmpimg, tmpvel)
            if k is not None:
                ph.savesource(resfolder, tmpname, k[idx][gen][dat])
            ph.saveerror(resfolder, tmpname, tmpimg, tmpvel, tmproi, tmpspl)
            ph.save_spl_streamlines(resfolder, tmpname, tmpimg,
                                    tmpvel, tmproi, tmpspl)
            ph.save_roi_streamlines(resfolder, tmpname, tmpimg, tmpvel, tmproi)


# Output best results for average error.
output_best_result('best_avg_error', 'of1d', err_of1d, vel_of1d)
output_best_result('best_avg_error', 'cms1dl2',
                   err_cms1dl2, vel_cms1dl2, k_cms1dl2)
output_best_result('best_avg_error', 'cms1d',
                   err_cms1d, vel_cms1d, k_cms1d)
output_best_result('best_avg_error', 'cmscr1d',
                   err_cmscr1d, vel_cmscr1d, k_cmscr1d)

# Output best results for maximum error.
output_best_result('best_max_error', 'of1d', max_err_of1d, vel_of1d)
output_best_result('best_max_error', 'cms1dl2',
                   max_err_cms1dl2, vel_cms1dl2, k_cms1dl2)
output_best_result('best_max_error', 'cms1d',
                   max_err_cms1d, vel_cms1d, k_cms1d)
output_best_result('best_max_error', 'cmscr1d',
                   max_err_cmscr1d, vel_cmscr1d, k_cmscr1d)

print("Done.")
