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
import matplotlib.pyplot as plt
from matplotlib import cm

# Set font style.
font = {'family': 'sans-serif',
        'serif': ['DejaVu Sans'],
        'weight': 'normal',
        'size': 20}
plt.rc('font', **font)
plt.rc('text', usetex=True)

# Set colormap.
cmap = cm.viridis

# Streamlines.
density = 2
linewidth = 2
arrowstyle = '-'

# Set output quality.
dpi = 100

# Set path where results are saved.
resultpath = 'results/2018-08-17-09-40-25/'

# Flag whether to compute endpoint errors.
eval_endpoint = False


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
    error, curves = rh.compute_endpoint_error(vel, roi, spl)
    totalerr = 0
    maxerror = 0
    for v in roi:
        err = sum(error[v]) / len(error[v])
        totalerr += err
        maxerror = max(maxerror, max(error[v]))
    return (totalerr, maxerror, curves)


# Load dataset.
print('Loading results from {0}.'.format(resultpath))
with open(os.path.join(resultpath, 'pkl', 'name.pkl'), 'rb') as f:
    name = pickle.load(f)

with open(os.path.join(resultpath, 'pkl', 'prod_of1d.pkl'), 'rb') as f:
    prod_of1d = pickle.load(f)
with open(os.path.join(resultpath, 'pkl', 'prod_cms1dl2.pkl'), 'rb') as f:
    prod_cms1dl2 = pickle.load(f)
with open(os.path.join(resultpath, 'pkl', 'prod_cms1d.pkl'), 'rb') as f:
    prod_cms1d = pickle.load(f)
with open(os.path.join(resultpath, 'pkl', 'prod_cmscr1d.pkl'), 'rb') as f:
    prod_cmscr1d = pickle.load(f)

with open(os.path.join(resultpath, 'pkl', 'converged_cmscr1d.pkl'), 'rb') as f:
        converged_cmscr1d = pickle.load(f)


# Check if error evaluation is present, otherwise compute.
def load_error(model: str):
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
        print('No error file for {0}.'.format(model))
    return err, max_err


# Load or compute errors.
err_of1d, max_err_of1d = load_error('of1d')
err_cms1dl2, max_err_cms1dl2 = load_error('cms1dl2')
err_cms1d, max_err_cms1d = load_error('cms1d')
err_cmscr1d, max_err_cmscr1d = load_error('cmscr1d')


def plot_error(path: str, filename: str, err: tuple, title=None):
    """Takes a path string, a filename, and a tuple with errors, and saves the
    plotted array.

    Args:
        path (str): The path to save the image to.
        name (str): The filename.
        err (tuple): A tuple with dicts.
        title(str): An optional title.

    Returns:
    """
    if not os.path.exists(path):
        os.makedirs(path)

    # Plot image.
    fig, ax = plt.subplots(figsize=(10, 5))
    for gen in sorted(name.keys()):
        for dat in sorted(name[gen].keys()):
            vec = np.array([x[gen][dat] for x in err])
            vec[vec > 1] = 1
            plt.plot(vec, linewidth=1)

    if title is not None:
        ax.set_title(title)

    # Set axis limits.
    ax.set_ylim((0, 1))

    # Save figure.
    fig.savefig(os.path.join(path, '{0}.png'.format(filename)),
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)


plot_error(resultpath, 'of1d-avg_error', err_of1d,
           'Average error OF-H1.')
plot_error(resultpath, 'cms1dl2-avg_error', err_cms1dl2,
           'Average error CMS-H1-L2.')
plot_error(resultpath, 'cms1d-avg_error', err_cms1d,
           'Average error CMS-H1-H1.')
plot_error(resultpath, 'cmscr1d-avg_error', err_cmscr1d,
           'Average error CMS-H1-H1-CR.')

plot_error(resultpath, 'of1d-max_error', max_err_of1d,
           'Max. error OF-H1.')
plot_error(resultpath, 'cms1dl2-max_error', max_err_cms1dl2,
           'Max. error CMS-H1-L2.')
plot_error(resultpath, 'cms1d-max_error', max_err_cms1d,
           'Max. error CMS-H1-H1.')
plot_error(resultpath, 'cmscr1d-max_error', max_err_cmscr1d,
           'Max. error CMS-H1-H1-CR.')
print("Done.")
