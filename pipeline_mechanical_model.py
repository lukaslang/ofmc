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
import numpy as np
from matplotlib import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ofmc.model.cmscr import cmscr1d_img
from scipy.sparse import spdiags
import scipy.stats as stats
import sys
sys.path.append('../cuts-octave')
from timestepping import timestepping

# Set font style.
font = {'weight': 'normal',
        'size': 20}
plt.rc('font', **font)

# Set colormap.
cmap = cm.viridis

# Streamlines.
density = 2
linewidth = 2

# Set path where results are saved.
resultpath = 'results/{0}'.format(
        datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
if not os.path.exists(resultpath):
    os.makedirs(resultpath)


def savequantity(path: str, name: str, img: np.array, title: str):
    if not os.path.exists(path):
        os.makedirs(path)

    # Plot image.
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(img, cmap=cmap)
    ax.set_title(title)

    # Create colourbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # Save figure.
    fig.savefig(os.path.join(path, '{0}.png'.format(name)),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def saveimage(path: str, name: str, img: np.array):
    if not os.path.exists(path):
        os.makedirs(path)

    # Plot image.
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(img, cmap=cm.gray)
    ax.set_title('Fluorescence intensity')

    # Create colourbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # Save figure.
    fig.savefig(os.path.join(path, '{0}.png'.format(name)),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def savesource(path: str, name: str, img: np.array):
    if not os.path.exists(path):
        os.makedirs(path)

    # Plot image.
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(img, cmap=cmap)
    ax.set_title('Source')

    # Create colourbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # Save figure.
    fig.tight_layout()
    fig.savefig(os.path.join(path, '{0}-source.png'.format(name)),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def savevelocity(path: str, name: str, img: np.array, vel: np.array):
    if not os.path.exists(path):
        os.makedirs(path)

    maxvel = abs(vel).max()
    normi = mpl.colors.Normalize(vmin=-maxvel, vmax=maxvel)

    # Plot velocity.
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(vel, interpolation='nearest', norm=normi, cmap=cmap)
    ax.set_title('Velocity')

    # Create colourbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # Save figure.
    fig.tight_layout()
    fig.savefig(os.path.join(path, '{0}-velocity.png'.format(name)),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    m, n = vel.shape
    hx, hy = 1.0 / (m - 1), 1.0 / (n - 1)

    # Create grid for streamlines.
    Y, X = np.mgrid[0:m, 0:n]
    V = np.ones_like(X)

    # Plot streamlines.
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.imshow(img, cmap=cm.gray)
    ax.set_title('Streamlines')
    strm = ax.streamplot(X, Y, vel * hx / hy, V, density=density,
                         color=vel, linewidth=linewidth, norm=normi, cmap=cmap)
#    fig.colorbar(strm.lines, orientation='vertical')

    # Create colourbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(strm.lines, cax=cax, orientation='vertical')

    fig.tight_layout()
    fig.savefig(os.path.join(path, '{0}-streamlines.png'.format(name)),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Save velocity profile after cut.
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.plot(vel[5])
    ax.set_title('Velocity profile right after the cut')

    fig.tight_layout()
    fig.savefig(os.path.join(path, '{0}-profile.png'.format(name)),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def savestrainrate(path: str, name: str, img: np.array, vel: np.array):
    if not os.path.exists(path):
        os.makedirs(path)

    m, n = img.shape
    hy = 1.0 / (n - 1)
    sr = np.gradient(vel, hy, axis=1)

    maxsr = abs(sr).max()
    normi = mpl.colors.Normalize(vmin=-maxsr, vmax=maxsr)

    # Plot velocity.
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(sr, interpolation='nearest', norm=normi, cmap=cmap)
    ax.set_title('Strain rate')

    # Create colourbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # Save figure.
    fig.tight_layout()
    fig.savefig(os.path.join(path, '{0}-strainrate.png'.format(name)),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


# Settings.
artvel = False

# Run time stepping algorithm.
N, ii, TimeS, ca_sav, cd_sav, a_sav, v_sav = timestepping(artvel)

print('Done!\n')

# Plot results.
rng = range(ii)

# Set name.
name = 'mechanical_model'

resfolder = os.path.join(resultpath, 'mechanical_model')
if not os.path.exists(resfolder):
    os.makedirs(resfolder)

# Plot and save figures.
savequantity(resfolder, '{0}-a_sav'.format(name),
             a_sav[:, rng].transpose(), 'a_sav')
savequantity(resfolder, '{0}-ca_sav'.format(name),
             ca_sav[:, rng].transpose(), 'ca_sav')
savequantity(resfolder, '{0}-cd_sav'.format(name),
             cd_sav[:, rng].transpose(), 'cd_sav')
savequantity(resfolder, '{0}-ca_sav+a_sav'.format(name),
             (cd_sav[:, rng] + a_sav[:, rng]).transpose(), 'ca_sav + a_sav')
savequantity(resfolder, '{0}-v_sav'.format(name),
             v_sav[:, rng].transpose(), 'v_sav')

# Set regularisation parameter.
alpha0 = 5e0
alpha1 = 1e-1
alpha2 = 1e-2
alpha3 = 1e-2
beta = 5e-4

# Define concentration.
# img = (ca_sav[:, 11:ii] + a_sav[:, 11:ii]).transpose()
# img = ca_sav[:, 1:ii].transpose()
img = ca_sav[:, 11:ii].transpose()

# Compute velocity and source.
vel, k = cmscr1d_img(img, alpha0, alpha1, alpha2, alpha3,
                     beta, 'mesh')

# Plot and save figures.
saveimage(resfolder, name, img)
savevelocity(resfolder, name, img, vel)
savesource(resfolder, name, k)
savestrainrate(resfolder, name, img, vel)
