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
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import cm
# from ofmc.model.cm import cm1d
# from ofmc.model.cms import cms1d
from ofmc.model.cmscr import cmscr1d
from ofmc.model.cmscr import cmscr1dnewton

# Set path where results are saved.
resultpath = os.path.join('results', 'analytic_example')
if not os.path.exists(resultpath):
    os.makedirs(resultpath)

# Set regularisation parameter.
alpha0 = 1e1
alpha1 = 1e-1
alpha2 = 1e-3
alpha3 = 1e-3
beta = 1e-6


def saveimage(path: str, name: str, img: np.array):
    if not os.path.exists(path):
        os.makedirs(path)

    # Plot image.
    fig, ax = plt.subplots(figsize=(10, 5))
    cax = ax.imshow(img, cmap=cm.coolwarm)
    ax.set_title('Density')
    fig.colorbar(cax, orientation='horizontal')

    # Save figure.
    fig.savefig(os.path.join(path, '{0}.png'.format(name)))
    plt.close(fig)


def savesource(path: str, name: str, img: np.array):
    if not os.path.exists(path):
        os.makedirs(path)

    # Plot image.
    fig, ax = plt.subplots(figsize=(10, 5))
    cax = ax.imshow(img, cmap=cm.coolwarm)
    ax.set_title('Source')
    fig.colorbar(cax, orientation='horizontal')

    # Save figure.
    fig.savefig(os.path.join(path, '{0}-source.png'.format(name)))
    plt.close(fig)


def savevelocity(path: str, name: str, img: np.array, vel: np.array):
    if not os.path.exists(path):
        os.makedirs(path)

    #maxvel = abs(vel).max()
    #normi = mpl.colors.Normalize(vmin=-maxvel, vmax=maxvel)

    # Plot velocity.
    fig, ax = plt.subplots(figsize=(10, 5))
    #cax = ax.imshow(vel, interpolation='nearest', norm=normi, cmap=cm.coolwarm)
    cax = ax.imshow(vel, interpolation='nearest', cmap=cm.coolwarm)
    ax.set_title('Velocity')
    fig.colorbar(cax, orientation='horizontal')

    # Save figure.
    fig.savefig(os.path.join(path, '{0}-velocity.png'.format(name)))
    plt.close(fig)

    m, n = vel.shape
    hx, hy = 1.0/(m-1), 1.0/(n-1)

    # Create grid for streamlines.
    Y, X = np.mgrid[0:m, 0:n]
    V = np.ones_like(X)*hy

    # Plot streamlines.
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.imshow(img, cmap=cm.gray)
    strm = ax.streamplot(X, Y, vel*hx, V, density=2,
#                         color=vel, linewidth=1, norm=normi, cmap=cm.coolwarm)
                         color=vel, linewidth=1, cmap=cm.coolwarm)
    fig.colorbar(strm.lines, orientation='horizontal')

    fig.savefig(os.path.join(path, '{0}-streamlines.png'.format(name)))
    plt.close(fig)

    # Save velocity profile after cut.
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.plot(vel[0])
    ax.set_title('Velocity profile at time zero')

    fig.savefig(os.path.join(path, '{0}-profile.png'.format(name)))
    plt.close(fig)


def saveerror(path: str, name: str, img: np.array, k: np.array):
    if not os.path.exists(path):
        os.makedirs(path)

    # Plot image.
    fig, ax = plt.subplots(figsize=(10, 5))
    cax = ax.imshow(np.abs(-img - k), cmap=cm.coolwarm)
    ax.set_title('abs(-c - k)')
    fig.colorbar(cax, orientation='horizontal')

    # Save figure.
    fig.savefig(os.path.join(path, '{0}-error.png'.format(name)))
    plt.close(fig)


def createimage(m: int, n: int) -> np.array:
    v = 1.0
    ell = 5.0
    tau = 1.0

    def f(t, x):
        return np.exp(-t/tau)*np.cos((x - v*t)/ell)
        #return np.cos((x - v*t)/ell)

    x, t = np.meshgrid(np.linspace(0, n, n), np.linspace(0, m, m))

    img = f(t, x)
    img = (img - img.min()) / (img.max() - img.min())
    return img


# Set name.
name = 'analytic_example'

# Create artificial image.
img = createimage(5, 100)

# Compute velocities.
# vel = of1d(img, alpha0, alpha1)
# vel = cm1d(img, alpha0, alpha1)
# vel, k = cms1d(img, alpha0, alpha1, alpha2, alpha3)
vel, k = cmscr1d(img, alpha0, alpha1, alpha2, alpha3, beta)
#vel, k = cmscr1dnewton(img, alpha0, alpha1, alpha2, alpha3, beta)

# Plot and save figures.
saveimage(resultpath, name, img)
savevelocity(resultpath, name, img, vel)
savesource(resultpath, name, k)
saveerror(resultpath, name, img, k)
