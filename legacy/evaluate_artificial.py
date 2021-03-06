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
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats as stats
from dolfin import UserExpression
from dolfin import FunctionSpace
from dolfin import interpolate
from dolfin import UnitIntervalMesh
from matplotlib import cm
from ofmc.model.cm import cm1d_img
from ofmc.model.cms import cms1d_img
from ofmc.model.cmscr import cmscr1d_img
from ofmc.model.cmscr import cmscr1dnewton
from ofmc.model.of import of1d_img
from ofmc.util.transport import transport1d
from ofmc.util.velocity import velocity
from ofmc.util.dolfinhelpers import funvec2img


def saveimage(path: str, name: str, img: np.array):
    if not os.path.exists(path):
        os.makedirs(path)

    # Plot image.
    fig, ax = plt.subplots(figsize=(10, 5))
    cax = ax.imshow(img, cmap=cm.gray)
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

    maxvel = abs(vel).max()
    normi = mpl.colors.Normalize(vmin=-maxvel, vmax=maxvel)

    # Plot velocity.
    fig, ax = plt.subplots(figsize=(10, 5))
    cax = ax.imshow(vel, interpolation='nearest', norm=normi, cmap=cm.coolwarm)
    ax.set_title('Velocity')
    fig.colorbar(cax, orientation='horizontal')

    # Save figure.
    fig.savefig(os.path.join(path, '{0}-velocity.png'.format(name)))
    plt.close(fig)

    m, n = vel.shape
    hx, hy = 1./(m-1), 1./(n-1)

    # Create grid for streamlines.
    Y, X = np.mgrid[0:m, 0:n]
    V = np.ones_like(X)*hy

    # Plot streamlines.
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.imshow(img, cmap=cm.gray)
    strm = ax.streamplot(X, Y, vel*hx, V, density=2,
                         color=vel, linewidth=1, norm=normi, cmap=cm.coolwarm)
    fig.colorbar(strm.lines, orientation='horizontal')

    fig.savefig(os.path.join(path, '{0}-streamlines.png'.format(name)))
    plt.close(fig)

    # Save velocity profile after cut.
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.plot(vel[5])
    ax.set_title('Velocity profile right after the cut')

    fig.savefig(os.path.join(path, '{0}-profile.png'.format(name)))
    plt.close(fig)


def savestrainrate(path: str, name: str, sr: np.array):
    if not os.path.exists(path):
        os.makedirs(path)

    maxsr = abs(sr).max()
    normi = mpl.colors.Normalize(vmin=-maxsr, vmax=maxsr)

    # Plot velocity.
    fig, ax = plt.subplots(figsize=(10, 5))
    cax = ax.imshow(sr, interpolation='nearest', norm=normi, cmap=cm.coolwarm)
    ax.set_title('Strain rate')
    fig.colorbar(cax, orientation='horizontal')

    # Save figure.
    fig.savefig(os.path.join(path, '{0}-strainrate.png'.format(name)))
    plt.close(fig)


def saveerror(path: str, name: str, err: np.array):
    if not os.path.exists(path):
        os.makedirs(path)

    maxerr = abs(err).max()
    normi = mpl.colors.Normalize(vmin=-maxerr, vmax=maxerr)

    # Plot velocity.
    fig, ax = plt.subplots(figsize=(10, 5))
    cax = ax.imshow(err, interpolation='nearest', norm=normi, cmap=cm.coolwarm)
    ax.set_title('Error in velocity')
    fig.colorbar(cax, orientation='horizontal')

    # Save figure.
    fig.savefig(os.path.join(path, '{0}-error-velocity.png'.format(name)))
    plt.close(fig)


# Set path where results are saved.
resultpath = 'results'

# Set name of experiment.
name = 'artificial'

# Define temporal and spacial number of cells.
m, n = 40, 200

# Define parameters of artificial velocity field.
c0 = 0.05
v0 = 0.5
tau0 = 0.5
tau1 = 0.5

# Create artificial velocity.
v = velocity(m, n, c0, v0, tau0, tau1)

# Convert to array.
v = funvec2img(v.vector().get_local(), m, n)

# Define mesh.
mesh = UnitIntervalMesh(n - 1)

# Define function spaces
V = FunctionSpace(mesh, 'CG', 1)


class DoubleHat(UserExpression):

    def eval(self, value, x):
        value[0] = max(0, 0.1 - abs(x[0] - 0.4)) \
            + max(0, 0.1 - abs(x[0] - 0.6))

    def value_shape(self):
        return ()


class InitialData(UserExpression):

    def eval(self, value, x):
        value[0] = stats.uniform.pdf(x[0], 0, 1) \
            + (1 + math.sin(40 * x[0] + math.cos(40 * x[0]))) / 10

    def value_shape(self):
        return ()


f0 = DoubleHat(degree=1)
# f0 = InitialData(degree=1)
f0 = interpolate(f0, V)

# Compute transport
f = transport1d(v, np.zeros_like(v), f0.vector().get_local())

# Normalise to [0, 1].
f = np.array(f, dtype=float)
f = (f - f.min()) / (f.max() - f.min())

# Add some noise.
# f = f + 0.05 * np.random.randn(m + 1, n)


# Check continuity equation for rho.
def flux(v, f):
    return f * v


fv = np.array([flux(a, b) for a, b in zip(v, f)])
dtf = np.diff(f, n=1, axis=0)

err = np.abs(dtf[:, :-1] + np.diff(fv, n=1, axis=1))
fig, ax = plt.subplots(figsize=(10, 5))
cax = ax.imshow(err, cmap=cm.viridis)
ax.set_title('Error')
fig.colorbar(cax, orientation='vertical')


# Set regularisation parameters.
alpha0 = 1e-1
alpha1 = 1e-2
alpha2 = 1e3
alpha3 = 1e3
beta = 1e-10

# Compute velocities.
# vel = of1d(f, alpha0, alpha1)
# vel = cm1d(f, alpha0, alpha1)
# vel, k = cms1d(f, alpha0, alpha1, alpha2, alpha3)
vel, k, res, fun, converged = cmscr1d_img(f, alpha0, alpha1, alpha2,
                                          alpha3, beta, 'mesh')
# vel, k, res, fun converged = cmscr1dnewton(f, alpha0, alpha1, alpha2,
# alpha3, beta)

# Plot and save figures.
saveimage(os.path.join(resultpath, 'artificial'), name, f)
savevelocity(os.path.join(resultpath, 'artificial'), name, f, vel)
savevelocity(os.path.join(resultpath, 'artificial'), 'artificial-gt', f, v)
savesource(os.path.join(resultpath, 'artificial'), name, k)
saveerror(os.path.join(resultpath, 'artificial'), name,
          np.abs(v - vel[:-1, :]))

# Compute differences.

# Compute and save strain rate.
m, n = f.shape
hy = 1.0 / (n - 1)
sr = np.gradient(vel, hy, axis=1)
savestrainrate(os.path.join(resultpath, 'artificial'), name, sr)
