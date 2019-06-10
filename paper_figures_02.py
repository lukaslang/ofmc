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
# Figure 4: analytic example showing an error in estimated velocity when no
# source is estimated.
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
from dolfin import Expression
from dolfin import interpolate
from dolfin import UnitSquareMesh
from matplotlib import cm
from ofmc.model.cmscr import cmscr1d_exp_pb
from ofmc.model.cm import cm1d_exp_pb
from mpl_toolkits.axes_grid1 import make_axes_locatable
import ofmc.util.dolfinhelpers as dh

# Set font style.
font = {'family': 'sans-serif',
        'serif': ['DejaVu Sans'],
        'weight': 'normal',
        'size': 30}
plt.rc('font', **font)
plt.rc('text', usetex=True)

# Set colormap.
cmap = cm.viridis

# Streamlines.
density = 1
linewidth = 2

# Set output quality.
dpi = 100

# Streamlines.
density = 1
linewidth = 1


def saveimage(path: str, name: str, img: np.array):
    if not os.path.exists(path):
        os.makedirs(path)

    # Plot image.
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(img, cmap=cm.gray)
    # ax.set_title('Concentration')

    # Create colourbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # Save figure.
    fig.savefig(os.path.join(path, '{0}.png'.format(name)),
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def savesource(path: str, name: str, img: np.array):
    if not os.path.exists(path):
        os.makedirs(path)

    # Plot image.
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(img, cmap=cm.gray)
    # ax.set_title('Source')

    # Create colourbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # Save figure.
    fig.tight_layout()
    fig.savefig(os.path.join(path, '{0}-source.png'.format(name)),
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def savevelocity(path: str, name: str, img: np.array, vel: np.array):
    if not os.path.exists(path):
        os.makedirs(path)

    # maxvel = abs(vel).max()
    # normi = mpl.colors.Normalize(vmin=-maxvel, vmax=maxvel)

    # Plot velocity.
    fig, ax = plt.subplots(figsize=(10, 5))
    # im = ax.imshow(vel, interpolation='nearest', norm=normi, cmap=cmap)
    im = ax.imshow(vel, interpolation='nearest', cmap=cm.gray)
    # ax.set_title('Velocity')

    # Create colourbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # Save figure.
    fig.tight_layout()
    fig.savefig(os.path.join(path, '{0}-velocity.png'.format(name)),
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    m, n = vel.shape
    hx, hy = 1.0 / (m - 1), 1.0 / (n - 1)

    # Create grid for streamlines.
    Y, X = np.mgrid[0:m, 0:n]
    V = np.ones_like(X)

    # Plot streamlines.
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.imshow(img, cmap=cm.gray)
    # ax.set_title('Streamlines')
    strm = ax.streamplot(X, Y, vel * hx / hy, V, density=density,
                         color=vel, linewidth=linewidth, cmap=cmap)
#                        color=vel, linewidth=linewidth, norm=normi, cmap=cmap)
#    fig.colorbar(strm.lines, orientation='vertical')

    # Create colourbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(strm.lines, cax=cax, orientation='vertical')

    fig.tight_layout()
    fig.savefig(os.path.join(path, '{0}-streamlines.png'.format(name)),
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    # Save velocity profile after cut.
    fig, ax = plt.subplots(figsize=(10, 5))
    t = np.linspace(0, m, 4, dtype=int, endpoint=False)
    t0, = plt.plot(vel[t[0]], label='t={0}'.format(t[0]), linewidth=linewidth)
    t1, = plt.plot(vel[t[1]], label='t={0}'.format(t[1]), linewidth=linewidth)
    t2, = plt.plot(vel[t[2]], label='t={0}'.format(t[2]), linewidth=linewidth)
    t3, = plt.plot(vel[t[3]], label='t={0}'.format(t[3]), linewidth=linewidth)
    plt.legend(handles=[t0, t1, t2, t3], bbox_to_anchor=(1, 1))
    # ax.set_title('Velocity profile at different times')

    fig.tight_layout()
    fig.savefig(os.path.join(path, '{0}-profile.png'.format(name)),
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def saveerror(path: str, name: str, k: np.array, kgt: np.array):
    if not os.path.exists(path):
        os.makedirs(path)

    # Plot image.
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(np.abs(k - kgt), cmap=cm.gray)
    # ax.set_title('Error in k')

    # Create colourbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # Save figure.
    fig.tight_layout()
    fig.savefig(os.path.join(path, '{0}-sourceerror.png'.format(name)),
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def saveparameters(resultpath: str, method: str):
    f = open(os.path.join(resultpath, 'parameters.txt'), 'w')
    f.write('Regularisation parameters:\n')
    if method == 'l2h1':
        f.write('alpha0={0} (v_x)\n'.format(alpha0))
        f.write('alpha1={0} (v_t)\n\n'.format(alpha1))
    elif method == 'l2h1h1':
        f.write('alpha0={0} (v_x)\n'.format(alpha0))
        f.write('alpha1={0} (v_t)\n'.format(alpha1))
        f.write('alpha2={0} (k_x)\n'.format(alpha2))
        f.write('alpha3={0} (k_t)\n\n'.format(alpha3))
    elif method == 'l2h1h1cr':
        f.write('alpha0={0} (v_x)\n'.format(alpha0))
        f.write('alpha1={0} (v_t)\n'.format(alpha1))
        f.write('alpha2={0} (k_x)\n'.format(alpha2))
        f.write('alpha3={0} (k_t)\n'.format(alpha3))
        f.write('beta={0} (D_v k)\n\n'.format(beta))
    elif method == 'l2h1l2':
        f.write('alpha0={0} (v_x)\n'.format(alpha0))
        f.write('alpha1={0} (v_t)\n'.format(alpha1))
        f.write('gamma={0} (k)\n\n'.format(gamma))
    else:
        f.write('Method not found!\n\n')

    f.write('Data:\n')
    f.write('c(t, x)={0}\n\n'.format(datastr))
    f.write('Data parameters:\n')
    f.write('w={0}\n'.format(w))
    f.write('lambda={0}\n'.format(lambdap))
    f.write('tau={0}\n'.format(tau))
    f.write('c0={0}\n'.format(c0))
    f.close()


class Data:
    def create(self, m: int, n: int, v: float,
               lambdap: float, tau: float) -> np.array:

        x, t = np.meshgrid(np.linspace(0, 1, num=n - 1),
                           np.linspace(0, 1, num=m - 1))
        return self.f(t, x)


class ConstantData(Data):
    def f(self, t, x):
        return np.cos((x - w * t) / lambdap) + c0

    def string(self):
        return "cos((x - w * t) / lambda) + c0"


class DecayingData(Data):
    def f(self, t, x):
        return np.exp(- t / tau) * np.cos((x - w * t) / lambdap) + c0

    def string(self):
        return "exp(- t / tau) * cos((x - w * t) / lambda) + c0"


class f_const(Expression):
    def eval(self, value, x):
        value[0] = np.cos((x[1] - w * x[0]) / lambdap) + c0

    def value_shape(self):
        return ()


class f_const_x(Expression):
    def eval(self, value, x):
        value[0] = - np.sin((x[1] - w * x[0]) / lambdap) / lambdap

    def value_shape(self):
        return ()


class f_const_t(Expression):
    def eval(self, value, x):
        value[0] = np.sin((x[1] - w * x[0]) / lambdap) * w / lambdap

    def value_shape(self):
        return ()


class f_decay(Expression):
    def eval(self, value, x):
        value[0] = np.exp(-x[0] / tau) * np.cos((x[1] - w * x[0]) / lambdap) \
            + c0

    def value_shape(self):
        return ()


class k_decay(Expression):
    def eval(self, value, x):
        value[0] = - np.exp(-x[0] / tau) \
            * np.cos((x[1] - w * x[0]) / lambdap) / tau

    def value_shape(self):
        return ()


class v_decay(Expression):
    def eval(self, value, x):
        value[0] = w

    def value_shape(self):
        return ()


class f_decay_x(Expression):
    def eval(self, value, x):
        value[0] = - np.exp(- x[0] / tau) \
            * np.sin((x[1] - w * x[0]) / lambdap) / lambdap

    def value_shape(self):
        return ()


class f_decay_t(Expression):
    def eval(self, value, x):
        value[0] = np.exp(- x[0] / tau) * np.sin((x[1] - w * x[0]) / lambdap) \
            * w / lambdap - np.exp(- x[0] / tau) \
            * np.cos((x[1] - w * x[0]) / lambdap) / tau

    def value_shape(self):
        return ()


def saveresults(resultpath: str, name: str, method: str, f: np.array,
                v: np.array, k=None, kgt=None):
    resultpath = os.path.join(resultpath, name)
    if not os.path.exists(resultpath):
        os.makedirs(resultpath)

    saveimage(resultpath, name, f)
    savevelocity(resultpath, name, f, v)
    saveparameters(resultpath, method)
    if k is not None:
        savesource(resultpath, name, k)
    if kgt is not None:
        saveerror(resultpath, name, k, kgt)


# Set path where results are saved.
resultpath = 'results/{0}'.format(
        datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
if not os.path.exists(resultpath):
    os.makedirs(resultpath)

# Set regularisation parameters.
alpha0 = 1e-3  # v_x
alpha1 = 1e-3  # v_t
alpha2 = 1e-4  # k_x
alpha3 = 1e-4  # k_t
beta = 1e-3  # D_v k
# gamma = 1e-4  # k

# Set parameters of data.
w = 0.1
lambdap = 1 / (4 * np.pi)
tau = 1.0
c0 = 0.0

# Create mesh and function spaces.
m, n = 40, 100
mesh = UnitSquareMesh(m - 1, n - 1)
V = dh.create_function_space(mesh, 'default')
W = dh.create_function_space(mesh, 'periodic')

# Run experiments with decaying data.
f = f_decay(degree=2)
ft = f_decay_t(degree=1)
fx = f_decay_x(degree=1)
datastr = DecayingData().string()

# Interpolate function.
fa = interpolate(f, V)
fa = dh.funvec2img(fa.vector().get_local(), m, n)

fa_pb = interpolate(f, W)
fa_pb = dh.funvec2img_pb(fa_pb.vector().get_local(), m, n)

v, res, fun = cm1d_exp_pb(m, n, f, ft, fx, alpha0, alpha1)
saveresults(resultpath, 'decay_cm1d_l2h1_exp_pb',
            'l2h1', fa_pb, v)

v, k, res, fun, converged = cmscr1d_exp_pb(m, n, f, ft, fx, alpha0, alpha1,
                                           alpha2, alpha3, beta)
saveresults(resultpath, 'decay_cmscr1d_l2h1h1cr_exp_pb',
            'l2h1h1cr', fa_pb, v, k, (c0 - fa_pb)/tau)


# The next example shows that for the initial concentration used in the
# mechanical models the algorithm picks up the source well.
class f_decay(Expression):
    def eval(self, value, x):
        value[0] = 20 - np.sin(10 * np.pi * x[1]
                               + np.cos(10 * np.pi * x[1])) / 5 + c0 * x[0]

    def value_shape(self):
        return ()


class f_decay_x(Expression):
    def eval(self, value, x):
        value[0] = - 2 * np.pi * (np.sin(10 * np.pi * x[1]) - 1) \
            * np.cos(10 * np.pi * x[1] + np.cos(10 * np.pi * x[1]))

    def value_shape(self):
        return ()


class f_decay_t(Expression):
    def eval(self, value, x):
        value[0] = c0

    def value_shape(self):
        return ()


# Set parameters of data.
w = 0
lambdap = 1 / (4 * np.pi)
tau = 1.0
c0 = 0.1

# Run experiments with decaying data.
f = f_decay(degree=2)
ft = f_decay_t(degree=1)
fx = f_decay_x(degree=1)
datastr = DecayingData().string()

fa_pb = interpolate(f, W)
fa_pb = dh.funvec2img_pb(fa_pb.vector().get_local(), m, n)

v, k, res, fun, converged = cmscr1d_exp_pb(m, n, f, ft, fx, alpha0, alpha1,
                                           alpha2, alpha3, beta)
saveresults(resultpath,
            'decay_static_cmscr1d_l2h1h1cr_exp_pb',
            'l2h1h1cr', fa_pb, v, k, c0)
