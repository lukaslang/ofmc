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
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
from dolfin import Constant
from dolfin import Expression
from dolfin import interpolate
from dolfin import UnitSquareMesh
from matplotlib import cm
from ofmc.model.of import of1d_img
from ofmc.model.of import of1d_img_pb
from ofmc.model.of import of1d_exp
from ofmc.model.of import of1d_exp_pb
from ofmc.model.cm import cm1d_img
from ofmc.model.cm import cm1d_img_pb
from ofmc.model.cm import cm1d_exp
from ofmc.model.cm import cm1d_exp_pb
from ofmc.model.cms import cms1d_img
from ofmc.model.cms import cms1d_img_pb
from ofmc.model.cms import cms1d_exp
from ofmc.model.cms import cms1d_exp_pb
from ofmc.model.cms import cms1dl2_img
from ofmc.model.cms import cms1dl2_img_pb
from ofmc.model.cms import cms1dl2_exp
from ofmc.model.cms import cms1dl2_exp_pb
from ofmc.model.cms import cms1d_given_source_exp
from ofmc.model.cms import cms1d_given_source_exp_pb
from ofmc.model.cms import cms1d_given_velocity_exp
from ofmc.model.cms import cms1d_given_velocity_exp_pb
from ofmc.model.cmscr import cmscr1d_img
from ofmc.model.cmscr import cmscr1d_img_pb
from ofmc.model.cmscr import cmscr1d_exp
from ofmc.model.cmscr import cmscr1d_exp_pb
from mpl_toolkits.axes_grid1 import make_axes_locatable
import ofmc.util.dolfinhelpers as dh

# Set font style.
font = {'weight': 'normal',
        'size': 20}
plt.rc('font', **font)

# Set colormap.
cmap = cm.viridis

# Streamlines.
density = 2
linewidth = 2


def saveimage(path: str, name: str, img: np.array):
    if not os.path.exists(path):
        os.makedirs(path)

    # Plot image.
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(img, cmap=cmap)
    ax.set_title('Concentration')

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

    #maxvel = abs(vel).max()
    #normi = mpl.colors.Normalize(vmin=-maxvel, vmax=maxvel)

    # Plot velocity.
    fig, ax = plt.subplots(figsize=(10, 5))
    #im = ax.imshow(vel, interpolation='nearest', norm=normi, cmap=cmap)
    im = ax.imshow(vel, interpolation='nearest', cmap=cmap)
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
                         color=vel, linewidth=linewidth, cmap=cmap)
#                        color=vel, linewidth=linewidth, norm=normi, cmap=cmap)
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
    t = np.linspace(0, m, 4, dtype=int, endpoint=False)
    t0, = plt.plot(vel[t[0]], label='t={0}'.format(t[0]), linewidth=linewidth)
    t1, = plt.plot(vel[t[1]], label='t={0}'.format(t[1]), linewidth=linewidth)
    t2, = plt.plot(vel[t[2]], label='t={0}'.format(t[2]), linewidth=linewidth)
    t3, = plt.plot(vel[t[3]], label='t={0}'.format(t[3]), linewidth=linewidth)
    plt.legend(handles=[t0, t1, t2, t3], bbox_to_anchor=(1, 1))
    ax.set_title('Velocity profile at different times')

    fig.tight_layout()
    fig.savefig(os.path.join(path, '{0}-profile.png'.format(name)),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def saveerror(path: str, name: str, k: np.array, kgt: np.array):
    if not os.path.exists(path):
        os.makedirs(path)

    # Plot image.
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(np.abs(k - kgt), cmap=cmap)
    ax.set_title('Error in k')

    # Create colourbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # Save figure.
    fig.tight_layout()
    fig.savefig(os.path.join(path, '{0}-sourceerror.png'.format(name)),
                dpi=300, bbox_inches='tight')
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

resultpath = os.path.join(resultpath, 'analytical_example')
if not os.path.exists(resultpath):
    os.makedirs(resultpath)

# Set regularisation parameters.
alpha0 = 1e-3  # v_x
alpha1 = 1e-3  # v_t
alpha2 = 1e-3  # k_x
alpha3 = 1e-3  # k_t
beta = 1e-1  # D_v k
gamma = 1e-3  # k

# Set parameters of data.
w = 0.1
lambdap = 1 / (4 * np.pi)
tau = 1.0
c0 = 0.0

# Create mesh and function spaces.
m, n = 30, 100
mesh = UnitSquareMesh(m - 1, n - 1)
V = dh.create_function_space(mesh, 'default')
W = dh.create_function_space(mesh, 'periodic')

# Run experiments with non-decaying data.
f = ConstantData().create(m, n, w, lambdap, tau)
datastr = ConstantData().string()

v = of1d_img(f, alpha0, alpha1, 'mesh')
saveresults(resultpath, 'analytic_example_const_of1d_l2h1_img',
            'l2h1', f, v)

v = of1d_img_pb(f, alpha0, alpha1, 'mesh')
saveresults(resultpath, 'analytic_example_const_of1d_l2h1_img_pb',
            'l2h1', f, v)

v = cm1d_img(f, alpha0, alpha1, 'mesh')
saveresults(resultpath, 'analytic_example_const_cm1d_l2h1_img',
            'l2h1', f, v)

v = cm1d_img_pb(f, alpha0, alpha1, 'mesh')
saveresults(resultpath, 'analytic_example_const_cm1d_l2h1_img_pb',
            'l2h1', f, v)

v, k = cms1d_img(f, alpha0, alpha1, alpha2, alpha3, 'mesh')
saveresults(resultpath, 'analytic_example_const_cms1d_l2h1h1_img',
            'l2h1h1', f, v, k)

v, k = cms1d_img_pb(f, alpha0, alpha1, alpha2, alpha3, 'mesh')
saveresults(resultpath, 'analytic_example_const_cms1d_l2h1h1_img_pb',
            'l2h1h1', f, v, k)

v, k = cms1dl2_img(f, alpha0, alpha1, gamma, 'mesh')
saveresults(resultpath, 'analytic_example_const_cms1d_l2h1l2_img',
            'l2h1l2', f, v, k)

v, k = cms1dl2_img_pb(f, alpha0, alpha1, gamma, 'mesh')
saveresults(resultpath, 'analytic_example_const_cms1d_l2h1l2_img_pb',
            'l2h1l2', f, v, k)

v, k = cmscr1d_img(f, alpha0, alpha1, alpha2, alpha3, beta, 'mesh')
saveresults(resultpath, 'analytic_example_const_cmscr1d_l2h1h1cr_img',
            'l2h1h1cr', f, v, k)

v, k = cmscr1d_img_pb(f, alpha0, alpha1, alpha2, alpha3, beta, 'mesh')
saveresults(resultpath, 'analytic_example_const_cmscr1d_l2h1h1cr_img_pb',
            'l2h1h1cr', f, v, k)

# Run experiments with decaying data.
f = DecayingData().create(m, n, w, lambdap, tau)
datastr = DecayingData().string()

v = of1d_img(f, alpha0, alpha1, 'mesh')
saveresults(resultpath, 'analytic_example_decay_of1d_l2h1_img',
            'l2h1', f, v)

v = of1d_img_pb(f, alpha0, alpha1, 'mesh')
saveresults(resultpath, 'analytic_example_decay_of1d_l2h1_img_pb',
            'l2h1', f, v)

v = cm1d_img(f, alpha0, alpha1, 'mesh')
saveresults(resultpath, 'analytic_example_decay_cm1d_l2h1_img',
            'l2h1', f, v)

v = cm1d_img_pb(f, alpha0, alpha1, 'mesh')
saveresults(resultpath, 'analytic_example_decay_cm1d_l2h1_img_pb',
            'l2h1', f, v)

v, k = cms1d_img(f, alpha0, alpha1, alpha2, alpha3, 'mesh')
saveresults(resultpath, 'analytic_example_decay_cms1d_l2h1h1_img',
            'l2h1h1', f, v, k, (c0 - f)/tau)

v, k = cms1d_img_pb(f, alpha0, alpha1, alpha2, alpha3, 'mesh')
saveresults(resultpath, 'analytic_example_decay_cms1d_l2h1h1_img_pb',
            'l2h1h1', f, v, k, (c0 - f)/tau)

v, k = cms1dl2_img(f, alpha0, alpha1, gamma, 'mesh')
saveresults(resultpath, 'analytic_example_decay_cms1dl2_l2h1l2_img',
            'l2h1l2', f, v, k, (c0 - f)/tau)

v, k = cms1dl2_img_pb(f, alpha0, alpha1, gamma, 'mesh')
saveresults(resultpath, 'analytic_example_decay_cms1dl2_l2h1l2_img_pb',
            'l2h1l2', f, v, k, (c0 - f)/tau)

v, k = cmscr1d_img(f, alpha0, alpha1, alpha2, alpha3, beta, 'mesh')
saveresults(resultpath, 'analytic_example_decay_cmscr1d_l2h1h1cr_img',
            'l2h1h1cr', f, v, k, (c0 - f)/tau)

v, k = cmscr1d_img_pb(f, alpha0, alpha1, alpha2, alpha3, beta, 'mesh')
saveresults(resultpath, 'analytic_example_decay_cmscr1d_l2h1h1cr_img_pb',
            'l2h1h1cr', f, v, k, (c0 - f)/tau)

# Run experiments with constant data.
f = f_const(degree=2)
ft = f_const_t(degree=1)
fx = f_const_x(degree=1)
datastr = ConstantData().string()

# Interpolate function.
fa = interpolate(f, V)
fa = dh.funvec2img(fa.vector().get_local(), m, n)

fa_pb = interpolate(f, W)
fa_pb = dh.funvec2img_pb(fa_pb.vector().get_local(), m, n)

v = of1d_exp(m, n, f, ft, fx, alpha0, alpha1)
saveresults(resultpath, 'analytic_example_const_of1d_l2h1_exp',
            'l2h1', fa, v)

v = of1d_exp_pb(m, n, f, ft, fx, alpha0, alpha1)
saveresults(resultpath, 'analytic_example_const_of1d_l2h1_exp_pb',
            'l2h1', fa_pb, v)

v = cm1d_exp(m, n, f, ft, fx, alpha0, alpha1)
saveresults(resultpath, 'analytic_example_const_cm1d_l2h1_exp',
            'l2h1', fa, v)

v = cm1d_exp_pb(m, n, f, ft, fx, alpha0, alpha1)
saveresults(resultpath, 'analytic_example_const_cm1d_l2h1_exp_pb',
            'l2h1', fa_pb, v)

v, k = cms1d_exp(m, n, f, ft, fx, alpha0, alpha1, alpha2, alpha3)
saveresults(resultpath, 'analytic_example_const_cms1d_l2h1h1_exp',
            'l2h1h1', fa, v, k)

v, k = cms1d_exp_pb(m, n, f, ft, fx, alpha0, alpha1, alpha2, alpha3)
saveresults(resultpath, 'analytic_example_const_cms1d_l2h1h1_exp_pb',
            'l2h1h1', fa_pb, v, k)

v, k = cms1dl2_exp(m, n, f, ft, fx, alpha0, alpha1, gamma)
saveresults(resultpath, 'analytic_example_const_cms1dl2_l2h1l2_exp',
            'l2h1l2', fa, v, k)

v, k = cms1dl2_exp_pb(m, n, f, ft, fx, alpha0, alpha1, gamma)
saveresults(resultpath, 'analytic_example_const_cms1dl2_l2h1l2_exp_pb',
            'l2h1l2', fa_pb, v, k)

v, k = cmscr1d_exp(m, n, f, ft, fx, alpha0, alpha1, alpha2, alpha3, beta)
saveresults(resultpath, 'analytic_example_const_cmscr1d_l2h1h1cr_exp',
            'l2h1h1cr', fa, v, k)

v, k = cmscr1d_exp_pb(m, n, f, ft, fx, alpha0, alpha1, alpha2, alpha3, beta)
saveresults(resultpath, 'analytic_example_const_cmscr1d_l2h1h1cr_exp_pb',
            'l2h1h1cr', fa_pb, v, k)

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

v = of1d_exp(m, n, f, ft, fx, alpha0, alpha1)
saveresults(resultpath, 'analytic_example_decay_of1d_l2h1_exp',
            'l2h1', fa, v)

v = of1d_exp_pb(m, n, f, ft, fx, alpha0, alpha1)
saveresults(resultpath, 'analytic_example_decay_of1d_l2h1_exp_pb',
            'l2h1', fa_pb, v)

v = cm1d_exp(m, n, f, ft, fx, alpha0, alpha1)
saveresults(resultpath, 'analytic_example_decay_cm1d_l2h1_exp',
            'l2h1', fa, v)

v = cm1d_exp_pb(m, n, f, ft, fx, alpha0, alpha1)
saveresults(resultpath, 'analytic_example_decay_cm1d_l2h1_exp_pb',
            'l2h1', fa_pb, v)

v, k = cms1d_exp(m, n, f, ft, fx, alpha0, alpha1, alpha2, alpha3)
saveresults(resultpath, 'analytic_example_decay_cms1d_l2h1h1_exp',
            'l2h1h1', fa, v, k, (c0 - fa)/tau)

v, k = cms1d_exp_pb(m, n, f, ft, fx, alpha0, alpha1, alpha2, alpha3)
saveresults(resultpath, 'analytic_example_decay_cms1d_l2h1h1_exp_pb',
            'l2h1h1', fa_pb, v, k, (c0 - fa_pb)/tau)

v, k = cms1dl2_exp(m, n, f, ft, fx, alpha0, alpha1, gamma)
saveresults(resultpath, 'analytic_example_decay_cms1dl2_l2h1l2_exp',
            'l2h1l2', fa, v, k, (c0 - fa)/tau)

v, k = cms1dl2_exp_pb(m, n, f, ft, fx, alpha0, alpha1, gamma)
saveresults(resultpath, 'analytic_example_decay_cms1dl2_l2h1l2_exp_pb',
            'l2h1l2', fa_pb, v, k, (c0 - fa_pb)/tau)

v, k = cmscr1d_exp(m, n, f, ft, fx, alpha0, alpha1, alpha2, alpha3, beta)
saveresults(resultpath, 'analytic_example_decay_cmscr1d_l2h1h1cr_exp',
            'l2h1h1cr', fa, v, k, (c0 - fa)/tau)

v, k = cmscr1d_exp_pb(m, n, f, ft, fx, alpha0, alpha1, alpha2, alpha3, beta)
saveresults(resultpath, 'analytic_example_decay_cmscr1d_l2h1h1cr_exp_pb',
            'l2h1h1cr', fa_pb, v, k, (c0 - fa_pb)/tau)

# Run experiments with given source.
k = k_decay(degree=2)
ka = interpolate(k, V)
ka = dh.funvec2img(ka.vector().get_local(), m, n)

ka_pb = interpolate(k, W)
ka_pb = dh.funvec2img_pb(ka_pb.vector().get_local(), m, n)

v = cms1d_given_source_exp(m, n, f, ft, fx, k, alpha0, alpha1)
saveresults(resultpath,
            'analytic_example_decay_given_source_cms1d_l2h1_exp',
            'l2h1', fa, v, ka)

v = cms1d_given_source_exp_pb(m, n, f, ft, fx, k, alpha0, alpha1)
saveresults(resultpath,
            'analytic_example_decay_given_source_cms1d_l2h1_exp_pb',
            'l2h1', fa_pb, v, ka_pb)

# Run experiments with given velocity.
v = Constant(w)
va = interpolate(v, V)
va = dh.funvec2img(va.vector().get_local(), m, n)

va_pb = interpolate(v, W)
va_pb = dh.funvec2img_pb(va_pb.vector().get_local(), m, n)

k = cms1d_given_velocity_exp(m, n, f, ft, fx, v, Constant(0.0),
                             alpha2, alpha3)
saveresults(resultpath,
            'analytic_example_decay_given_velocity_cms1d_l2h1_exp',
            'l2h1', fa, va, k)

k = cms1d_given_velocity_exp_pb(m, n, f, ft, fx, v, Constant(0.0),
                                alpha2, alpha3)
saveresults(resultpath,
            'analytic_example_decay_given_velocity_cms1d_l2h1_exp_pb',
            'l2h1', fa_pb, va_pb, k, (c0 - fa_pb)/tau)

# Visualise increasing regularisation parameter of convective regularisation.
beta = 1e-3
v, k = cmscr1d_exp_pb(m, n, f, ft, fx, alpha0, alpha1, alpha2, alpha3, beta)
saveresults(resultpath,
            'analytic_example_decay_cmscr1d_l2h1h1cr_exp_pb_beta_0.001',
            'l2h1h1cr', fa_pb, v, k, (c0 - fa_pb)/tau)
beta = 1e-2
v, k = cmscr1d_exp_pb(m, n, f, ft, fx, alpha0, alpha1, alpha2, alpha3, beta)
saveresults(resultpath,
            'analytic_example_decay_cmscr1d_l2h1h1cr_exp_pb_beta_0.01',
            'l2h1h1cr', fa_pb, v, k, (c0 - fa_pb)/tau)
beta = 1e-1
v, k = cmscr1d_exp_pb(m, n, f, ft, fx, alpha0, alpha1, alpha2, alpha3, beta)
saveresults(resultpath,
            'analytic_example_decay_cmscr1d_l2h1h1cr_exp_pb_beta_0.1',
            'l2h1h1cr', fa_pb, v, k, (c0 - fa_pb)/tau)

# Increasing regularistaion parameters.
alpha0 = 1e-5
alpha1 = 1e-5
alpha2 = 1e-5
alpha3 = 1e-5
v, k = cms1d_exp_pb(m, n, f, ft, fx, alpha0, alpha1, alpha2, alpha3)
saveresults(resultpath,
            'analytic_example_decay_cms1d_l2h1h1_exp_pb_reg_0.00001',
            'l2h1h1', fa_pb, v, k, (c0 - fa_pb)/tau)
alpha0 = 1e-4
alpha1 = 1e-4
alpha2 = 1e-4
alpha3 = 1e-4
v, k = cms1d_exp_pb(m, n, f, ft, fx, alpha0, alpha1, alpha2, alpha3)
saveresults(resultpath,
            'analytic_example_decay_cms1d_l2h1h1_exp_pb_reg_0.0001',
            'l2h1h1', fa_pb, v, k, (c0 - fa_pb)/tau)
alpha0 = 1e-3
alpha1 = 1e-3
alpha2 = 1e-3
alpha3 = 1e-3
v, k = cms1d_exp_pb(m, n, f, ft, fx, alpha0, alpha1, alpha2, alpha3)
saveresults(resultpath,
            'analytic_example_decay_cms1d_l2h1h1_exp_pb_reg_0.001',
            'l2h1h1', fa_pb, v, k, (c0 - fa_pb)/tau)
alpha0 = 1e-2
alpha1 = 1e-2
alpha2 = 1e-2
alpha3 = 1e-2
v, k = cms1d_exp_pb(m, n, f, ft, fx, alpha0, alpha1, alpha2, alpha3)
saveresults(resultpath, 'analytic_example_decay_cms1d_l2h1h1_exp_pb_reg_0.01',
            'l2h1h1', fa_pb, v, k, (c0 - fa_pb)/tau)
alpha0 = 1e-1
alpha1 = 1e-1
alpha2 = 1e-1
alpha3 = 1e-1
v, k = cms1d_exp_pb(m, n, f, ft, fx, alpha0, alpha1, alpha2, alpha3)
saveresults(resultpath, 'analytic_example_decay_cms1d_l2h1h1_exp_pb_reg_0.1',
            'l2h1h1', fa_pb, v, k, (c0 - fa_pb)/tau)
alpha0 = 1e0
alpha1 = 1e0
alpha2 = 1e0
alpha3 = 1e0
v, k = cms1d_exp_pb(m, n, f, ft, fx, alpha0, alpha1, alpha2, alpha3)
saveresults(resultpath, 'analytic_example_decay_cms1d_l2h1h1_exp_pb_reg_1.0',
            'l2h1h1', fa_pb, v, k, (c0 - fa_pb)/tau)

# Different mesh sizes.
m, n = 50, 50
mesh = UnitSquareMesh(m - 1, n - 1)
W = dh.create_function_space(mesh, 'periodic')

# Interpolate function.
fa_pb = interpolate(f, W)
fa_pb = dh.funvec2img_pb(fa_pb.vector().get_local(), m, n)

v, k = cms1d_exp_pb(m, n, f, ft, fx, alpha0, alpha1, alpha2, alpha3)
saveresults(resultpath, 'analytic_example_decay_cms1d_l2h1h1_exp_pb_mesh_50',
            'l2h1h1', fa_pb, v, k, (c0 - fa_pb)/tau)

m, n = 100, 100
mesh = UnitSquareMesh(m - 1, n - 1)
W = dh.create_function_space(mesh, 'periodic')

# Interpolate function.
fa_pb = interpolate(f, W)
fa_pb = dh.funvec2img_pb(fa_pb.vector().get_local(), m, n)

v, k = cms1d_exp_pb(m, n, f, ft, fx, alpha0, alpha1, alpha2, alpha3)
saveresults(resultpath, 'analytic_example_decay_cms1d_l2h1h1_exp_pb_mesh_100',
            'l2h1h1', fa_pb, v, k, (c0 - fa_pb)/tau)

m, n = 200, 200
mesh = UnitSquareMesh(m - 1, n - 1)
W = dh.create_function_space(mesh, 'periodic')

# Interpolate function.
fa_pb = interpolate(f, W)
fa_pb = dh.funvec2img_pb(fa_pb.vector().get_local(), m, n)

v, k = cms1d_exp_pb(m, n, f, ft, fx, alpha0, alpha1, alpha2, alpha3)
saveresults(resultpath, 'analytic_example_decay_cms1d_l2h1h1_exp_pb_mesh_200',
            'l2h1h1', fa_pb, v, k, (c0 - fa_pb)/tau)

# vel, k = cmscr1dnewton(img, alpha0, alpha1, alpha2, alpha3, beta)


#c = 1.0
#
#
#class f_decay(Expression):
#    def eval(self, value, x):
#        value[0] = np.cos((x[1] - w * x[0]) / lambdap) - c*x[0]
#
#    def value_shape(self):
#        return ()
#
#
#class f_decay_x(Expression):
#    def eval(self, value, x):
#        value[0] = - np.sin((x[1] - w * x[0]) / lambdap) / lambdap
#
#    def value_shape(self):
#        return ()
#
#
#class f_decay_t(Expression):
#    def eval(self, value, x):
#        value[0] = np.sin((x[1] - w * x[0]) / lambdap) \
#            * w / lambdap - c
#
#    def value_shape(self):
#        return ()
#
#
## Run experiments with decaying data.
#f = f_decay(degree=2)
#ft = f_decay_t(degree=1)
#fx = f_decay_x(degree=1)
#datastr = DecayingData().string()
#
#fa_pb = interpolate(f, W)
#fa_pb = dh.funvec2img_pb(fa_pb.vector().get_local(), m, n)
#
#m, n = 100, 100
#mesh = UnitSquareMesh(m - 1, n - 1)
#W = dh.create_function_space(mesh, 'periodic')
#
## Interpolate function.
#fa_pb = interpolate(f, W)
#fa_pb = dh.funvec2img_pb(fa_pb.vector().get_local(), m, n)
#
#v, k = cms1d_exp_pb(m, n, f, ft, fx, alpha0, alpha1, alpha2, alpha3)
#saveresults(resultpath, 'analytic_example_linear_decay_cms1d_exp_pb',
#            fa_pb, v, k)
