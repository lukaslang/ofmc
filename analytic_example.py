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
import ofmc.util.dolfinhelpers as dh


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
    V = np.ones_like(X)

    # Plot streamlines.
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.imshow(img, cmap=cm.gray)
    strm = ax.streamplot(X, Y, vel*hx/hy, V, density=2,
#    strm = ax.streamplot(X, Y, vel*hx, V, density=2,
#                         color=vel, linewidth=1, norm=normi, cmap=cm.coolwarm)
                         color=vel, linewidth=1, cmap=cm.coolwarm)
    fig.colorbar(strm.lines, orientation='horizontal')

    fig.savefig(os.path.join(path, '{0}-streamlines.png'.format(name)))
    plt.close(fig)

    # Save velocity profile after cut.
    fig, ax = plt.subplots(figsize=(10, 5))
    t = np.linspace(0, m, 4, dtype=int, endpoint=False)
    t0, = plt.plot(vel[t[0]], label='t={0}'.format(t[0]))
    t1, = plt.plot(vel[t[1]], label='t={0}'.format(t[1]))
    t2, = plt.plot(vel[t[2]], label='t={0}'.format(t[2]))
    t3, = plt.plot(vel[t[3]], label='t={0}'.format(t[3]))
    plt.legend(handles=[t0, t1, t2, t3])
    ax.set_title('Velocity profile at time zero')

    fig.savefig(os.path.join(path, '{0}-profile.png'.format(name)))
    plt.close(fig)


def saveerror(path: str, name: str, img: np.array, k: np.array):
    if not os.path.exists(path):
        os.makedirs(path)

    # Plot image.
    fig, ax = plt.subplots(figsize=(10, 5))
    cax = ax.imshow(np.abs(-img - k), cmap=cm.coolwarm)
    ax.set_title('abs(-c/tau - k)')
    fig.colorbar(cax, orientation='horizontal')

    # Save figure.
    fig.savefig(os.path.join(path, '{0}-error.png'.format(name)))
    plt.close(fig)


def saveparameters(resultpath: str):
    f = open(os.path.join(resultpath, 'parameters.txt'), 'w')
    f.write('Regularisation parameters:\n')
    f.write('alpha0={0}\n'.format(alpha0))
    f.write('alpha1={0}\n'.format(alpha1))
    f.write('alpha2={0}\n'.format(alpha2))
    f.write('alpha3={0}\n'.format(alpha3))
    f.write('beta={0}\n\n'.format(beta))
    f.write('c(t, x)={0}\n\n'.format(datastr))
    f.write('Data parameters:\n')
    f.write('w={0}\n'.format(w))
    f.write('lambda={0}\n'.format(lambdap))
    f.write('tau={0}\n'.format(tau))
    f.close()


class Data:
    def create(self, m: int, n: int, v: float,
               lambdap: float, tau: float) -> np.array:

        x, t = np.meshgrid(np.linspace(0, 1, num=n - 1),
                           np.linspace(0, 1, num=m - 1))
        return self.f(t, x)


class ConstantData(Data):
    def f(self, t, x):
        return np.cos((x - w * t) / lambdap)

    def string(self):
        return "cos((x - w * t) / lambda)"


class DecayingData(Data):
    def f(self, t, x):
        return np.exp(- t / tau) * np.cos((x - w * t) / lambdap)

    def string(self):
        return "exp(- t / tau) * cos((x - w * t) / lambda)"


class f_const(Expression):
    def eval(self, value, x):
        value[0] = np.cos((x[1] - w * x[0]) / lambdap)

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
        value[0] = np.exp(-x[0] / tau) * np.cos((x[1] - w * x[0]) / lambdap)

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


def saveresults(resultpath: str, name: str, f: np.array, v: np.array, k=None):
    resultpath = os.path.join('results', name)
    if not os.path.exists(resultpath):
        os.makedirs(resultpath)

    saveimage(resultpath, name, f)
    savevelocity(resultpath, name, f, v)
    saveparameters(resultpath)
    if k is not None:
        savesource(resultpath, name, k)
        # saveerror(resultpath, name, f, k)


# Set path where results are saved.
resultpath = os.path.join('results')
if not os.path.exists(resultpath):
    os.makedirs(resultpath)

# Set regularisation parameter.
alpha0 = 1e-3
alpha1 = 1e-3
alpha2 = 1e-3
alpha3 = 1e-3
beta = 1e-1

# Set parameters of data.
w = 0.1
lambdap = 1 / (4 * np.pi)
tau = 1.0

# Create mesh and function spaces.
m, n = 30, 100
mesh = UnitSquareMesh(m - 1, n - 1)
V = dh.create_function_space(mesh, 'default')
W = dh.create_function_space(mesh, 'periodic')

# Run experiments with non-decaying data.
f = ConstantData().create(m, n, w, lambdap, tau)
datastr = ConstantData().string()

v = of1d_img(f, alpha0, alpha1, 'mesh')
saveresults(resultpath, 'analytic_example_const_of1d_img', f, v)

v = of1d_img_pb(f, alpha0, alpha1, 'mesh')
saveresults(resultpath, 'analytic_example_const_of1d_img_pb', f, v)

v = cm1d_img(f, alpha0, alpha1, 'mesh')
saveresults(resultpath, 'analytic_example_const_cm1d_img', f, v)

v = cm1d_img_pb(f, alpha0, alpha1, 'mesh')
saveresults(resultpath, 'analytic_example_const_cm1d_img_pb', f, v)

v, k = cms1d_img(f, alpha0, alpha1, alpha2, alpha3, 'mesh')
saveresults(resultpath, 'analytic_example_const_cms1d_img', f, v, k)

v, k = cms1d_img_pb(f, alpha0, alpha1, alpha2, alpha3, 'mesh')
saveresults(resultpath, 'analytic_example_const_cms1d_img_pb', f, v, k)

v, k = cms1dl2_img(f, alpha0, alpha1, alpha2, 'mesh')
saveresults(resultpath, 'analytic_example_const_cms1dl2_img', f, v, k)

v, k = cms1dl2_img_pb(f, alpha0, alpha1, alpha2, 'mesh')
saveresults(resultpath, 'analytic_example_const_cms1dl2_img_pb', f, v, k)

v, k = cmscr1d_img(f, alpha0, alpha1, alpha2, alpha3, beta, 'mesh')
saveresults(resultpath, 'analytic_example_const_cmscr1d_img', f, v, k)

v, k = cmscr1d_img_pb(f, alpha0, alpha1, alpha2, alpha3, beta, 'mesh')
saveresults(resultpath, 'analytic_example_const_cmscr1d_img_pb', f, v, k)

# Run experiments with decaying data.
f = DecayingData().create(m, n, w, lambdap, tau)
datastr = DecayingData().string()

v = of1d_img(f, alpha0, alpha1, 'mesh')
saveresults(resultpath, 'analytic_example_decay_of1d_img', f, v)

v = of1d_img_pb(f, alpha0, alpha1, 'mesh')
saveresults(resultpath, 'analytic_example_decay_of1d_img_pb', f, v)

v = cm1d_img(f, alpha0, alpha1, 'mesh')
saveresults(resultpath, 'analytic_example_decay_cm1d_img', f, v)

v = cm1d_img_pb(f, alpha0, alpha1, 'mesh')
saveresults(resultpath, 'analytic_example_decay_cm1d_img_pb', f, v)

v, k = cms1d_img(f, alpha0, alpha1, alpha2, alpha3, 'mesh')
saveresults(resultpath, 'analytic_example_decay_cms1d_img', f, v, k)

v, k = cms1d_img_pb(f, alpha0, alpha1, alpha2, alpha3, 'mesh')
saveresults(resultpath, 'analytic_example_decay_cms1d_img_pb', f, v, k)

v, k = cms1dl2_img(f, alpha0, alpha1, alpha2, 'mesh')
saveresults(resultpath, 'analytic_example_decay_cms1dl2_img', f, v, k)

v, k = cms1dl2_img_pb(f, alpha0, alpha1, alpha2, 'mesh')
saveresults(resultpath, 'analytic_example_decay_cms1dl2_img_pb', f, v, k)

v, k = cmscr1d_img(f, alpha0, alpha1, alpha2, alpha3, beta, 'mesh')
saveresults(resultpath, 'analytic_example_decay_cmscr1d_img', f, v, k)

v, k = cmscr1d_img_pb(f, alpha0, alpha1, alpha2, alpha3, beta, 'mesh')
saveresults(resultpath, 'analytic_example_decay_cmscr1d_img_pb', f, v, k)

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
saveresults(resultpath, 'analytic_example_const_of1d_exp', fa, v)

v = of1d_exp_pb(m, n, f, ft, fx, alpha0, alpha1)
saveresults(resultpath, 'analytic_example_const_of1d_exp_pb', fa_pb, v)

v = cm1d_exp(m, n, f, ft, fx, alpha0, alpha1)
saveresults(resultpath, 'analytic_example_const_cm1d_exp', fa, v)

v = cm1d_exp_pb(m, n, f, ft, fx, alpha0, alpha1)
saveresults(resultpath, 'analytic_example_const_cm1d_exp_pb', fa_pb, v)

v, k = cms1d_exp(m, n, f, ft, fx, alpha0, alpha1, alpha2, alpha3)
saveresults(resultpath, 'analytic_example_const_cms1d_exp', fa, v, k)

v, k = cms1d_exp_pb(m, n, f, ft, fx, alpha0, alpha1, alpha2, alpha3)
saveresults(resultpath, 'analytic_example_const_cms1d_exp_pb', fa_pb, v, k)

v, k = cms1dl2_exp(m, n, f, ft, fx, alpha0, alpha1, alpha2)
saveresults(resultpath, 'analytic_example_const_cms1dl2_exp', fa, v, k)

v, k = cms1dl2_exp_pb(m, n, f, ft, fx, alpha0, alpha1, alpha2)
saveresults(resultpath, 'analytic_example_const_cms1dl2_exp_pb', fa_pb, v, k)

v, k = cmscr1d_exp(m, n, f, ft, fx, alpha0, alpha1, alpha2, alpha3, beta)
saveresults(resultpath, 'analytic_example_const_cmscr1d_exp', fa, v, k)

v, k = cmscr1d_exp_pb(m, n, f, ft, fx, alpha0, alpha1, alpha2, alpha3, beta)
saveresults(resultpath, 'analytic_example_const_cmscr1d_exp_pb', fa_pb, v, k)

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
saveresults(resultpath, 'analytic_example_decay_of1d_exp', fa, v)

v = of1d_exp_pb(m, n, f, ft, fx, alpha0, alpha1)
saveresults(resultpath, 'analytic_example_decay_of1d_exp_pb', fa_pb, v)

v = cm1d_exp(m, n, f, ft, fx, alpha0, alpha1)
saveresults(resultpath, 'analytic_example_decay_cm1d_exp', fa, v)

v = cm1d_exp_pb(m, n, f, ft, fx, alpha0, alpha1)
saveresults(resultpath, 'analytic_example_decay_cm1d_exp_pb', fa_pb, v)

v, k = cms1d_exp(m, n, f, ft, fx, alpha0, alpha1, alpha2, alpha3)
saveresults(resultpath, 'analytic_example_decay_cms1d_exp', fa, v, k)

v, k = cms1d_exp_pb(m, n, f, ft, fx, alpha0, alpha1, alpha2, alpha3)
saveresults(resultpath, 'analytic_example_decay_cms1d_exp_pb', fa_pb, v, k)

v, k = cms1dl2_exp(m, n, f, ft, fx, alpha0, alpha1, alpha2)
saveresults(resultpath, 'analytic_example_decay_cms1dl2_exp', fa, v, k)

v, k = cms1dl2_exp_pb(m, n, f, ft, fx, alpha0, alpha1, alpha2)
saveresults(resultpath, 'analytic_example_decay_cms1dl2_exp_pb', fa_pb, v, k)

v, k = cmscr1d_exp(m, n, f, ft, fx, alpha0, alpha1, alpha2, alpha3, beta)
saveresults(resultpath, 'analytic_example_decay_cmscr1d_exp', fa, v, k)

v, k = cmscr1d_exp_pb(m, n, f, ft, fx, alpha0, alpha1, alpha2, alpha3, beta)
saveresults(resultpath, 'analytic_example_decay_cmscr1d_exp_pb', fa_pb, v, k)

# Run experiments with given source.
k = k_decay(degree=2)
ka = interpolate(k, V)
ka = dh.funvec2img(ka.vector().get_local(), m, n)

ka_pb = interpolate(k, W)
ka_pb = dh.funvec2img_pb(ka_pb.vector().get_local(), m, n)

v = cms1d_given_source_exp(m, n, f, ft, fx, k, alpha0, alpha1)
saveresults(resultpath,
            'analytic_example_decay_given_source_cms1d_exp', fa, v, ka)

v = cms1d_given_source_exp_pb(m, n, f, ft, fx, k, alpha0, alpha1)
saveresults(resultpath, 'analytic_example_decay_given_source_cms1d_exp_pb',
            fa_pb, v, ka_pb)

# Run experiments with given velocity.
v = Constant(w)
va = interpolate(v, V)
va = dh.funvec2img(va.vector().get_local(), m, n)

va_pb = interpolate(v, W)
va_pb = dh.funvec2img_pb(va_pb.vector().get_local(), m, n)

k = cms1d_given_velocity_exp(m, n, f, ft, fx, v, Constant(0.0),
                             alpha2, alpha3)
saveresults(resultpath,
            'analytic_example_decay_given_velocity_cms1d_exp', fa, va, k)

k = cms1d_given_velocity_exp_pb(m, n, f, ft, fx, v, Constant(0.0),
                                alpha2, alpha3)
saveresults(resultpath, 'analytic_example_decay_given_velocity_cms1d_exp_pb',
            fa_pb, va_pb, k)

# Visualise increasing regularisation parameter of convective regularisation.
beta = 1e-3
v, k = cmscr1d_exp_pb(m, n, f, ft, fx, alpha0, alpha1, alpha2, alpha3, beta)
saveresults(resultpath, 'analytic_example_decay_cmscr1d_exp_pb_beta_0.001',
            fa_pb, v, k)
beta = 1e-2
v, k = cmscr1d_exp_pb(m, n, f, ft, fx, alpha0, alpha1, alpha2, alpha3, beta)
saveresults(resultpath, 'analytic_example_decay_cmscr1d_exp_pb_beta_0.01',
            fa_pb, v, k)
beta = 1e-1
v, k = cmscr1d_exp_pb(m, n, f, ft, fx, alpha0, alpha1, alpha2, alpha3, beta)
saveresults(resultpath, 'analytic_example_decay_cmscr1d_exp_pb_beta_0.1',
            fa_pb, v, k)

# Different mesh sizes.
m, n = 100, 100
mesh = UnitSquareMesh(m - 1, n - 1)
W = dh.create_function_space(mesh, 'periodic')

# Interpolate function.
fa_pb = interpolate(f, W)
fa_pb = dh.funvec2img_pb(fa_pb.vector().get_local(), m, n)

v, k = cms1d_exp_pb(m, n, f, ft, fx, alpha0, alpha1, alpha2, alpha3)
saveresults(resultpath, 'analytic_example_decay_cms1d_exp_pb_mesh_100',
            fa_pb, v, k)

m, n = 200, 200
mesh = UnitSquareMesh(m - 1, n - 1)
W = dh.create_function_space(mesh, 'periodic')

# Interpolate function.
fa_pb = interpolate(f, W)
fa_pb = dh.funvec2img_pb(fa_pb.vector().get_local(), m, n)

v, k = cms1d_exp_pb(m, n, f, ft, fx, alpha0, alpha1, alpha2, alpha3)
saveresults(resultpath, 'analytic_example_decay_cms1d_exp_pb_mesh_200',
            fa_pb, v, k)

m, n = 400, 400
mesh = UnitSquareMesh(m - 1, n - 1)
W = dh.create_function_space(mesh, 'periodic')

# Interpolate function.
fa_pb = interpolate(f, W)
fa_pb = dh.funvec2img_pb(fa_pb.vector().get_local(), m, n)

v, k = cms1d_exp_pb(m, n, f, ft, fx, alpha0, alpha1, alpha2, alpha3)
saveresults(resultpath, 'analytic_example_decay_cms1d_exp_pb_mesh_400',
            fa_pb, v, k)

# vel, k = cmscr1dnewton(img, alpha0, alpha1, alpha2, alpha3, beta)
