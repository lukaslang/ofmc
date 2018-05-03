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
from ofmc.model.mpi import mpi1d_exp_pb
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
    ax.set_title('Velocity profile')

    fig.savefig(os.path.join(path, '{0}-profile.png'.format(name)))
    plt.close(fig)


def saveerror(path: str, name: str, k: np.array, kgt: np.array):
    if not os.path.exists(path):
        os.makedirs(path)

    # Plot image.
    fig, ax = plt.subplots(figsize=(10, 5))
    cax = ax.imshow(np.abs(k - kgt), cmap=cm.coolwarm)
    ax.set_title('Error in k')
    fig.colorbar(cax, orientation='horizontal')

    # Save figure.
    fig.savefig(os.path.join(path, '{0}-sourceerror.png'.format(name)))
    plt.close(fig)


def saveparameters(resultpath: str):
    f = open(os.path.join(resultpath, 'parameters.txt'), 'w')
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


def saveresults(resultpath: str, name: str, f: np.array, v: np.array, k=None,
                kgt=None):
    resultpath = os.path.join('results', name)
    if not os.path.exists(resultpath):
        os.makedirs(resultpath)

    saveimage(resultpath, name, f)
    savevelocity(resultpath, name, f, v)
    saveparameters(resultpath)
    if k is not None:
        savesource(resultpath, name, k)
    if kgt is not None:
        saveerror(resultpath, name, k, kgt)


# Set path where results are saved.
resultpath = os.path.join('results')
if not os.path.exists(resultpath):
    os.makedirs(resultpath)

# Set parameters of data.
w = 0.1
lambdap = 1 / (4 * np.pi)
tau = 1.0

# Create mesh and function spaces.
m, n = 30, 100
mesh = UnitSquareMesh(m - 1, n - 1)
V = dh.create_function_space(mesh, 'default')
W = dh.create_function_space(mesh, 'periodic')


# Create noise.
class noise(Expression):
    def eval(self, value, x):
        value[0] = 0.5 * np.random.randn()

    def value_shape(self):
        return ()


noise = noise(degree=0)
noise_pb = interpolate(noise, W)
noise_pb = dh.funvec2img_pb(noise_pb.vector().get_local(), m, n)

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

v, k = mpi1d_exp_pb(m, n, f, ft, fx)
saveresults(resultpath, 'analytic_example_const_mpi_exp_pb', fa_pb, v, k)

v, k = mpi1d_exp_pb(m, n, f + noise, ft + noise, fx + noise)
saveresults(resultpath, 'analytic_example_const_mpi_exp_pb_noise', fa_pb, v, k)

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

v, k = mpi1d_exp_pb(m, n, f, ft, fx)
saveresults(resultpath, 'analytic_example_decay_mpi_exp_pb',
            fa_pb, v, k, -fa_pb/tau)

v, k = mpi1d_exp_pb(m, n, f + noise, ft + noise, fx + noise)
saveresults(resultpath, 'analytic_example_decay_mpi_exp_pb_noise',
            fa_pb, v, k, -fa_pb/tau)
