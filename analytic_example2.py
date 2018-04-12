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
from dolfin import assemble
from dolfin import dx
from dolfin import Expression
from dolfin import interpolate
from dolfin import near
from dolfin import plot
from dolfin import solve
from dolfin import Function
from dolfin import FunctionSpace
from dolfin import SubDomain
from dolfin import TestFunctions
from dolfin import TrialFunctions
from dolfin import UnitSquareMesh
from dolfin import VectorFunctionSpace
from matplotlib import cm
from ofmc.model.of import of1d_exp
from ofmc.model.of import of1d_exp_pb
from ofmc.model.cm import cm1d_exp
from ofmc.model.cm import cm1d_exp_pb
from ofmc.model.cms import cms1d_exp
from ofmc.model.cms import cms1d_exp_pb
from ofmc.model.cmscr import cmscr1d_exp
from ofmc.model.cmscr import cmscr1d_exp_pb
import ofmc.util.dolfinhelpers as dh

# Set path where results are saved.
resultpath = os.path.join('results', 'analytic_example2')
if not os.path.exists(resultpath):
    os.makedirs(resultpath)


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

    # maxvel = abs(vel).max()
    # normi = mpl.colors.Normalize(vmin=-maxvel, vmax=maxvel)

    # Plot velocity.
    fig, ax = plt.subplots(figsize=(10, 5))
    # cax = ax.imshow(vel, interpolation='nearest', norm=normi, cmap=cm.coolwarm)
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
                         color=vel, linewidth=1, cmap=cm.coolwarm)
    fig.colorbar(strm.lines, orientation='horizontal')

    fig.savefig(os.path.join(path, '{0}-streamlines.png'.format(name)))
    plt.close(fig)

    # Save velocity profile after cut.
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.plot(vel[0])
    #ax.set_ylim([0, 1])
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


# Set name.
name = 'analytic_example2'

# Set regularisation parameter.
alpha0 = 1e-3
alpha1 = 1e-3
alpha2 = 1e-3
alpha3 = 1e-3
beta = 1e-1

vgt = 0.1
# ell = 0.05
ell = 1/(4*np.pi)
tau = 1.0


class f(Expression):
    def eval(self, value, x):
        value[0] = np.exp(-x[0]/tau)*np.cos((x[1] - vgt*x[0])/ell)
        value[0] = np.cos((x[1] - vgt*x[0])/ell)

    def value_shape(self):
        return ()


class fx(Expression):
    def eval(self, value, x):
        value[0] = -np.exp(-x[0]/tau)*np.sin((x[1] - vgt*x[0])/ell)/ell
        value[0] = -np.sin((x[1] - vgt*x[0])/ell)/ell

    def value_shape(self):
        return ()


class ft(Expression):
    def eval(self, value, x):
        value[0] = np.exp(-x[0]/tau)*np.sin((x[1] - vgt*x[0])/ell)*vgt/ell \
            - np.exp(-x[0]/tau)*np.cos((x[1] - vgt*x[0])/ell)/tau
        value[0] = np.sin((x[1] - vgt*x[0])/ell)*vgt/ell

    def value_shape(self):
        return ()


class PeriodicBoundary(SubDomain):

        def inside(self, x, on_boundary):
            return bool(near(x[1], 0))

        def map(self, x, y):
            y[1] = x[1] - 1.0
            y[0] = x[0]


def cms1dl2(m: int, n: int) -> (np.array, np.array):
    mesh = UnitSquareMesh(m-1, n-1)

    # Define function space and functions.
    # P = FiniteElement('P', triangle, 1)
    # W = FunctionSpace(mesh, P * P)
    W = VectorFunctionSpace(mesh, 'CG', 1, dim=2)
    v, k = TrialFunctions(W)
    w1, w2 = TestFunctions(W)

    # Define weak formulation.
    A = - (fx*v + f*v.dx(1) - k) * (fx*w1 + f*w1.dx(1))*dx \
        - alpha0*v.dx(1)*w1.dx(1)*dx - alpha1*v.dx(0)*w1.dx(0)*dx \
        + (fx*v + f*v.dx(1) - k)*w2*dx - alpha2*k*w2*dx
    b = ft*(fx*w1 + f*w1.dx(1))*dx - ft*w2*dx

    # Compute solution.
    v = Function(W)
    solve(A == b, v)

    # Recover solution.
    v, k = v.split(deepcopy=True)

    # Evaluate residual and functional value.
    res = abs(ft + fx*v + f*v.dx(1) - k)
    func = res**2*dx + alpha0*v.dx(1)**2*dx + alpha1*v.dx(0)**2*dx \
        + alpha2*k**2*dx
    print('Res={0}, Func={1}\n'.format(assemble(res*dx),
                                       assemble(0.5*func)))

    # Convert back to array.
    vel = dh.funvec2img(v.vector().get_local(), m, n)
    k = dh.funvec2img(k.vector().get_local(), m, n)
    return vel, k


def cms1dl2periodic(m: int, n: int) -> (np.array, np.array):
    mesh = UnitSquareMesh(m-1, n-1)

    # Define function space and functions.
    # P = FiniteElement('P', triangle, 1)
    # W = FunctionSpace(mesh, P * P)
    W = VectorFunctionSpace(mesh, 'CG', 1, dim=2,
                            constrained_domain=PeriodicBoundary())

    v, k = TrialFunctions(W)
    w1, w2 = TestFunctions(W)

    # Define weak formulation.
    A = - (fx*v + f*v.dx(1) - k) * (fx*w1 + f*w1.dx(1))*dx \
        - alpha0*v.dx(1)*w1.dx(1)*dx - alpha1*v.dx(0)*w1.dx(0)*dx \
        + (fx*v + f*v.dx(1) - k)*w2*dx - alpha2*k*w2*dx
    b = ft*(fx*w1 + f*w1.dx(1))*dx - ft*w2*dx

    # Compute solution.
    v = Function(W)
    solve(A == b, v)

    # Recover solution.
    v, k = v.split(deepcopy=True)

    # Evaluate residual and functional value.
    res = abs(ft + fx*v + f*v.dx(1) - k)
    func = res**2*dx + alpha0*v.dx(1)**2*dx + alpha1*v.dx(0)**2*dx \
        + alpha2*k**2*dx
    print('Res={0}, Func={1}\n'.format(assemble(res*dx),
                                       assemble(0.5*func)))

    vel = dh.funvec2img(v.vector().get_local(), m, n)
    k = dh.funvec2img(k.vector().get_local(), m, n)
    return vel, k


f = f(degree=2)
ft = ft(degree=1)
fx = fx(degree=1)

m, n = 30, 100
# vel = of1d_exp(m, n, f, ft, fx, alpha0, alpha1)
# vel = of1d_exp_pb(m, n, f, ft, fx, alpha0, alpha1)
# vel = cm1d_exp(m, n, f, ft, fx, alpha0, alpha1)
# vel = cm1d_exp_pb(m, n, f, ft, fx, alpha0, alpha1)
# vel, k = cms1d_exp(m, n, f, ft, fx, alpha0, alpha1, alpha2, alpha3)
# vel, k = cms1d_exp_pb(m, n, f, ft, fx, alpha0, alpha1, alpha2, alpha3)
# vel, k = cms1dl2(m, n)
# vel, k = cms1dl2periodic(m, n)
# vel, k = cmscr1d_exp(m, n, f, ft, fx, alpha0, alpha1, alpha2, alpha3, beta)
vel, k = cmscr1d_exp_pb(m, n, f, ft, fx, alpha0, alpha1, alpha2, alpha3, beta)


# Convert back to array.
mesh = UnitSquareMesh(m-1, n-1)
V = FunctionSpace(mesh, 'CG', 1)
f = interpolate(f, V)
img = dh.funvec2img(f.vector().get_local(), m, n)


# Plot and save figures.
saveimage(resultpath, name, img)
savevelocity(resultpath, name, img, vel)
savesource(resultpath, name, k)
# saveerror(resultpath, name, img, k)

# Write parameters to file.
f = open(os.path.join(resultpath, 'parameters.txt'), 'w')
f.write('Regularisation parameters:\n')
f.write('alpha0={0}\n'.format(alpha0))
f.write('alpha1={0}\n'.format(alpha1))
f.write('alpha2={0}\n'.format(alpha2))
f.write('alpha3={0}\n'.format(alpha3))
f.write('beta={0}\n\n'.format(beta))
f.write('Data parameters:\n')
f.write('v={0}\n'.format(vgt))
f.write('ell={0}\n'.format(ell))
f.write('tau={0}\n'.format(tau))
f.close()
