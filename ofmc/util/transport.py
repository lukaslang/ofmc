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
from dolfin import action
from dolfin import as_vector
from dolfin import assemble
from dolfin import dot
from dolfin import ds
from dolfin import dS
from dolfin import dx
from dolfin import FacetNormal
from dolfin import Function
from dolfin import FunctionSpace
from dolfin import interpolate
from dolfin import jump
from dolfin import project
from dolfin import solve
from dolfin import TestFunction
from dolfin import TrialFunction
from dolfin import UnitIntervalMesh
import numpy as np


def transport1d(vel: np.array, source: np.array, finit: np.array) -> np.array:
    """Computes the 1D transport via continuity equation.

    Takes an array v of velocities, a source array, and initial data f0, and
    computes a solution to the continuity equation via discontinuous galerkin
    method with third order Runge-Kutta method.

    Args:
        vel (np.array): A 2D velocity field of size (m, n)
        source (np.array): A 2D source of size (m, n)
        finit (np.array): A vector of length n with initial data.

    Returns:
        np.array: A 2D array of size (m + 1, n) with the transport.

    """
    # Define mesh.
    m, n = vel.shape
    mesh = UnitIntervalMesh(n - 1)

    # Define function spaces
    V = FunctionSpace(mesh, 'DG', 2)
    W = FunctionSpace(mesh, 'CG', 1)

    # Define mesh-related functions.
    fn = FacetNormal(mesh)
    h = mesh.hmin()

    # Define trial and test function.
    f = TrialFunction(V)
    w = TestFunction(V)

    # Define functions.
    df1 = Function(V)
    f1 = Function(V)
    df2 = Function(V)
    f2 = Function(V)
    df3 = Function(V)

    # Initialise solution.
    f0 = Function(W)
    f0.vector()[:] = finit
    f0 = interpolate(f0, V)
    initial_mass = assemble(f0*dx)

    # Define start time and end time.
    t = 0.0
    T = 1.0

    # Define iteration number and dump number.
    k = 0
    d = 1

    # Define time step size based on CFL condition.
    maxvel = np.amax(np.abs(vel))
    maxvel = maxvel if maxvel > 0 else 1
    dt = 0.95 * 0.209 * h / abs(maxvel)
    print('Size of time step: {0}'.format(dt))

    # Compute dump frequency.
    dumpfreq = max(np.int(T / dt / m), 1)

    # Create solution array.
    fsol = np.zeros((m + 1, n))
    fsol[0, :] = project(f0, W).vector().get_local()

    while(d <= m):
        # Create velocity function.
        v = Function(W)
        v.vector()[:] = -vel[d - 1, :]
        v = project(v, V)

        # Create source function.
        src = Function(W)
        src.vector()[:] = -source[d - 1, :]
        src = project(src, V)

        # Define non-negative dot product with normal.
        vv = as_vector((v, ))
        vn = 0.5 * (dot(vv, fn) + abs(dot(vv, fn)))

        # Define bilinear form.
        a_mass = f*w*dx
        a_int = -w.dx(0)*f*v*dx
        a_flux = jump(w)*(vn('+')*f('+') - vn('-')*f('-'))*dS + w*vn*f*ds
        a_source = src*w*dx

        # Assemble mass matrix.
        M = assemble(a_mass)

        # Define right-hand side.
        rhs = -dt*(a_int + a_flux + a_source)

        L = assemble(action(rhs, f0))
        solve(M, df1.vector(), L)

        f1.vector()[:] = f0.vector().copy()
        f1.vector().axpy(1.0, df1.vector())
        L = assemble(action(rhs, f1))
        solve(M, df2.vector(), L)

        f2.vector()[:] = f0.vector().copy()
        f2.vector().axpy(0.25, df1.vector())
        f2.vector().axpy(0.25, df2.vector())
        L = assemble(action(rhs, f2))
        solve(M, df3.vector(), L)

        f0.vector().axpy((1.0/6.0), df1.vector())
        f0.vector().axpy((1.0/6.0), df2.vector())
        f0.vector().axpy((2.0/3.0), df3.vector())

        if(k % dumpfreq == 0 and k > 0):
            print('Iteration {0}, dump {1}'.format(k, d))
            fsol[d, :] = project(f0, W).vector().get_local()
            d += 1

        t += dt
        k += 1

    # Compute error.
    conservation_error = assemble(f0*dx) - initial_mass
    print('Conservation error: {0}'.format(conservation_error))

    return fsol
