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
import numpy as np
from dolfin import dx
from dolfin import solve
from dolfin import split
from dolfin import Function
from dolfin import VectorFunctionSpace
from dolfin import TestFunctions
from dolfin import FunctionSpace
from dolfin import UnitSquareMesh
import ofmc.util.dolfinhelpers as dh


def cmscr1d(img: np.array, alpha0: float, alpha1: float,
            alpha2: float, alpha3: float, beta: float) -> (np.array, np.array):
    """Computes the L2-H1-H1 mass conserving flow with source and convective
    regularisation for a 1D image sequence.

    Takes a one-dimensional image sequence and returns a minimiser of the
    L2-H1-H1 mass conservation functional with source term with spatio-temporal
    regularisation for the velocity and for the source, and convective
    regularisation.

    Args:
        img (np.array): A 1D image sequence of shape (m, n), where m is the
                        number of time steps and n is the number of pixels.
        alpha0 (float): The spatial regularisation parameter for v.
        alpha1 (float): The temporal regularisation parameter for v.
        alpha2 (float): The spatial regularisation parameter for k.
        alpha3 (float): The temporal regularisation parameter for k.
        beta (float): The parameter for convective regularisation.

    Returns:
        v: A velocity array of shape (m, n).
        k: A source array of shape (m, n).

    """
    # Create mesh.
    [m, n] = img.shape
    mesh = UnitSquareMesh(m, n)

    # Define function space and functions.
    W = VectorFunctionSpace(mesh, 'CG', 1, dim=2)
    w = Function(W)
    v, k = split(w)
    w1, w2 = TestFunctions(W)

    # Convert image to function.
    V = FunctionSpace(mesh, 'CG', 1)
    f = Function(V)
    f.vector()[:] = dh.img2funvec(img)

    # Define derivatives of data.
    ft = f.dx(0)
    fx = f.dx(1)

    # Define weak formulation.
    A = - (ft + fx*v + f*v.dx(1) - k) * (fx*w1 + f*w1.dx(1))*dx \
        - alpha0*v.dx(1)*w1.dx(1)*dx - alpha1*v.dx(0)*w1.dx(0)*dx \
        - beta*k.dx(1)*(k.dx(0) + k.dx(1)*v)*w1*dx \
        + (ft + fx*v + f*v.dx(1) - k)*w2*dx \
        - alpha2*k.dx(1)*w2.dx(1)*dx - alpha3*k.dx(0)*w2.dx(0)*dx \
        - beta*(k.dx(0)*w2.dx(0) + k.dx(1)*v*w2.dx(0)
                + k.dx(0)*v*w2.dx(1) + k.dx(1)*v*v*w2.dx(1))*dx

    # Compute solution via Newton method.
    solve(A == 0, w)

    # Recover solution.
    v, k = w.split(deepcopy=True)

    # Convert back to array.
    vel = dh.funvec2img(v.vector().array(), m, n)
    k = dh.funvec2img(k.vector().array(), m, n)
    return vel, k
