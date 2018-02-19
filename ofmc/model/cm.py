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
from dolfin import Function
from dolfin import TrialFunction
from dolfin import TestFunction
from dolfin import FunctionSpace
from dolfin import UnitSquareMesh
import ofmc.util.dolfinhelpers as dh


def cm1d(img: np.array, alpha0: float, alpha1: float) -> np.array:
    """Computes the L2-H1 mass conserving flow for a 1D image sequence.

    Takes a one-dimensional image sequence and returns a minimiser of the
    L2-H1 mass conservation functional with spatio-temporal regularisation.

    Args:
        img (np.array): A 1D image sequence of shape (m, n), where m is the
                        number of time steps and n is the number of pixels.
        alpha0 (float): The spatial regularisation parameter.
        alpha1 (float): The temporal regularisation parameter.

    Returns:
        v: A velocity array of shape (m, n).

    """
    # Create mesh.
    [m, n] = img.shape
    mesh = UnitSquareMesh(m, n)

    # Define function space and functions.
    V = FunctionSpace(mesh, 'CG', 1)
    v = TrialFunction(V)
    w = TestFunction(V)

    # Convert image to function.
    f = Function(V)
    f.vector()[:] = dh.img2funvec(img)

    # Define derivatives of data.
    ft = Function(V)
    ftv = np.diff(img, axis=0) * (m - 1)
    ftv = np.concatenate((ftv, ftv[-1, :].reshape(1, n)), axis=0)
    ft.vector()[:] = dh.img2funvec(ftv)

    fx = Function(V)
    fxv = np.gradient(img, 1 / (n - 1), axis=1)
    fx.vector()[:] = dh.img2funvec(fxv)

    # Define weak formulation.
    A = - (fx*v + f*v.dx(1)) * (fx*w + f*w.dx(1))*dx \
        - alpha0*v.dx(1)*w.dx(1)*dx - alpha1*v.dx(0)*w.dx(0)*dx
    b = ft*(fx*w + f*w.dx(1))*dx

    # Compute solution.
    v = Function(V)
    solve(A == b, v)

    # Convert back to array.
    vel = dh.funvec2img(v.vector().array(), m, n)
    return vel
