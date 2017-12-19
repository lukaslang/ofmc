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


def of1d(img: np.array, alpha0: float, alpha1: float) -> np.array:
    """Computes the L2-H1 optical flow for a 1D image sequence.

    Takes a one-dimensional image sequence and returns a minimiser of the
    Horn-Schunck functional with spatio-temporal regularisation.

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
    fx = f.dx(1)
    ft = f.dx(0)
    fxft = ft*fx
    fxfx = fx*fx

    # Define weak formulation.
    A = fxfx*v*w*dx + alpha0*v.dx(1)*w.dx(1)*dx + alpha1*v.dx(0)*w.dx(0)*dx
    b = -fxft*w*dx

    # Compute solution.
    v = Function(V)
    solve(A == b, v)

    vel = dh.funvec2img(v.vector().array(), m, n)
    return vel
