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
from dolfin import assemble
from dolfin import dx
from dolfin import solve
from dolfin import project
from dolfin import Expression
from dolfin import Function
from dolfin import lhs
from dolfin import rhs
from dolfin import TrialFunctions
from dolfin import TrialFunction
from dolfin import TestFunction
from dolfin import TestFunctions
from dolfin import FunctionSpace
from dolfin import UnitSquareMesh
from dolfin import VectorFunctionSpace
import ofmc.util.dolfinhelpers as dh
import ofmc.util.numpyhelpers as nh

# Note that for the periodic case v and k should be of shape (m, n-1).
# There seems to be a problem with the handling of periodic vector-valued
# spaces, see:
# https://bitbucket.org/fenics-project/dolfin/issues/405
# https://fenicsproject.org/qa/6462/split-mixed-spaces-does-not-preserve-number-degrees-freedom/


def mpi1d_weak_solution(V: FunctionSpace,
                        f: Function,
                        ft: Function, fx: Function) -> (Function, Function):

    # Define trial and test functions.
    v = TrialFunction(V)
    w = TestFunction(V)

    # Define weak formulation.
    A = (v.dx(1) * w.dx(1) + v * w
         + (fx * v + f * v.dx(1)) * (fx * w + f * w.dx(1))) * dx
    b = -ft * (fx * w + f * w.dx(1)) * dx

#    F = (v.dx(1) * w.dx(1) + v * w
#         + (fx * v + f * v.dx(1) + ft) * (fx * w + f * w.dx(1))) * dx

 #   A = lhs(F)
 #   b = rhs(F)

    # Compute solution.
    v = Function(V)
    solve(A == b, v)

    # Compute k.
    k = project(fx * v + f * v.dx(1) + ft, V)

    return v, k


def mpi1d_exp_pb(m: int, n: int,
                 f: Expression,
                 ft: Expression, fx: Expression) -> (np.array, np.array):

    # Define mesh and function space.
    mesh = UnitSquareMesh(m - 1, n - 1)
    V = dh.create_function_space(mesh, 'periodic')

    # Compute velocity.
    v, k = mpi1d_weak_solution(V, f, ft, fx)

    # Convert back to array and return.
    v = dh.funvec2img_pb(v.vector().get_local(), m, n)
    k = dh.funvec2img_pb(k.vector().get_local(), m, n)
    return v, k
