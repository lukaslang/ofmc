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
from dolfin import UnitSquareMesh
from dolfin import FunctionSpace
from dolfin import Function
from dolfin import dof_to_vertex_map
from dolfin import vertex_to_dof_map
import numpy as np


def img2fun(img: np.array) -> Function:
    """Takes a 2D array and returns a piecewise linear interpolation.

    Each pixel corresponds to one vertex of a triangle mesh.

    Args:
        img (array): The input array.

    Returns:
        Function: A dolfin function.

    """
    # Create mesh.
    [m, n] = img.shape
    mesh = UnitSquareMesh(m-1, n-1)
    x = mesh.coordinates().reshape((-1, 2))

    # Evaluate function at vertices.
    hx, hy = 1./(m-1), 1./(n-1)
    x, y = np.array(x[:, 0]/hx, dtype=int), np.array(x[:, 1]/hy, dtype=int)
    fv = img[x, y]

    # Create function space and function.
    V = FunctionSpace(mesh, 'CG', 1)
    f = Function(V)

    # Map pixel values to vertices.
    d2v = dof_to_vertex_map(V)
    f.vector()[:] = fv[d2v]
    return f


def fun2img(f: Function, m: int, n: int) -> np.array:
    """Takes piecewise linear interpolation function and returns an array.

    Each degree of freedom corresponds to one pixel in the array of
    size (m, n).

    Args:
        f (Function): The piecewise linear function.
        m (int): The number of rows.
        n (int): The number of columns.

    Returns:
        np.array: An array of size (m, n).

    """
    # Create image.
    img = np.zeros((m, n))

    # Create mesh and function space.
    mesh = UnitSquareMesh(m-1, n-1)
    x = mesh.coordinates().reshape((-1, 2))

    # Evaluate function at vertices.
    hx, hy = 1./(m-1), 1./(n-1)
    x, y = np.array(x[:, 0]/hx, dtype=int), np.array(x[:, 1]/hy, dtype=int)

    # Create function space and function.
    V = FunctionSpace(mesh, 'CG', 1)

    # Create image from function.
    v2d = vertex_to_dof_map(V)
    values = f.vector().array()[v2d]
    for (i, j, v) in zip(x, y, values):
        img[i, j] = v
    return img
