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
from dolfin import dof_to_vertex_map
from dolfin import vertex_to_dof_map
import numpy as np


def img2funvec(img: np.array) -> np.array:
    """Takes a 2D array and returns an array suited to assign to piecewise
    linear approximation on a triangle grid.

    Each pixel corresponds to one vertex of a triangle mesh.

    Args:
        img (np.array): The input array.

    Returns:
        np.array: A vector.

    """
    # Create mesh.
    [m, n] = img.shape
    mesh = UnitSquareMesh(m, n)
    x = mesh.coordinates().reshape((-1, 2))

    # Evaluate function at vertices.
    hx, hy = 1./(m-1), 1./(n-1)
    x, y = np.array(x[:, 0]/hx, dtype=int), np.array(x[:, 1]/hy, dtype=int)
    fv = img[x, y]

    # Create function space.
    V = FunctionSpace(mesh, 'CG', 1)

    # Map pixel values to vertices.
    d2v = dof_to_vertex_map(V)
    return fv[d2v]


def funvec2img(v: np.array, m: int, n: int) -> np.array:
    """Takes values of piecewise linear interpolation of a function at the
    vertices and returns a 2-dimensional array.

    Each degree of freedom corresponds to one pixel in the array of
    size (m, n).

    Args:
        v (np.array): Values at vertices of triangle mesh.
        m (int): The number of rows.
        n (int): The number of columns.

    Returns:
        np.array: An array of size (m, n).

    """
    # Create image.
    img = np.zeros((m, n))

    # Create mesh and function space.
    mesh = UnitSquareMesh(m, n)
    x = mesh.coordinates().reshape((-1, 2))

    # Evaluate function at vertices.
    hx, hy = 1./(m-1), 1./(n-1)
    x, y = np.array(x[:, 0]/hx, dtype=int), np.array(x[:, 1]/hy, dtype=int)

    # Create function space and function.
    V = FunctionSpace(mesh, 'CG', 1)

    # Create image from function.
    v2d = vertex_to_dof_map(V)
    values = v[v2d]
    for (i, j, v) in zip(x, y, values):
        img[i, j] = v
    return img
