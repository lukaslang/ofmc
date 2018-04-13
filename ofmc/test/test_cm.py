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
import unittest
import numpy as np
import numpy.matlib as matlib
from dolfin import Constant
from dolfin import Function
from dolfin import UnitSquareMesh
from ofmc.model.cm import cm1d_weak_solution
from ofmc.model.cm import cm1d_exp
from ofmc.model.cm import cm1d_exp_pb
from ofmc.model.cm import cm1d_img
from ofmc.model.cm import cm1d_img_pb
from ofmc.model.cm import cm1dsource
from ofmc.model.cm import cm1dvelocity
import ofmc.util.dolfinhelpers as dh


class TestCm(unittest.TestCase):

    def test_cm1d_weak_solution_default(self):
        # Define temporal and spatial sample points.
        m, n = 10, 20

        # Define mesh and function space.
        mesh = UnitSquareMesh(m - 1, n - 1)
        V = dh.create_function_space(mesh, 'default')

        # Create zero function.
        f = Function(V)

        # Compute velocity.
        v = cm1d_weak_solution(V, f, f.dx(0), f.dx(1), 1.0, 1.0)
        v = v.vector().get_local()

        np.testing.assert_allclose(v.shape, m*n)
        np.testing.assert_allclose(v, np.zeros_like(v))

        V = dh.create_function_space(mesh, 'periodic')

        # Create zero function.
        f = Function(V)

        # Compute velocity.
        v = cm1d_weak_solution(V, f, f.dx(0), f.dx(1), 1.0, 1.0)
        v = v.vector().get_local()

        np.testing.assert_allclose(v.shape, m*(n - 1))
        np.testing.assert_allclose(v, np.zeros_like(v))

    def test_cm1d_exp_default(self):
        # Define temporal and spatial sample points.
        m, n = 10, 20

        # Create constant image sequence.
        f = Constant(1.0)
        fd = Constant(0.0)

        # Compute velocity.
        v = cm1d_exp(m, n, f, fd, fd, 1.0, 1.0)

        np.testing.assert_allclose(v.shape, (m, n))
        np.testing.assert_allclose(v, np.zeros_like(v))

    def test_cm1d_exp_pb(self):
        # Define temporal and spatial sample points.
        m, n = 10, 20

        # Create constant image sequence.
        f = Constant(1.0)
        fd = Constant(0.0)

        # Compute velocity.
        v = cm1d_exp_pb(m, n, f, fd, fd, 1.0, 1.0)

        np.testing.assert_allclose(v.shape, (m, n))
        np.testing.assert_allclose(v, np.zeros_like(v))

    def test_cm1d_img_default_fd(self):
        # Create zero image.
        img = np.zeros((10, 25))
        v = cm1d_img(img, 1, 1, 'fd')

        np.testing.assert_allclose(v.shape, img.shape)
        np.testing.assert_allclose(v, np.zeros_like(v))

    def test_cm1d_img_default_mesh(self):
        # Create zero image.
        img = np.zeros((10, 25))
        v = cm1d_img(img, 1, 1, 'mesh')

        np.testing.assert_allclose(v.shape, img.shape)
        np.testing.assert_allclose(v, np.zeros_like(v))

    def test_cm1d_img_periodic_mesh(self):
        # Create zero image.
        img = np.zeros((10, 25))
        v = cm1d_img_pb(img, 1, 1, 'mesh')

        np.testing.assert_allclose(v.shape, img.shape)
        np.testing.assert_allclose(v, np.zeros_like(v))

    def test_cm1dsource(self):
        # Create zero image.
        img = np.zeros((10, 25))
        v = cm1dsource(img, img, 1, 1)

        np.testing.assert_allclose(v.shape, img.shape)
        np.testing.assert_allclose(v, np.zeros_like(v))

    def test_cm1dvelocity(self):
        # Create zero image.
        img = np.zeros((10, 25))
        k = cm1dvelocity(img, img, 1, 1)

        np.testing.assert_allclose(k.shape, img.shape)
        np.testing.assert_allclose(k, np.zeros_like(k))

    def test_cm1d_random(self):
        # Create random non-moving image.
        img = matlib.repmat(np.random.rand(1, 25), 10, 1)
        v = cm1d_img(img, 1, 1, 'mesh')

        np.testing.assert_allclose(v.shape, img.shape)
        np.testing.assert_allclose(v, np.zeros_like(v))


if __name__ == '__main__':
    unittest.main()
