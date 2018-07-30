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
from dolfin import Constant
from dolfin import Function
from dolfin import UnitSquareMesh
from ofmc.model.of import of1d_exp
from ofmc.model.of import of1d_exp_pb
from ofmc.model.of import of1d_img
from ofmc.model.of import of1d_img_pb
from ofmc.model.of import of1d_weak_solution
from ofmc.model.of import of2dmcs
import unittest
import numpy as np
import ofmc.util.dolfinhelpers as dh


class TestOf(unittest.TestCase):

    def test_of1d_weak_solution_default(self):
        # Define temporal and spatial sample points.
        m, n = 10, 20

        # Define mesh and function space.
        mesh = UnitSquareMesh(m - 1, n - 1)
        V = dh.create_function_space(mesh, 'default')

        # Create zero function.
        f = Function(V)

        # Compute velocity.
        v, res, fun = of1d_weak_solution(V, f, f.dx(0), f.dx(1), 1.0, 1.0)
        v = v.vector().get_local()

        np.testing.assert_allclose(v.shape, m*n)
        np.testing.assert_allclose(v, np.zeros_like(v))

        V = dh.create_function_space(mesh, 'periodic')

        # Create zero function.
        f = Function(V)

        # Compute velocity.
        v, res, fun = of1d_weak_solution(V, f, f.dx(0), f.dx(1), 1.0, 1.0)
        v = v.vector().get_local()

        np.testing.assert_allclose(v.shape, m*(n - 1))
        np.testing.assert_allclose(v, np.zeros_like(v))

    def test_of1d_exp_default(self):
        # Define temporal and spatial sample points.
        m, n = 10, 20

        # Create constant image sequence.
        f = Constant(1.0)
        fd = Constant(0.0)

        # Compute velocity.
        v, res, fun = of1d_exp(m, n, f, fd, fd, 1.0, 1.0)

        np.testing.assert_allclose(v.shape, (m, n))
        np.testing.assert_allclose(v, np.zeros_like(v))

    def test_of1d_exp_pb(self):
        # Define temporal and spatial sample points.
        m, n = 10, 20

        # Create constant image sequence.
        f = Constant(1.0)
        fd = Constant(0.0)

        # Compute velocity.
        v, res, fun = of1d_exp_pb(m, n, f, fd, fd, 1.0, 1.0)

        np.testing.assert_allclose(v.shape, (m, n))
        np.testing.assert_allclose(v, np.zeros_like(v))

    def test_of1d_img_default_fd(self):
        # Create zero image.
        img = np.zeros((10, 25))
        v, res, fun = of1d_img(img, 1, 1, 'fd')

        np.testing.assert_allclose(v.shape, img.shape)
        np.testing.assert_allclose(v, np.zeros_like(v))

    def test_of1d_img_default_mesh(self):
        # Create zero image.
        img = np.zeros((10, 25))
        v, res, fun = of1d_img(img, 1, 1, 'mesh')

        np.testing.assert_allclose(v.shape, img.shape)
        np.testing.assert_allclose(v, np.zeros_like(v))
        
    def test_of1d_img_periodic_mesh(self):
        # Create zero image.
        img = np.zeros((10, 25))
        v, res, fun = of1d_img_pb(img, 1, 1, 'mesh')

        np.testing.assert_allclose(v.shape, img.shape)
        np.testing.assert_allclose(v, np.zeros_like(v))

    def test_of2dmcs(self):
        # Create zero images.
        img1 = np.zeros((5, 10, 25))
        img2 = np.zeros((5, 10, 25))
        v, k = of2dmcs(img1, img2, 1, 1, 1, 1)

        np.testing.assert_allclose(v.shape, (4, 10, 25, 2))
        np.testing.assert_allclose(k.shape, (4, 10, 25))
        np.testing.assert_allclose(v, np.zeros_like(v))
        np.testing.assert_allclose(k, np.zeros_like(k))


if __name__ == '__main__':
    unittest.main()
