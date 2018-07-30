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
import numpy.matlib as matlib
from scipy import ndimage
import imageio
import unittest
import numpy as np
from dolfin import Constant
from dolfin import Function
from dolfin import UnitSquareMesh
from ofmc.model.cmscr import cmscr1d_weak_solution
from ofmc.model.cmscr import cmscr1d_exp
from ofmc.model.cmscr import cmscr1d_exp_pb
from ofmc.model.cmscr import cmscr1d_img
from ofmc.model.cmscr import cmscr1d_img_pb
from ofmc.model.cmscr import cmscr1dnewton
import ofmc.util.dolfinhelpers as dh


class TestCmscr(unittest.TestCase):

    def test_cmscr1d_weak_solution_default(self):
        print("Running test 'test_cmscr1d_weak_solution_default'")
        # Define temporal and spatial sample points.
        m, n = 10, 20

        # Define mesh and function space.
        mesh = UnitSquareMesh(m - 1, n - 1)
        V = dh.create_function_space(mesh, 'default')
        W = dh.create_vector_function_space(mesh, 'default')

        # Create zero function.
        f = Function(V)

        # Compute velocity.
        v, k, res, fun, converged = cmscr1d_weak_solution(W, f,
                                                          f.dx(0), f.dx(1),
                                                          1.0, 1.0,
                                                          1.0, 1.0, 1.0)
        v = v.vector().get_local()
        k = k.vector().get_local()

        np.testing.assert_allclose(v.shape, m * n)
        np.testing.assert_allclose(v, np.zeros_like(v))
        np.testing.assert_allclose(k.shape, m * n)
        np.testing.assert_allclose(k, np.zeros_like(k))

        V = dh.create_function_space(mesh, 'periodic')

        # Create zero function.
        f = Function(V)

        # Compute velocity.
        v, k, res, fun, converged = cmscr1d_weak_solution(W, f,
                                                          f.dx(0), f.dx(1),
                                                          1.0, 1.0, 1.0,
                                                          1.0, 1.0)
        v = v.vector().get_local()
        k = k.vector().get_local()

        np.testing.assert_allclose(v.shape, m * n)
        np.testing.assert_allclose(v, np.zeros_like(v))
        np.testing.assert_allclose(k.shape, m * n)
        np.testing.assert_allclose(k, np.zeros_like(k))

    def test_cmscr1d_exp_default(self):
        print("Running test 'test_cmscr1d_exp_default'")
        # Define temporal and spatial sample points.
        m, n = 10, 20

        # Create constant image sequence.
        f = Constant(1.0)
        fd = Constant(0.0)

        # Compute velocity.
        v, k, res, fun, converged = cmscr1d_exp(m, n, f, fd, fd,
                                                1.0, 1.0, 1.0, 1.0, 1.0)

        np.testing.assert_allclose(v.shape, (m, n))
        np.testing.assert_allclose(v, np.zeros_like(v))
        np.testing.assert_allclose(k.shape, (m, n))
        np.testing.assert_allclose(k, np.zeros_like(k))

    def test_cmscr1d_exp_pb(self):
        print("Running test 'test_cmscr1d_exp_pb'")
        # Define temporal and spatial sample points.
        m, n = 10, 20

        # Create constant image sequence.
        f = Constant(1.0)
        fd = Constant(0.0)

        # Compute velocity.
        v, k, res, fun, converged = cmscr1d_exp_pb(m, n, f, fd, fd,
                                                   1.0, 1.0, 1.0, 1.0, 1.0)

        np.testing.assert_allclose(v.shape, (m, n))
        np.testing.assert_allclose(v, np.zeros_like(v))
        np.testing.assert_allclose(k.shape, (m, n))
        np.testing.assert_allclose(k, np.zeros_like(k))

    def test_cmscr1d_img_default_fd(self):
        print("Running test 'test_cmscr1d_img_default_fd'")
        # Create zero image.
        img = np.zeros((10, 25))
        v, k, res, fun, converged = cmscr1d_img(img,
                                                1.0, 1.0, 1.0, 1.0, 1.0, 'fd')

        np.testing.assert_allclose(v.shape, img.shape)
        np.testing.assert_allclose(v, np.zeros_like(v))
        np.testing.assert_allclose(k.shape, img.shape)
        np.testing.assert_allclose(k, np.zeros_like(k))

    def test_cmscr1d_img_default_mesh(self):
        print("Running test 'test_cmscr1d_img_default_mesh'")
        # Create zero image.
        img = np.zeros((10, 25))
        v, k, res, fun, converged = cmscr1d_img(img, 1.0, 1.0, 1.0, 1.0,
                                                1.0, 'mesh')

        np.testing.assert_allclose(v.shape, img.shape)
        np.testing.assert_allclose(v, np.zeros_like(v))
        np.testing.assert_allclose(k.shape, img.shape)
        np.testing.assert_allclose(k, np.zeros_like(k))

    def test_cmscr1d_img_periodic_mesh(self):
        print("Running test 'test_cmscr1d_img_periodic_mesh'")
        # Create zero image.
        img = np.zeros((10, 25))
        v, k, res, fun, converged = cmscr1d_img_pb(img, 1.0, 1.0, 1.0, 1.0,
                                                   1.0, 'mesh')

        np.testing.assert_allclose(v.shape, img.shape)
        np.testing.assert_allclose(v, np.zeros_like(v))
        np.testing.assert_allclose(k.shape, img.shape)
        np.testing.assert_allclose(k, np.zeros_like(k))

    def test_cmscr1dnewton(self):
        print("Running test 'test_cmscr1dnewton'")
        # Create zero image.
        img = np.zeros((10, 25))
        v, k, res, fun, converged = cmscr1dnewton(img, 1, 1, 1, 1, 1)

        np.testing.assert_allclose(v.shape, img.shape)
        np.testing.assert_allclose(v, np.zeros_like(v))
        np.testing.assert_allclose(k.shape, img.shape)
        np.testing.assert_allclose(k, np.zeros_like(k))

    def test_cmscr1dnewton_random(self):
        print("Running test 'test_cmscr1dnewton_random'")
        # Create random non-moving image.
        img = matlib.repmat(np.random.rand(1, 25), 10, 1)
        v, k, res, fun, converged = cmscr1dnewton(img, 1, 1, 1, 1, 1)

        np.testing.assert_allclose(v.shape, img.shape)
        np.testing.assert_allclose(v, np.zeros_like(v))
        np.testing.assert_allclose(k.shape, img.shape)
        np.testing.assert_allclose(k, np.zeros_like(k))

    def test_cmscr1dnewton_data(self):
        print("Running test 'test_cmscr1dnewton_data'")
        # Load test image.
        name = 'ofmc/test/data/DynamicReslice of E2PSB1PMT_10px.tif'
        img = imageio.imread(name)

        # Remove cut.
        img = np.vstack((img[0:4, :], img[6:, :]))

        # Filter image.
        img = ndimage.gaussian_filter(img, sigma=1)

        # Normalise to [0, 1].
        img = np.array(img, dtype=float)
        img = (img - img.min()) / (img.max() - img.min())

        # Compute solution.
        v, k, res, fun, converged = cmscr1dnewton(img, 1, 1, 1, 1, 1)

        np.testing.assert_allclose(v.shape, img.shape)
        np.testing.assert_allclose(k.shape, img.shape)


if __name__ == '__main__':
    unittest.main()
