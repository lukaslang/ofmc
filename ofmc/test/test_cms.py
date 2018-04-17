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
from dolfin import Constant
from dolfin import Function
from dolfin import UnitSquareMesh
from ofmc.model.cms import cms1d_weak_solution
from ofmc.model.cms import cms1d_weak_solution_given_source
from ofmc.model.cms import cms1d_weak_solution_given_velocity
from ofmc.model.cms import cms1dl2_weak_solution
from ofmc.model.cms import cms1d_exp
from ofmc.model.cms import cms1d_exp_pb
from ofmc.model.cms import cms1dl2_exp
from ofmc.model.cms import cms1dl2_exp_pb
from ofmc.model.cms import cms1d_given_source_exp
from ofmc.model.cms import cms1d_given_source_exp_pb
from ofmc.model.cms import cms1d_given_velocity_exp
from ofmc.model.cms import cms1d_given_velocity_exp_pb
from ofmc.model.cms import cms1d_img
from ofmc.model.cms import cms1d_img_pb
from ofmc.model.cms import cms1dl2_img
from ofmc.model.cms import cms1dl2_img_pb
import ofmc.util.dolfinhelpers as dh


class TestCms(unittest.TestCase):

    def test_cms1d_weak_solution_default(self):
        # Define temporal and spatial sample points.
        m, n = 10, 20

        # Define mesh and function space.
        mesh = UnitSquareMesh(m - 1, n - 1)
        V = dh.create_function_space(mesh, 'default')
        W = dh.create_vector_function_space(mesh, 'default')

        # Create zero function.
        f = Function(V)

        # Compute velocity.
        v, k = cms1d_weak_solution(W, f, f.dx(0), f.dx(1), 1.0, 1.0, 1.0, 1.0)
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
        v, k = cms1d_weak_solution(W, f, f.dx(0), f.dx(1), 1.0, 1.0, 1.0, 1.0)
        v = v.vector().get_local()
        k = k.vector().get_local()

        np.testing.assert_allclose(v.shape, m * n)
        np.testing.assert_allclose(v, np.zeros_like(v))
        np.testing.assert_allclose(k.shape, m * n)
        np.testing.assert_allclose(k, np.zeros_like(k))

    def test_cms1dl2_weak_solution_default(self):
        # Define temporal and spatial sample points.
        m, n = 10, 20

        # Define mesh and function space.
        mesh = UnitSquareMesh(m - 1, n - 1)
        V = dh.create_function_space(mesh, 'default')
        W = dh.create_vector_function_space(mesh, 'default')

        # Create zero function.
        f = Function(V)

        # Compute velocity.
        v, k = cms1dl2_weak_solution(W, f, f.dx(0), f.dx(1),
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
        v, k = cms1dl2_weak_solution(W, f, f.dx(0), f.dx(1),
                                     1.0, 1.0, 1.0)
        v = v.vector().get_local()
        k = k.vector().get_local()

        np.testing.assert_allclose(v.shape, m * n)
        np.testing.assert_allclose(v, np.zeros_like(v))
        np.testing.assert_allclose(k.shape, m * n)
        np.testing.assert_allclose(k, np.zeros_like(k))

    def test_cms1d_weak_solution_given_source_default(self):
        # Define temporal and spatial sample points.
        m, n = 10, 20

        # Define mesh and function space.
        mesh = UnitSquareMesh(m - 1, n - 1)
        V = dh.create_function_space(mesh, 'default')

        # Create zero function.
        f = Function(V)

        # Create source.
        k = Function(V)

        # Compute velocity.
        v = cms1d_weak_solution_given_source(V, f, f.dx(0), f.dx(1), k,
                                             1.0, 1.0)
        v = v.vector().get_local()

        np.testing.assert_allclose(v.shape, m * n)
        np.testing.assert_allclose(v, np.zeros_like(v))

        V = dh.create_function_space(mesh, 'periodic')

        # Create zero function.
        f = Function(V)

        # Compute velocity.
        v = cms1d_weak_solution_given_source(V, f, f.dx(0), f.dx(1), k,
                                             1.0, 1.0)
        v = v.vector().get_local()

        np.testing.assert_allclose(v.shape, m * (n - 1))
        np.testing.assert_allclose(v, np.zeros_like(v))

    def test_cms1d_weak_solution_given_velocity_default(self):
        # Define temporal and spatial sample points.
        m, n = 10, 20

        # Define mesh and function space.
        mesh = UnitSquareMesh(m - 1, n - 1)
        V = dh.create_function_space(mesh, 'default')

        # Create zero function.
        f = Function(V)

        # Create velocity.
        v = Function(V)
        vx = Function(V)

        # Compute source.
        k = cms1d_weak_solution_given_velocity(V, f, f.dx(0), f.dx(1), v, vx,
                                               1.0, 1.0)
        k = k.vector().get_local()

        np.testing.assert_allclose(k.shape, m * n)
        np.testing.assert_allclose(k, np.zeros_like(k))

        V = dh.create_function_space(mesh, 'periodic')

        # Create zero function.
        f = Function(V)

        # Compute source.
        k = cms1d_weak_solution_given_velocity(V, f, f.dx(0), f.dx(1), v, vx,
                                               1.0, 1.0)
        k = k.vector().get_local()

        np.testing.assert_allclose(k.shape, m * (n - 1))
        np.testing.assert_allclose(k, np.zeros_like(k))

    def test_cms1d_exp_default(self):
        # Define temporal and spatial sample points.
        m, n = 10, 20

        # Create constant image sequence.
        f = Constant(1.0)
        fd = Constant(0.0)

        # Compute velocity.
        v, k = cms1d_exp(m, n, f, fd, fd, 1.0, 1.0, 1.0, 1.0)

        np.testing.assert_allclose(v.shape, (m, n))
        np.testing.assert_allclose(v, np.zeros_like(v))
        np.testing.assert_allclose(k.shape, (m, n))
        np.testing.assert_allclose(k, np.zeros_like(k))

    def test_cms1d_exp_pb(self):
        # Define temporal and spatial sample points.
        m, n = 10, 20

        # Create constant image sequence.
        f = Constant(1.0)
        fd = Constant(0.0)

        # Compute velocity.
        v, k = cms1d_exp_pb(m, n, f, fd, fd, 1.0, 1.0, 1.0, 1.0)

        np.testing.assert_allclose(v.shape, (m, n))
        np.testing.assert_allclose(v, np.zeros_like(v))
        np.testing.assert_allclose(k.shape, (m, n))
        np.testing.assert_allclose(k, np.zeros_like(k))

    def test_cms1dl2_exp_default(self):
        # Define temporal and spatial sample points.
        m, n = 10, 20

        # Create constant image sequence.
        f = Constant(1.0)
        fd = Constant(0.0)

        # Compute velocity.
        v, k = cms1dl2_exp(m, n, f, fd, fd, 1.0, 1.0, 1.0)

        np.testing.assert_allclose(v.shape, (m, n))
        np.testing.assert_allclose(v, np.zeros_like(v))
        np.testing.assert_allclose(k.shape, (m, n))
        np.testing.assert_allclose(k, np.zeros_like(k))

    def test_cms1dl2_exp_pb(self):
        # Define temporal and spatial sample points.
        m, n = 10, 20

        # Create constant image sequence.
        f = Constant(1.0)
        fd = Constant(0.0)

        # Compute velocity.
        v, k = cms1dl2_exp_pb(m, n, f, fd, fd, 1.0, 1.0, 1.0)

        np.testing.assert_allclose(v.shape, (m, n))
        np.testing.assert_allclose(v, np.zeros_like(v))
        np.testing.assert_allclose(k.shape, (m, n))
        np.testing.assert_allclose(k, np.zeros_like(k))

    def test_cms1d_given_source_exp_default(self):
        # Define temporal and spatial sample points.
        m, n = 10, 20

        # Create constant image sequence.
        f = Constant(1.0)
        fd = Constant(0.0)

        # Create source.
        k = Constant(0.0)

        # Compute velocity.
        v = cms1d_given_source_exp(m, n, f, fd, fd, k, 1.0, 1.0)

        np.testing.assert_allclose(v.shape, (m, n))
        np.testing.assert_allclose(v, np.zeros_like(v))

    def test_cms1d_given_source_exp_pb(self):
        # Define temporal and spatial sample points.
        m, n = 10, 20

        # Create constant image sequence.
        f = Constant(1.0)
        fd = Constant(0.0)

        # Create source.
        k = Constant(0.0)

        # Compute velocity.
        v = cms1d_given_source_exp_pb(m, n, f, fd, fd, k, 1.0, 1.0)

        np.testing.assert_allclose(v.shape, (m, n))
        np.testing.assert_allclose(v, np.zeros_like(v))

    def test_cms1d_given_velocity_exp_default(self):
        # Define temporal and spatial sample points.
        m, n = 10, 20

        # Create constant image sequence.
        f = Constant(1.0)
        fd = Constant(0.0)

        # Create velocity.
        v = Constant(0.0)
        vx = Constant(0.0)

        # Compute source.
        k = cms1d_given_velocity_exp(m, n, f, fd, fd, v, vx, 1.0, 1.0)

        np.testing.assert_allclose(k.shape, (m, n))
        np.testing.assert_allclose(k, np.zeros_like(k))

    def test_cms1d_given_velocity_exp_pb(self):
        # Define temporal and spatial sample points.
        m, n = 10, 20

        # Create constant image sequence.
        f = Constant(1.0)
        fd = Constant(0.0)

        # Create velocity.
        v = Constant(0.0)
        vx = Constant(0.0)

        # Compute velocity.
        k = cms1d_given_velocity_exp_pb(m, n, f, fd, fd, v, vx, 1.0, 1.0)

        np.testing.assert_allclose(k.shape, (m, n))
        np.testing.assert_allclose(k, np.zeros_like(k))

    def test_cms1d_img_default_fd(self):
        # Create zero image.
        img = np.zeros((10, 25))
        v, k = cms1d_img(img, 1.0, 1.0, 1.0, 1.0, 'fd')

        np.testing.assert_allclose(v.shape, img.shape)
        np.testing.assert_allclose(v, np.zeros_like(v))
        np.testing.assert_allclose(k.shape, img.shape)
        np.testing.assert_allclose(k, np.zeros_like(k))

    def test_cms1d_img_default_mesh(self):
        # Create zero image.
        img = np.zeros((10, 25))
        v, k = cms1d_img(img, 1.0, 1.0, 1.0, 1.0, 'mesh')

        np.testing.assert_allclose(v.shape, img.shape)
        np.testing.assert_allclose(v, np.zeros_like(v))
        np.testing.assert_allclose(k.shape, img.shape)
        np.testing.assert_allclose(k, np.zeros_like(k))

    def test_cms1d_img_periodic_mesh(self):
        # Create zero image.
        img = np.zeros((10, 25))
        v, k = cms1d_img_pb(img, 1.0, 1.0, 1.0, 1.0, 'mesh')

        np.testing.assert_allclose(v.shape, img.shape)
        np.testing.assert_allclose(v, np.zeros_like(v))
        np.testing.assert_allclose(k.shape, img.shape)
        np.testing.assert_allclose(k, np.zeros_like(k))

    def test_cms1dl2_img_default_mesh(self):
        # Create zero image.
        img = np.zeros((10, 25))
        v, k = cms1dl2_img(img, 1.0, 1.0, 1.0, 'mesh')

        np.testing.assert_allclose(v.shape, img.shape)
        np.testing.assert_allclose(v, np.zeros_like(v))
        np.testing.assert_allclose(k.shape, img.shape)
        np.testing.assert_allclose(k, np.zeros_like(k))

    def test_cms1dl2_img_periodic_mesh(self):
        # Create zero image.
        img = np.zeros((10, 25))
        v, k = cms1dl2_img_pb(img, 1.0, 1.0, 1.0, 'mesh')

        np.testing.assert_allclose(v.shape, img.shape)
        np.testing.assert_allclose(v, np.zeros_like(v))
        np.testing.assert_allclose(k.shape, img.shape)
        np.testing.assert_allclose(k, np.zeros_like(k))


if __name__ == '__main__':
    unittest.main()
