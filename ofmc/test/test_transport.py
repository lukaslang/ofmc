#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import unittest
from dolfin import UserExpression
from dolfin import FunctionSpace
from dolfin import interpolate
from dolfin import UnitIntervalMesh
from matplotlib import cm
from ofmc.util.transport import transport1d
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# Define parameters.
c0 = 0.1
v0 = 0.5
tau0 = 1
tau1 = 0.2


class Hat(UserExpression):

    def eval(self, value, x):
        value[0] = max(0, 0.1 - abs(x[0] - 0.5))

    def value_shape(self):
        return ()


class DoubleHat(UserExpression):

    def eval(self, value, x):
        value[0] = max(0, 0.1 - abs(x[0] - 0.4)) \
            + max(0, 0.1 - abs(x[0] - 0.6))

    def value_shape(self):
        return ()


class Rectangle(UserExpression):

    def eval(self, value, x):
        value[0] = 1 if x[0] >= 0.4 and x[0] <= 0.6 else 0

    def value_shape(self):
        return ()


class Membrane(UserExpression):

    def eval(self, value, x):
        value[0] = 1 if x[0] <= 0.4 or x[0] >= 0.6 else 0

    def value_shape(self):
        return ()


class TestTransport(unittest.TestCase):

    def test_transport_hat_zero(self):
        m, n = 10, 100
        v = np.zeros((m, n))

        # Define mesh and function space.
        mesh = UnitIntervalMesh(n - 1)
        V = FunctionSpace(mesh, 'CG', 1)

        # Define initial condition.
        f0 = Hat(degree=1)
        f0 = interpolate(f0, V)

        # Compute transport
        f = transport1d(v, np.zeros_like(v), f0.vector().get_local())

        finit = f0.vector().get_local().reshape(1, n)
        np.testing.assert_equal(f.shape, (m + 1, n))
        np.testing.assert_allclose(f, np.tile(finit, (m + 1, 1)), atol=1e-3)

    def test_transport_hat(self):
        m, n = 10, 100
        v = 0.1 * np.ones((m, n))

        # Define mesh and function space.
        mesh = UnitIntervalMesh(n - 1)
        V = FunctionSpace(mesh, 'CG', 1)

        # Define initial condition.
        f0 = Hat(degree=1)
        f0 = interpolate(f0, V)

        # Compute transport
        f = transport1d(v, np.zeros_like(v), f0.vector().get_local())
        np.testing.assert_equal(f.shape, (m + 1, n))

        fig = plt.figure()
        plt.imshow(f, cmap=cm.coolwarm)
        plt.show()
        plt.close(fig)

    def test_transport_hat_source(self):
        m, n = 10, 100
        v = 0.1 * np.ones((m, n))

        # Define mesh and function space.
        mesh = UnitIntervalMesh(n - 1)
        V = FunctionSpace(mesh, 'CG', 1)

        # Define initial condition.
        f0 = Hat(degree=1)
        f0 = interpolate(f0, V)

        # Compute transport
        f = transport1d(v, np.zeros_like(v), f0.vector().get_local())
        np.testing.assert_equal(f.shape, (m + 1, n))

        # Compute transport with f as source.
        f = transport1d(v, f[:-1], f0.vector().get_local())

        fig, ax = plt.subplots(figsize=(10, 5))
        cax = ax.imshow(f, cmap=cm.coolwarm)
        fig.colorbar(cax, orientation='vertical')
        plt.show()
        plt.close(fig)

    def test_transport_rectangle(self):
        m, n = 10, 100
        v = 0.1 * np.ones((m, n))

        # Define mesh.
        mesh = UnitIntervalMesh(n - 1)

        # Define function spaces
        V = FunctionSpace(mesh, 'CG', 1)

        f0 = Rectangle(degree=1)
        f0 = interpolate(f0, V)

        # Compute transport
        f = transport1d(v, np.zeros_like(v), f0.vector().get_local())

        fig, ax = plt.subplots(figsize=(10, 5))
        cax = ax.imshow(f, cmap=cm.coolwarm)
        fig.colorbar(cax, orientation='vertical')
        plt.show()
        plt.close(fig)

        np.testing.assert_equal(f.shape, (m + 1, n))

    def test_transport_rectangle_source(self):
        m, n = 10, 100
        v = 0.1 * np.ones((m, n))

        # Define mesh.
        mesh = UnitIntervalMesh(n - 1)

        # Define function spaces
        V = FunctionSpace(mesh, 'CG', 1)

        f0 = Rectangle(degree=1)
        f0 = interpolate(f0, V)

        # Compute transport
        f = transport1d(v, np.zeros_like(v), f0.vector().get_local())

        fig, ax = plt.subplots(figsize=(10, 5))
        cax = ax.imshow(f, cmap=cm.coolwarm)
        fig.colorbar(cax, orientation='vertical')
        plt.show()
        plt.close(fig)

        np.testing.assert_equal(f.shape, (m + 1, n))

    def test_transport_rectangle_zero(self):
        m, n = 5, 50
        v = np.zeros((m, n))

        # Define mesh.
        mesh = UnitIntervalMesh(n - 1)

        # Define function spaces
        V = FunctionSpace(mesh, 'CG', 1)

        f0 = Rectangle(degree=1)
        f0 = interpolate(f0, V)
        f0 = f0.vector().get_local()

        # Compute transport
        f = transport1d(v, np.zeros_like(v), f0)

        fig = plt.figure()
        plt.imshow(f, cmap=cm.coolwarm)
        plt.show()
        plt.close(fig)

        np.testing.assert_equal(f.shape, (m + 1, n))
        np.testing.assert_allclose(f, np.tile(f0, (m + 1, 1)),
                                   atol=1e-6, rtol=1e-6)

    def test_transport_rectangle_zero_source(self):
        m, n = 5, 100
        v = np.zeros((m, n))

        # Define mesh.
        mesh = UnitIntervalMesh(n - 1)

        # Define function spaces
        V = FunctionSpace(mesh, 'CG', 1)

        f0 = Rectangle(degree=1)
        f0 = interpolate(f0, V)
        f0 = f0.vector().get_local()

        # Create source.
        src = np.tile(f0, (m, 1))

        # Compute transport
        f = transport1d(v, src, f0)

        fig, ax = plt.subplots(figsize=(10, 5))
        cax = ax.imshow(f, cmap=cm.coolwarm)
        fig.colorbar(cax, orientation='vertical')
        plt.show()
        plt.close(fig)

        np.testing.assert_equal(f.shape, (m + 1, n))


if __name__ == '__main__':
    unittest.main()
