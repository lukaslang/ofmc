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
from dolfin import Expression
from dolfin import Function
from dolfin import FunctionSpace
from dolfin import interpolate
from dolfin import UnitIntervalMesh
from dolfin import UnitSquareMesh
from scipy.integrate import ode
import numpy as np


def gammavel(t, v0, tau0):
    return v0 * np.exp(-t / tau0)


def gamma(t, c0, v0, tau0):
    return c0 + tau0 * v0 * (1 - np.exp(-t / tau0))


def vel(t, x, c0, v0, tau0, tau1):
    gvel = gammavel(t, v0, tau0)
    g = gamma(t, c0, v0, tau0)

    if abs(x) <= gamma(t, c0, v0, tau0):
        v = gvel * abs(x) / g
    else:
        v = gvel * np.exp(-abs(abs(x) - g) / tau1)

    return -v if x < 0 else v


def velocityslice(t: float, n: int, c0: float, v0: float, tau0: float,
                  tau1: float) -> Function:
    """Creates a 1D slice of velocity at a specified time.

    Takes time t, number of mesh cells, and parameters, and returns a 1D
    function.

    Args:
        t (int): The time.
        n (int): The number of cells in space.
        c0 (float): The characteristic start at 0.5 - c0 and 0.5 + c0.
        v0 (float): Initial velocity of the characteristic.
        tau0 (float): Decay parameter of the velocity of the characteristic.
        tau1 (float): Decay of velocity outside the characteristic.

    Returns:
        Function: A 1D function.

    """
    # Create mesh.
    mesh = UnitIntervalMesh(n-1)

    # Define function space.
    V = FunctionSpace(mesh, 'CG', 1)

    # Create velocity expression.
    class velocity(Expression):

        def eval(self, value, x):
            value[0] = vel(t, x[0] - 0.5, c0, v0, tau0, tau1)

        def value_shape(self):
            return ()

    v = velocity(degree=1)
    v = interpolate(v, V)
    return v


def velocity(m: int, n: int, c0: float, v0: float, tau0: float,
             tau1: float) -> Function:
    """Creates a velocity field (1D + time).

    Takes number of mesh cells t in time, number of mesh cells n in space, and
    parameters, and returns a 2D function.

    Args:
        m (int): The number of cells in time.
        n (int): The number of cells in space.
        c0 (float): The characteristic start at 0.5 - c0 and 0.5 + c0.
        v0 (float): Initial velocity of the characteristic.
        tau0 (float): Decay parameter of the velocity of the characteristic.
        tau1 (float): Decay of velocity outside the characteristic.

    Returns:
        Function: A 2D function.

    """
    # Create mesh.
    mesh = UnitSquareMesh(m-1, n-1)

    # Define function space.
    V = FunctionSpace(mesh, 'CG', 1)

    # Create velocity expression.
    class velocity(Expression):

        def eval(self, value, x):
            value[0] = vel(x[0], x[1] - 0.5, c0, v0, tau0, tau1)

        def value_shape(self):
            return ()

    v = velocity(degree=1)
    v = interpolate(v, V)
    return v


def characteristic(m: int, n: int, c0: float, v0: float, tau0: float,
                   tau1: float, y0: float) -> np.array:
    """Computes a characteristic for a parametrised velocity field.

    Takes number of mesh cells t in time, number of mesh cells n in space, and
    velocity parameters, an initial position y0, and returns a vector of
    positions.

    Args:
        m (int): The number of cells in time.
        n (int): The number of cells in space.
        c0 (float): The characteristic start at 0.5 - c0 and 0.5 + c0.
        v0 (float): Initial velocity of the characteristic.
        tau0 (float): Decay parameter of the velocity of the characteristic.
        tau1 (float): Decay of velocity outside the characteristic.
        y0 (float): Initial position of the characteristic.

    Returns:
        np.array: An array of shape (m + 1, 1).

    """
    def f(t, y):
        return vel(t, y - 0.5, c0, v0, tau0, tau1)

    # Set initial parameter.
    t0 = 0

    # Create integrator.
    r = ode(f).set_integrator('dopri5')
    r.set_initial_value(y0, t0)

    # Set parameters.
    T = 1
    dt = 1 / (m - 1)

    # Create solution array.
    y = np.zeros((m + 1, 1))
    y[0] = y0

    k = 1
    while r.successful() and r.t < T:
        r.integrate(r.t + dt)
        y[k] = r.y
        k += 1

    return y


def characteristics(m: int, n: int, c0: float, v0: float, tau0: float,
                    tau1: float, y0: np.array) -> np.array:
    """Computes a characteristic for a parametrised velocity field.

    Takes number of mesh cells t in time, number of mesh cells n in space, and
    velocity parameters, initial positions y0, and returns a vector of
    positions.

    Args:
        m (int): The number of cells in time.
        n (int): The number of cells in space.
        c0 (float): The characteristic start at 0.5 - c0 and 0.5 + c0.
        v0 (float): Initial velocity of the characteristic.
        tau0 (float): Decay parameter of the velocity of the characteristic.
        tau1 (float): Decay of velocity outside the characteristic.
        y0 (np.array): An array of initial positions of shape (k, 1).

    Returns:
        np.array: An array of shape (m + 1, k).

    """
    p = y0.shape[0]
    c = np.zeros((m + 1, p))

    for k in range(p):
        c[:, k] = characteristic(m, n, c0, v0, tau0, tau1, y0[k]).flatten()

    return c
