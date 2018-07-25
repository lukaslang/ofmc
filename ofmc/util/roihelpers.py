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
from scipy import interpolate
from scipy.interpolate import UnivariateSpline
from scipy.integrate import solve_ivp


def roi2splines(roi: dict) -> dict:
    """Takes a dict of ROIs and returns a dict of fitted cubic splines.

    Iterates through all elements in roi and returns a dict of cubic splines.
    The returned dict has the same keys as roi. The function assumes that roi
    consists of 'polyline's.

    Note that the axis direction for each spline is switched!

    See also https://github.com/hadim/read-roi

    Args:
        roi (dict): A dict of dicts.

    Returns:
        dict: A dict with elements of type scipy.interpolate.UnivariateSpline

    """
    spl = dict()
    for v in roi:
        x, y = removeduplicates(np.asarray(roi[v]['y']),
                                np.asarray(roi[v]['x']))
        spl[v] = UnivariateSpline(x, y, k=3)
    return spl


def removeduplicates(x: np.array, y: np.array) -> (np.array, np.array):
    """Takes to arrays of ints and removes entries that appear as duplicates in
    the first array.

    Args:
        x (np.array): The first array.
        y (np.array): The second array.

    Returns:
        np.array: Cleaned first array.
        np.array: Cleaned second array.
    """
    seen = set()
    ind = []
    for k in range(x.size):
        if x[k] not in seen:
            seen.add(x[k])
            ind.append(k)

    xr = x[ind]
    yr = y[ind]
    return xr, yr


def compute_error(vel: np.array, roi, spl) -> dict:
    """Takes a velocity array, a roi instance, fitted splines, and returns a
    dictionary of error arrays.

    Args:
        vel (np.array): The velocity.
        roi: A roi instance.
        spl: Fitted spolines.

    Returns:
        error (dict): A dictionary of errors.
    """
    m, n = vel.shape
    gridx, gridy = np.mgrid[0:m, 0:n]
    gridpoints = np.hstack([gridx.reshape(m * n, 1), gridy.reshape(m * n, 1)])

    error = dict()
    for v in roi:
        y = roi[v]['y']

        # Interpolate velocity.
        y = np.arange(y[0], y[-1] + 1, 1)
        x = np.array(spl[v](y))
        veval = interpolate.griddata(gridpoints, vel.flatten(), (y, x),
                                     method='linear')

        # Compute derivative of spline.
        derivspl = spl[v].derivative()

        # Compute error in velocity.
        error[v] = abs(derivspl(y) * m / n - veval)
    return error


def compute_endpoint_error(vel: np.array, roi, spl) -> (dict, dict):
    """Takes a velocity array, a roi instance, fitted splines, and returns a
    dictionary of error arrays.

    Args:
        vel (np.array): The velocity.
        roi: A roi instance.
        spl: Fitted spolines.

    Returns:
        error (dict): A dictionary of errors.
        curve (dict): A dictionary of points of the trajectory.
    """
    m, n = vel.shape
    gridx, gridy = np.mgrid[0:m, 0:n]
    gridpoints = np.hstack([gridx.reshape(m * n, 1), gridy.reshape(m * n, 1)])

    # Define ODE.
    def ode(t, y): return interpolate.griddata(gridpoints,
                                               vel.flatten(), (t, y),
                                               method='cubic')
    error = dict()
    curve = dict()
    for v in roi:
        y = roi[v]['y']

        # Interpolate velocity.
        y = np.arange(y[0], y[-1] + 1, 1)
        x = np.array(spl[v](y))

        # Solve initial value problem.
        sol = solve_ivp(ode, [y[0], y[-1]], [x[0]], t_eval=y, method='RK45')

        # Compute error in velocity.
        error[v] = abs(x - sol.y[0, :])
        curve[v] = sol.y[0, :]
    return error, curve
