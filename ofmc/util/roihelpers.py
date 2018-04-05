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
from scipy.interpolate import UnivariateSpline


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
