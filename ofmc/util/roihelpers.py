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
        spl[v] = UnivariateSpline(roi[v]['y'], roi[v]['x'], k=3)
    return spl
