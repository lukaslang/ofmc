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


def partial_derivatives(f: np.array) -> (np.array, np.array):
    """Takes a 2D array and returns numerical approximations of partial
    derivatives with respect to the dimensions.

    Uses first order forward finite differences wrt. first dimension with zero
    Neumann boundary conditions.

    Uses second order centred finite differences wrt. second dimension and
    first order inward finite differences at the boundaries.

    Args:
        f (np.array): The input array of shape (m, n)

    Returns:
        (np.array, np.array): Partial derivatives wrt. first and second
                              dimension. Both arrays are of shape (m, n).

    """
    # Get shape of array.
    m, n = f.shape

    # Compute partial derivatives wrt. first dimension.
    ft = np.diff(f, axis=0) * (m - 1)
    ft = np.concatenate((ft, ft[-1, :].reshape(1, n)), axis=0)

    # Compute partial derivatives wrt. second dimension.
    fx = np.gradient(f, 1 / (n - 1), axis=1)

    return (ft, fx)
