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
from ofmc.model.cm import cm1d
from ofmc.model.cm import cm1dsource
from ofmc.model.cm import cm1dvelocity


class TestOf(unittest.TestCase):

    def test_cm1d(self):
        # Create zero image.
        img = np.zeros((10, 25))
        v = cm1d(img, 1, 1)

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
        v = cm1d(img, 1, 1)

        np.testing.assert_allclose(v.shape, img.shape)
        np.testing.assert_allclose(v, np.zeros_like(v))


if __name__ == '__main__':
    unittest.main()
