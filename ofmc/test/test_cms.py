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
from ofmc.model.cms import cms1d
from ofmc.model.cms import cms1dpbc
from ofmc.model.cms import cms1dl2


class TestOf(unittest.TestCase):

    def test_cms1d(self):
        # Create zero image.
        img = np.zeros((10, 25))
        v, k = cms1d(img, 1, 1, 1, 1)

        np.testing.assert_allclose(v.shape, img.shape)
        np.testing.assert_allclose(v, np.zeros_like(v))
        np.testing.assert_allclose(k.shape, img.shape)
        np.testing.assert_allclose(k, np.zeros_like(k))

    def test_cms1d_random(self):
        # Create random non-moving image.
        img = matlib.repmat(np.random.rand(1, 25), 10, 1)
        v, k = cms1d(img, 1, 1, 1, 1)

        np.testing.assert_allclose(v.shape, img.shape)
        np.testing.assert_allclose(v, np.zeros_like(v))
        np.testing.assert_allclose(k.shape, img.shape)
        np.testing.assert_allclose(k, np.zeros_like(k))

    def test_cms1dl2(self):
        # Create zero image.
        img = np.zeros((10, 25))
        v, k = cms1dl2(img, 1, 1, 1)

        np.testing.assert_allclose(v.shape, img.shape)
        np.testing.assert_allclose(v, np.zeros_like(v))
        np.testing.assert_allclose(k.shape, img.shape)
        np.testing.assert_allclose(k, np.zeros_like(k))

    def test_cms1dpbc(self):
        # Create zero image.
        img = np.zeros((10, 25))
        v, k = cms1dpbc(img, 1, 1, 1, 1)

        np.testing.assert_allclose(v.shape, img.shape)
        np.testing.assert_allclose(v, np.zeros_like(v))
        np.testing.assert_allclose(k.shape, img.shape)
        np.testing.assert_allclose(k, np.zeros_like(k))


if __name__ == '__main__':
    unittest.main()
