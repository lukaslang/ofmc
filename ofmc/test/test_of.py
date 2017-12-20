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
from ofmc.model.of import of1d


class TestOf(unittest.TestCase):

    def test_of1d(self):
        # Create zero image.
        img = np.zeros((10, 25))
        v = of1d(img, 1, 1)

        np.testing.assert_allclose(v.shape, img.shape)
        np.testing.assert_allclose(v, np.zeros_like(v))

    def test_of1d_random(self):
        # Create random non-moving image.
        img = np.matlib.repmat(np.random.rand(1, 25), 10, 1)
        v = of1d(img, 1, 1)

        np.testing.assert_allclose(v.shape, img.shape)
        np.testing.assert_allclose(v, np.zeros_like(v))


if __name__ == '__main__':
    unittest.main()
