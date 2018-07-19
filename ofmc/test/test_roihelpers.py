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
import imageio
from read_roi import read_roi_zip
from matplotlib import cm
from matplotlib.collections import LineCollection
from scipy.interpolate import UnivariateSpline
import ofmc.util.roihelpers as rh
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class TestRoiHelpers(unittest.TestCase):

    def test_removeduplicates(self):
        x = np.array([-1, 1, 3, 3, 5, 7])
        x, y = rh.removeduplicates(x, x)
        np.testing.assert_array_equal(x, np.array([-1, 1, 3, 5, 7]))
        np.testing.assert_array_equal(y, np.array([-1, 1, 3, 5, 7]))

        z = np.array([-1, 1, 3, 5, 7])
        x, y = rh.removeduplicates(z, z)
        np.testing.assert_array_equal(x, z)
        np.testing.assert_array_equal(y, z)

        x = np.array([-1, 1, 3, 3, 5, 7])
        y = np.array([7, 3, 2, 8, 2, 1])
        a, b = rh.removeduplicates(x, y)
        np.testing.assert_array_equal(a, np.array([-1, 1, 3, 5, 7]))
        np.testing.assert_array_equal(b, np.array([7, 3, 2, 2, 1]))

    def test_roi2splines(self):
        # Load test zip.
        roi = read_roi_zip('ofmc/test/data/Manual_ROIs.zip')

        # Create splines.
        spl = rh.roi2splines(roi)
        self.assertEqual(len(spl), len(roi))

    def test_plotroi(self):
        # Load test zip.
        roi = read_roi_zip('ofmc/test/data/Manual_ROIs.zip')

        # Load test image.
        name = 'ofmc/test/data/DynamicReslice of E2PSB1PMT_10px.tif'
        img = imageio.imread(name)

        # Plot image.
        plt.imshow(img, cmap=cm.gray)

        for v in roi:
            plt.plot(roi[v]['x'], roi[v]['y'], lw=2)
        plt.show()

    def test_plotsplines(self):
        # Load test zip.
        roi = read_roi_zip('ofmc/test/data/Manual_ROIs.zip')

        # Create splines.
        spl = rh.roi2splines(roi)

        # Load test image.
        name = 'ofmc/test/data/DynamicReslice of E2PSB1PMT_10px.tif'
        img = imageio.imread(name)

        # Plot image.
        plt.imshow(img, cmap=cm.gray)

        # Plot splines.
        for v in roi:
            xs = np.linspace(max(roi[v]['y'][0], 5),
                             min(roi[v]['y'][-1], 30), 30)
            plt.plot(spl[v](xs), xs, lw=2)
        plt.show()

    def test_plotsplinesderivcolour(self):
        # Load test zip.
        roi = read_roi_zip('ofmc/test/data/Manual_ROIs.zip')

        # Create splines.
        spl = rh.roi2splines(roi)

        # Load test image.
        name = 'ofmc/test/data/DynamicReslice of E2PSB1PMT_10px.tif'
        img = imageio.imread(name)

        # Plot image.
        plt.imshow(img, cmap=cm.gray)

        # Plot splines.
        for v in roi:
            y = roi[v]['y']
            # Compute derivative of spline.
            derivspl = spl[v].derivative()

            points = np.array([spl[v](y), y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(segments, cmap=cm.coolwarm,
                                norm=plt.Normalize(-2, 2))
            lc.set_array(derivspl(y))
            lc.set_linewidth(2)
            plt.gca().add_collection(lc)
        plt.show()

    def test_spline_derivative(self):

        # Define image.
        m, n = 30, 100
        img = np.zeros((m, n))

        num = 5

        # Create trajectory.
        x = np.linspace(0, m, num)
        y = np.linspace(0, 10, num)

        # Fit spline.
        f = UnivariateSpline(x, y, k=3)

        plt.imshow(img, cmap=cm.gray)
        points = np.array([f(x), x]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap=cm.coolwarm,
                            norm=plt.Normalize(-2, 2))
        lc.set_linewidth(2)
        plt.gca().add_collection(lc)
        plt.show()

        fd = f.derivative()
        np.testing.assert_allclose(fd(x) * m / n, np.repeat(0.1, num))

    def test_spline_derivative_points(self):
        m, n = 30, 100

        # Create spline woth points.
        x = [0, 30, 60, 90]
        y = [0, 10, 20, 30]
        spl = UnivariateSpline(x, y, k=3)
        splderiv = spl.derivative()
        np.testing.assert_allclose(splderiv(0) * m / n, 0.1)


if __name__ == '__main__':
    unittest.main()
