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
from scipy import misc
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import LineCollection
from read_roi import read_roi_zip
import ofmc.util.roihelpers as rh


class TestRoihelpers(unittest.TestCase):

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
        img = misc.imread(name)

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
        img = misc.imread(name)

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
        img = misc.imread(name)

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


if __name__ == '__main__':
    unittest.main()
