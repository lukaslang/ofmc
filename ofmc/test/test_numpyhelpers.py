#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
import numpy as np
import ofmc.util.numpyhelpers as nh


class TestNumpyHelpers(unittest.TestCase):

    def test_partial_derivatives(self):
        m, n = 7, 13
        f = np.ones((m, n))
        ft, fx = nh.partial_derivatives(f)
        np.testing.assert_allclose(ft, np.zeros_like(f))
        np.testing.assert_allclose(fx, np.zeros_like(f))


if __name__ == '__main__':
    unittest.main()
