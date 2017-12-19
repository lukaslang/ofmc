#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
import numpy as np
import ofmc.util.dolfinhelpers as dh


class TestDolfinhelpers(unittest.TestCase):

    def test_img2fun_fun2img(self):
        m, n = 7, 13
        img = np.random.rand(m, n)
        v = dh.img2funvec(img)
        np.testing.assert_allclose(dh.funvec2img(v, m, n), img)


if __name__ == '__main__':
    unittest.main()
