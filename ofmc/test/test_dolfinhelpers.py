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

    def test_imgseq2fun_fun2imgseq(self):
        m, n, o = 7, 13, 10
        img = np.random.rand(m, n, o)
        v = dh.imgseq2funvec(img)
        np.testing.assert_allclose(dh.funvec2imgseq(v, m, n, o), img)


if __name__ == '__main__':
    unittest.main()
