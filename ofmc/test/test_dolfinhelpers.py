#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
import numpy as np
import ofmc.util.dolfinhelpers as dh
from dolfin import Constant
from dolfin import FunctionSpace
from dolfin import project
from dolfin import UnitSquareMesh


class TestDolfinHelpers(unittest.TestCase):

    def test_img2fun_fun2img(self):
        m, n = 7, 13
        img = np.random.rand(m, n)
        v = dh.img2funvec(img)
        np.testing.assert_allclose(dh.funvec2img(v, m, n), img)

    def test_funvec2img(self):
        m, n = 30, 100

        # Define mesh.
        mesh = UnitSquareMesh(m - 1, n - 1)

        # Define function spaces
        V = FunctionSpace(mesh, 'CG', 1)
        f = project(Constant(1.0), V)

        v = dh.funvec2img(f.vector().get_local(), m, n)
        np.testing.assert_allclose(v, np.ones((m, n)))

    def test_img2funvec(self):
        m, n = 3, 4
        img = np.ones((m, n))

        v = dh.img2funvec(img)
        np.testing.assert_allclose(v, np.ones(m*n))

    def test_imgseq2funvec_funvec2imgseq(self):
        m, n, o = 7, 13, 10
        img = np.random.rand(m, n, o)
        v = dh.imgseq2funvec(img)
        np.testing.assert_allclose(dh.funvec2imgseq(v, m, n, o), img)


if __name__ == '__main__':
    unittest.main()
