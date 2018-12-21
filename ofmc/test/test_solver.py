#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
import numpy as np
import ofmc.mechanics.solver as solver
import scipy.sparse as sparse


class TestSolver(unittest.TestCase):

    def test_solve(self):
        # Create model and solver parameters.
        mp = solver.ModelParams()
        mp.k_on = 0
        mp.k_off = 0
        mp.t_cut = 1

        sp = solver.SolverParams()
        sp.n = 100
        sp.m = 10

        # Define initial values.
        def const(x):
            return 1.0

        # Initialise tracers.
        x = np.array(np.linspace(0, 1, num=5))

        # Run solver.
        rho, ca, v, sigma, x, idx = solver.solve(mp, sp, const, const, x)

        # Check returned arrays.
        const_exp = np.vectorize(const)(np.zeros((sp.m + 1, sp.n)))

        np.testing.assert_equal(rho.shape, (sp.m + 1, sp.n))
        np.testing.assert_equal(ca.shape, (sp.m + 1, sp.n))
        np.testing.assert_allclose(rho, const_exp, atol=1e-7)
        np.testing.assert_allclose(ca, const_exp, atol=1e-7)
        np.testing.assert_allclose(v, np.zeros((sp.m + 1, sp.n + 1)),
                                   atol=1e-7)

    def test_stress_matrix(self):
        # Check for small matrix.
        n = 3
        dx = 0.01
        A = solver.stress_matrix(n, dx, 1, 1)
        Aexp = sparse.csc_matrix(np.matrix([[1 + dx**2, -1, 0],
                                            [-1, 2 + dx**2, -1],
                                            [0, -1, 1 + dx**2]]))
        # Check returned arrays.
        np.testing.assert_allclose(A.toarray(), Aexp.toarray(), atol=1e-3)


if __name__ == '__main__':
    unittest.main()
