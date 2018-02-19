#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
import matplotlib.pyplot as plt
import numpy as np
import ofmc.util.dolfinhelpers as dh
import ofmc.util.velocity as vel
from dolfin import plot
from matplotlib import cm

# Define parameters.
c0 = 0.1
v0 = 0.5
tau0 = 1
tau1 = 0.2


class TestVelocity(unittest.TestCase):

    def test_velocityslice(self):
        n = 100

        # Set evaluations.
        rng = np.linspace(0, 1, num=10)

        # Plot evaluations for different times.
        plt.figure()
        for t in rng:
            v = vel.velocityslice(t, n, c0, v0, tau0, tau1)
            plot(v, range_min=-v0, range_max=v0)

        plt.show()
        plt.close()

    def test_velocity(self):
        m, n = 100, 100
        v = vel.velocity(m, n, c0, v0, tau0, tau1)

        # Plot velocity.
        plt.figure()
        p = plot(v)
        p.set_cmap("coolwarm")
        plt.colorbar(p)
        plt.show()
        plt.close()

        # Convert to matrix.
        v = dh.funvec2img(v.vector().array(), m, n)

        plt.figure()
        plt.imshow(v, cmap=cm.coolwarm)
        plt.show()
        plt.close()


if __name__ == '__main__':
    unittest.main()
