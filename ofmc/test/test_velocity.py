#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
import numpy as np
import ofmc.util.velocity as vel
from dolfin import plot
from matplotlib import pyplot

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
        pyplot.figure()
        for t in rng:
            v = vel.velocityslice(t, n, c0, v0, tau0, tau1)
            plot(v, range_min=-v0, range_max=v0)

        pyplot.show()
        pyplot.close()

    def test_velocity(self):
        m, n = 100, 100
        v = vel.velocity(m, n, c0, v0, tau0, tau1)

        # Plot velocity.
        pyplot.figure()
        p = plot(v)
        p.set_cmap("coolwarm")
        pyplot.colorbar(p)
        pyplot.show()
        pyplot.close()


if __name__ == '__main__':
    unittest.main()
