#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
import matplotlib.pyplot as plt
import numpy as np
import ofmc.util.dolfinhelpers as dh
import ofmc.util.velocity as vel
from dolfin import plot
from matplotlib import cm
from matplotlib.collections import LineCollection

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

    def test_characteristic_size(self):
        m, n = 100, 100

        # Compute characteristic.
        c = vel.characteristic(m, n, c0, v0, tau0, tau1, 0.5)
        np.testing.assert_equal(c.shape, (m + 1, 1))

    def test_characteristic(self):
        m, n = 100, 100

        # Compute characteristic.
        c = vel.characteristic(m, n, c0, v0, tau0, tau1, 0.6)

        # Compute velocity field.
        v = vel.velocity(m, n, c0, v0, tau0, tau1)

        # Convert to matrix.
        v = dh.funvec2img(v.vector().array(), m, n)

        # Plot velocity.
        fig = plt.figure()
        plt.imshow(v, cmap=cm.coolwarm)

        # Plot characteristic.
        y = np.linspace(0, m, m + 1).reshape(m + 1, 1)

        points = np.array([n * c, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments[:-1])
        lc.set_linewidth(2)
        plt.gca().add_collection(lc)
        plt.show()

        plt.close(fig)

    def test_characteristics_size(self):
        m, n = 100, 100

        # Compute characteristic.
        y0 = np.linspace(0, 1, 30)
        c = vel.characteristics(m, n, c0, v0, tau0, tau1, y0)
        np.testing.assert_equal(c.shape, (m + 1, 30))

    def test_characteristics(self):
        m, n = 50, 100

        # Compute characteristics.
        y0 = np.linspace(0, 1, 30)
        c = vel.characteristics(m, n, c0, v0, tau0, tau1, y0)

        # Compute velocity field.
        v = vel.velocity(m, n, c0, v0, tau0, tau1)

        # Convert to matrix.
        v = dh.funvec2img(v.vector().array(), m, n)

        # Plot velocity.
        fig = plt.figure()
        plt.imshow(v, cmap=cm.coolwarm)

        # Plot characteristics.
        y = np.linspace(0, m, m + 1).reshape(m + 1, 1)

        for k in range(y0.size):
            ck = n * c[:, k].reshape(m + 1, 1)
            points = np.array([ck, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(segments[:-1])
            lc.set_linewidth(2)
            plt.gca().add_collection(lc)

        plt.show()
        plt.close(fig)


if __name__ == '__main__':
    unittest.main()
