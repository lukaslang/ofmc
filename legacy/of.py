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
from dolfin import UnitSquareMesh
from dolfin import FunctionSpace
from dolfin import Function
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy import misc
from scipy import ndimage

# Set regularisation parameter.
alpha = 1e-2
beta = 1e-3

# Load image.
name = ['E2PSB1PMT-560-C1', 'E5PSB1PMT-1', 'E8-PSB1PMT', 'PSB1_E1PMT', 'PSB4PMT', 'PSB5PMT', 'epidermalejonction', 'artificialcut_small']
ending = ['png', 'tif', 'tif', 'tif', 'tif', 'tif', 'png', 'png']

# Select file.
idx = 0
name = name[idx]

# Read image.
img = misc.imread('../data/{0}.{1}'.format(name, ending[idx]))

# Remove cut.
img = np.vstack((img[0:4, :], img[6:, :]))

# Filter image.
img = ndimage.gaussian_filter(img, sigma=1)

# Normalise to [0, 1].
img = np.array(img, dtype=float)
img = (img - img.min()) / (img.max() - img.min())

#plt.imshow(img, cmap=plt.cm.gray)
#plt.show()

# Create mesh.
[m, n] = img.shape
mesh = UnitSquareMesh(m, n)
x = mesh.coordinates().reshape((-1, 2))

# Evaluate function at vertices.
hx, hy = 1./(m-1), 1./(n-1)
x, y = np.array(x[:, 0]/hx, dtype=int), np.array(x[:, 1]/hy, dtype=int)
fv = img[x, y]

# Create function space.
V = FunctionSpace(mesh, 'CG', 1)
f = Function(V)

d2v = dof_to_vertex_map(V)
f.vector()[:] = fv[d2v]

# Define function space for derivatives of data.
#Q = FunctionSpace(mesh, "DG", 0)
fx = f.dx(1);
ft = f.dx(0);
fxft = ft*fx;
fxfx = fx*fx;

# Define functions.
v = TrialFunction(V)
w = TestFunction(V)

# Define weak formulation.
A = inner(fxfx*v, w)*dx + alpha*inner(v.dx(1), w.dx(1))*dx + beta*inner(v.dx(0), w.dx(0))*dx
l = -inner(fxft, w)*dx

# Compute solution.
v = Function(V)
solve(A == l, v)

# Create image from solution.
v2d = vertex_to_dof_map(V)
vel = np.zeros_like(img)
values = v.vector().array()[v2d]
for (i, j, v) in zip(x, y, values): vel[i, j] = v

# Plot image.
plt.figure(figsize=(10, 10))
plt.imshow(img, cmap=cm.gray)

# Plot velocity.
fig, ax = plt.subplots()
cax = ax.imshow(vel, interpolation='nearest', cmap=cm.coolwarm)
ax.set_title('Velocity')

maxvel = abs(vel).max()
# Add colorbar, make sure to specify tick locations to match desired ticklabels
cbar = fig.colorbar(cax, orientation='horizontal')
fig.savefig('results/of/{0}-vel.png'.format(name))

# Create grid for streamlines.
Y, X = np.mgrid[0:m:1, 0:n:1]
V = np.ones_like(X)*hy

# Plot streamlines.
fig, ax = plt.subplots(figsize=(10, 10))
plt.imshow(img, cmap=cm.gray)
strm = ax.streamplot(X, Y, vel*hx, V, density=3, color=vel, linewidth=1, cmap=cm.coolwarm)
fig.colorbar(strm.lines, orientation='horizontal')
plt.show()
fig.savefig('results/of/{0}-img.png'.format(name))