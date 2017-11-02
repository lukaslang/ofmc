#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dolfin import *
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy import misc
from scipy import ndimage

# Set regularisation parameter.
alpha0 = 5e-3
alpha1 = 5e-3
beta0 = 5e-4
beta1 = 5e-4
gamma = 1e-3

# Load image.
name = ['E2PSB1PMT-560-C1', 'E5PSB1PMT-1', 'E8-PSB1PMT', 'PSB1_E1PMT', 'PSB4PMT', 'PSB5PMT', 'epidermalejonction', 'artificialcut_small']
ending = ['png', 'tif', 'tif', 'tif', 'tif', 'tif', 'png', 'png']

# Select file.
idx = 0
name = name[idx]

# Read image.
img = misc.imread('../../data/{0}.{1}'.format(name, ending[idx]))

# Remove cut.
img = np.vstack((img[0:4, :], img[6:, :]))

# Filter image.
img = ndimage.gaussian_filter(img, sigma=1)

# Normalise to [0, 1].
img = np.array(img, dtype=float)
img = (img - img.min()) / (img.max() - img.min())

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

# Define derivatives of data.
fx = f.dx(1);
ft = f.dx(0);

# Define function space and functions.
W = VectorFunctionSpace(mesh, 'CG', 1, dim=2)

w = Function(W)
v, k = split(w)
w1, w2 = TestFunctions(W)

# Define weak formulation.
A = - (ft + fx*v + f*v.dx(1) - k)*(fx*w1 + f*w1.dx(1))*dx \
    - alpha0*v.dx(1)*w1.dx(1)*dx - alpha1*v.dx(0)*w1.dx(0)*dx \
    - gamma*k.dx(0)*k.dx(1)*w1*dx - gamma*k.dx(1)*k.dx(1)*v*w1*dx \
    + (ft + fx*v + f*v.dx(1) - k)*w2*dx - beta0*k.dx(1)*w2.dx(1)*dx - beta1*k.dx(0)*w2.dx(0)*dx \
    - gamma*k.dx(0)*w2.dx(0)*dx - gamma*k.dx(1)*v*w2.dx(0)*dx - gamma*k.dx(0)*v*w2.dx(1)*dx - gamma*k.dx(1)*v*v*w2.dx(1)*dx

# Compute solution.
solve(A == 0, w)

# Recover solution.
v, k = w.split(deepcopy=True)

# Create image from solution.
v2d = vertex_to_dof_map(V)
vel = np.zeros_like(img)
values = v.vector().array()[v2d]
for (i, j, v) in zip(x, y, values): vel[i, j] = v

source = np.zeros_like(img)
values = k.vector().array()[v2d]
for (i, j, v) in zip(x, y, values): source[i, j] = v

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
fig.savefig('results/cmscr_newton/{0}-vel.png'.format(name))

# Create grid for streamlines.
Y, X = np.mgrid[0:m:1, 0:n:1]
V = np.ones_like(X)*hy

# Plot streamlines.
fig, ax = plt.subplots(figsize=(10, 10))
plt.imshow(img, cmap=cm.gray)

strm = ax.streamplot(X, Y, vel*hx, V, density=2, color=vel, linewidth=1, cmap=cm.coolwarm)
fig.colorbar(strm.lines, orientation='horizontal')
fig.savefig('results/cmscr_newton/{0}-img.png'.format(name))

# Plot source.
fig, ax = plt.subplots()
cax = ax.imshow(source, interpolation='nearest', cmap=cm.coolwarm)
ax.set_title('Source')
cbar = fig.colorbar(cax, orientation='vertical')
plt.show()
fig.savefig('results/cmscr_newton/{0}-source.png'.format(name))