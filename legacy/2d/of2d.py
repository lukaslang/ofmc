#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dolfin import *
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy import misc
from scipy import ndimage

import ofmc.external.tifffile as tiff

# Set regularisation parameter.
alpha = 1e-2
beta = 1e-3

# Load image.
name = ['epidermalejonction2d', 'CadherinFrames', 'MyosinFrames']
ending = ['tif', 'tif', 'tif']

# Select file.
idx = 1
name = name[idx]

# Read image.
#img = tiff.imread('../../../data/{0}.{1}'.format(name, ending[idx]))
img = tiff.imread('data/3LU/{0}.{1}'.format(name, ending[idx]))

# Select first frames.
img = img[0:3, 100:200, 100:200]

# Filter image.
for k in range(img.shape[0]):
    img[k] = ndimage.gaussian_filter(img[k], sigma=1)

# Normalise to [0, 1].
img = np.array(img, dtype=float)
img = (img - img.min()) / (img.max() - img.min())

# Remove last slice.
imgr = img[0:-1]

# Create mesh.
[m, n, o] = imgr.shape
mesh = UnitCubeMesh(m, n, o)
xc = mesh.coordinates().reshape((-1, 3))

# Evaluate function at vertices.
hx, hy, hz = 1./(m-1), 1./(n-1), 1./(o-1)
x, y, z = np.array(xc[:, 0]/hx, dtype=int), np.array(xc[:, 1]/hy, dtype=int), np.array(xc[:, 2]/hz, dtype=int)
fv = imgr[x, y, z]

# Create function space.
V = FunctionSpace(mesh, 'CG', 1)
W = VectorFunctionSpace(mesh, 'CG', 1, dim=2)
f = Function(V)

d2v = dof_to_vertex_map(V)
f.vector()[:] = fv[d2v]

# Compute derivative wrt. time.
imgt = img[1:] - img[0:-1]
ftv = imgt[x, y, z]

ft = Function(V)
ft.vector()[:] = ftv[d2v]

#v2d = vertex_to_dof_map(V)
#ftv = np.zeros_like(img)
#values = ft.vector().array()[v2d]
#for (i, j, k, v) in zip(x, y, z, values): ftv[i, j, k] = v
#plt.imshow(ftv[1])
#plt.show()

fx = f.dx(1)
fy = f.dx(2)

# Define derivatives of data.
#fxft = Function(V);
#fyft = Function(V);
#fxft.vector()[:] = project(fx, V).vector()[:] * ftv[d2v]
#fyft.vector()[:] = project(fy, V).vector()[:] * ftv[d2v]


# Define functions.
v1, v2 = TrialFunctions(W)
w1, w2 = TestFunction(W)

# Define weak formulation.
alpha = 0.025
beta = 1.0
A = fx*(fx*v1 + fy*v2)*w1*dx + fy*(fx*v1 + fy*v2)*w2*dx + beta*v1.dx(0)*w1.dx(0)*dx + alpha*v1.dx(1)*w1.dx(1)*dx + alpha*v1.dx(2)*w1.dx(2)*dx + beta*v2.dx(0)*w2.dx(0)*dx + alpha*v2.dx(1)*w2.dx(1)*dx + alpha*v2.dx(2)*w2.dx(2)*dx
l = -fx*ft*w1*dx - fy*ft*w2*dx

# Compute solution.
v = Function(W)
solve(A == l, v)

v1, v2 = v.split(deepcopy=True)

# Create image from solution.
v2d = vertex_to_dof_map(V)
vel1 = np.zeros_like(imgr)
vel2 = np.zeros_like(imgr)
values1 = v1.vector().array()[v2d]
values2 = v2.vector().array()[v2d]
for (i, j, k, v) in zip(x, y, z, values1): vel1[i, j, k] = v
for (i, j, k, v) in zip(x, y, z, values2): vel2[i, j, k] = v

# Plot image.
#plt.figure(figsize=(10, 10))
#plt.imshow(img[0], cmap=cm.gray)

#plt.figure(figsize=(10, 10))
#plt.imshow(vel1[0], cmap=cm.coolwarm)
#plt.figure(figsize=(10, 10))
#plt.imshow(vel2[0], cmap=cm.coolwarm)

# Plot velocity.
#fig, ax = plt.subplots()
#cax = ax.imshow(img[0], interpolation='nearest', cmap=cm.coolwarm)
#ax.set_title('Velocity')

#maxvel = abs(vel).max()
# Add colorbar, make sure to specify tick locations to match desired ticklabels
#cbar = fig.colorbar(cax, orientation='horizontal')
#fig.savefig('results/cm/{0}-vel.png'.format(name))

# Create grid for streamlines.
Y, X = np.mgrid[0:n:1, 0:o:1]
V, W = vel1, vel2
#V, W = np.ones_like(X), np.ones_like(X)
col = np.sqrt(V**2 + W**2)

for k in range(imgr.shape[0]):
    # Plot vectors.
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plt.imshow(imgr[k], cmap=cm.gray)
    ax.quiver(X[::4, ::4], Y[::4, ::4], W[k, ::4, ::4], -V[k, ::4, ::4], col[k, ::4, ::4])
    fig.savefig('results/of/{0}-vel-{1}.png'.format(name, k))

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plt.imshow(imgr[k], cmap=cm.gray)
    strm = ax.streamplot(X, Y, W[k], V[k], density=2, color=col[k], linewidth=1, cmap=cm.coolwarm)
    fig.savefig('results/of/{0}-streamlines-{1}.png'.format(name, k))