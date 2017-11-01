#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dolfin import *
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy import misc
from scipy import ndimage

# Set regularisation parameter.
alpha0 = 1e-1
alpha1 = 1e-3
beta0 = 1e-5
beta1 = 1e-5
gamma = 1e-2

# Load image.
img = misc.imread('../../data/E2PSB1PMT-560-C1.png')
#img = misc.imread('../../data/E5PSB1PMT-1.tif')
#img = misc.imread('../../data/E8-PSB1PMT.tif')
#img = misc.imread('../../data/PSB1_E1PMT.tif')
#img = misc.imread('../../data/PSB4PMT.tif')
#img = misc.imread('../../data/PSB5PMT.tif')
#img = misc.imread('../../data/epidermalejonction.png')
#img = misc.imread('../../data/artificialcut_small.png')

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
#P = FiniteElement('P', triangle, 1)
#W = FunctionSpace(mesh, P * P)

# Initialise source.
kprev = Function(V)
vprev = Function(V)

for i in range(10):
    print('Iteration ', i)
    
    ## Solve for velocity field.
    
    v = TrialFunction(V)
    w = TestFunction(V)
    
    # Define weak formulation.
    A = - fx*fx*v*w*dx - fx*f*v*w.dx(1)*dx - f*fx*v.dx(1)*w*dx - f*f*v.dx(1)*w.dx(1)*dx - alpha0*v.dx(1)*w.dx(1)*dx - alpha1*v.dx(0)*w.dx(0)*dx - gamma*kprev.dx(1)*kprev.dx(1)*v*w*dx
    l = ft*fx*w*dx + ft*f*w.dx(1)*dx + kprev*fx*w*dx + kprev*f*w.dx(1)*dx + gamma*kprev.dx(1)*kprev.dx(0)*w*dx
    
    # Compute solution.
    v = Function(V)
    solve(A == l, v)
    vprev.assign(v)
    
#    plot(vprev)
#    plt.show()
    
    ## Solve for source.
    
    k = TrialFunction(V)
    w = TestFunction(V)
    
    # Define weak formulation.
    A = - k*w*dx - beta0*k.dx(1)*w.dx(1)*dx - beta1*k.dx(0)*w.dx(0)*dx - gamma*k.dx(0)*w.dx(0)*dx - gamma*k.dx(1)*v*w.dx(0)*dx - gamma*k.dx(0)*vprev*w.dx(1)*dx - gamma*k.dx(1)*vprev*vprev*w.dx(1)*dx
    l = - ft*w*dx - fx*vprev*w*dx - f*vprev.dx(1)*w*dx
    
    # Compute solution.
    k = Function(V)
    solve(A == l, k)
    kprev.assign(k)
    
#    plot(kprev)
#    plt.show()

# Recover solution.
#v, k = vfun.split(deepcopy=True)

# Create image from solution.
v2d = vertex_to_dof_map(V)
vel = np.zeros_like(img)
values = vprev.vector().array()[v2d]
for (i, j, v) in zip(x, y, values): vel[i, j] = v

source = np.zeros_like(img)
values = kprev.vector().array()[v2d]
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

# Create grid for streamlines.
Y, X = np.mgrid[0:m:1, 0:n:1]
V = np.ones_like(X)*hy

# Plot streamlines.
fig, ax = plt.subplots(figsize=(10, 10))
plt.imshow(img, cmap=cm.gray)

strm = ax.streamplot(X, Y, vel*hx, V, density=2, color=vel, linewidth=1, cmap=cm.coolwarm)
fig.colorbar(strm.lines, orientation='horizontal')

# Plot source.
fig, ax = plt.subplots()
cax = ax.imshow(source, interpolation='nearest', cmap=cm.coolwarm)
ax.set_title('Source')
cbar = fig.colorbar(cax, orientation='vertical')
plt.show()

