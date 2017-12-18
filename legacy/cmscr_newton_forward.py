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
name = ['E2PSB1PMT-560-C1', 'e2_PSB4_kymograph1', 'E5PSB1PMT-1', 'E8-PSB1PMT', 'PSB1_E1PMT', 'PSB4PMT', 'PSB5PMT', 'epidermalejonction', 'artificialcut_small']
ending = ['png', 'tif', 'tif', 'tif', 'tif', 'tif', 'tif', 'png', 'png']

# Select file.
idx = 1
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

# Compute transport.

# Create mesh.
mm, nn = img.shape
mesh = UnitIntervalMesh(nn-1)

# Mesh-related functions.
n = FacetNormal(mesh)
h = mesh.hmin()

# Defining the function spaces
V_dg = FunctionSpace(mesh, "DG", 2)
V_cg = FunctionSpace(mesh, "CG", 1)

# Define time step.
dt = 0.1*h/np.amax(abs(vel))

# Define initial condition.
f0 = Function(V_cg)
f0.vector()[:] = img[0, :]

df1 = Function(V_dg)
f1 = Function(V_dg)
df2 = Function(V_dg)
f2 = Function(V_dg)
df3 = Function(V_dg)


t = 0.0
T = 1
iter = 0

niter = np.int(T/dt)
print('Running iterations ', niter)

dumpfreq = np.ceil(niter/mm)

# Initialise solution array.
f = project(f0, V_dg)
initial_tracer_mass = assemble(f*dx)


fsol = np.zeros((mm, nn))
fsol[0, :] = f0.vector().array().reshape(1, nn)

while(t < T - dt):
    # Create velocity function.
    v = Function(V_cg)
    v.vector()[:] = -vel[np.int(iter/dumpfreq), :]
    
    # Source term.
    k = Function(V_cg)
    k.vector()[:] = source[np.int(iter/dumpfreq), :]
        
    vv = as_vector((v, ))
    vn = (dot(vv, n) + abs(dot(vv, n))) / 2.0
    
    # Test and trial functions.
    D = TrialFunction(V_dg)
    phi = TestFunction(V_dg)
    
    # Define bilinear form.
    a_mass = D*phi*dx
    a_int = - phi.dx(0)*D*v*dx
    a_flux = jump(phi)*(vn('+')*D('+') - vn('-')*D('-'))*dS + phi*vn*D*ds # + (alpha*jump(f*vn)*jump(phi)/avg(h))*dS
    a_source = - k*phi*dx
    
    M = assemble(a_mass)
    arhs = -dt*(a_int + a_flux + a_source)
    
    L = assemble(action(arhs, f))
    solve(M, df1.vector(), L)

    f1.vector()[:] = f.vector().copy()
    f1.vector().axpy(1.0, df1.vector())
    L = assemble(action(arhs,f1))
    solve(M,df2.vector(),L)

    f2.vector()[:] = f.vector().copy()
    f2.vector().axpy(0.25, df1.vector())
    f2.vector().axpy(0.25, df2.vector())
    L = assemble(action(arhs, f2))
    solve(M, df3.vector(), L)

    f.vector().axpy((1.0/6.0), df1.vector())
    f.vector().axpy((1.0/6.0), df2.vector())
    f.vector().axpy((2.0/3.0), df3.vector())

    if(iter % dumpfreq == 0):
        print('Iteration ', iter)
        fproj = interpolate(f, V_cg)
        fsol[np.int(iter/dumpfreq), :] = fproj.vector().array().reshape(1, nn)

    t += dt
    iter += 1

plt.figure()
plt.imshow(fsol, cmap=plt.cm.coolwarm)
plt.colorbar()
plt.show()

# Compute error.
conservation_error = assemble(f*dx) - initial_tracer_mass
print(conservation_error)

plt.figure()
plt.imshow(abs(fsol - img))
plt.colorbar()
plt.show()