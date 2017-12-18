#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dolfin import *
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy import misc
from scipy import ndimage

# Create mesh.
m = 100
mesh = UnitIntervalMesh(m)

# Defining the function spaces
V_dg = FunctionSpace(mesh, "DG", 2)
V_cg = FunctionSpace(mesh, "CG", 1)

# Create velocity function.
#v = Expression('exp(-pow(x[0] - 0.5, 2.0)/(2*sigmasq))/sqrt(2*pi*sigmasq)', sigmasq=0.01, degree=2)
#v = interpolate(v, V_cg)
v = project(Constant(-0.1), V_cg)
maxvel = np.amax(np.abs((v.vector().array())))

# Source term.
#k = Expression('20*exp(-pow(x[0] - 0.5, 2.0)/(2*sigmasq))/sqrt(2*pi*sigmasq)', sigmasq=0.01, degree=2)
#k = interpolate(k, V_cg)
k = project(Constant(0.0), V_cg)

# Mesh-related functions.
n = FacetNormal(mesh)
h = mesh.hmin()

#vn = (v*n + abs(v*n)) / 2.0
vv = as_vector((v, ))
vn = (dot(vv, n) + abs(dot(vv, n))) / 2.0

# Test and trial functions.
f = TrialFunction(V_dg)
phi = TestFunction(V_dg)

# Define time step.
dt = 0.2*h/abs(maxvel)

#alpha = 1e-5

# Define weak formulation.
#A = f.dx(0)*w*dx - f*v*w.dx(1)*dx + jump(f*v)*avg(w)*dS + avg(f*v)*jump(w)*dS + (alpha*jump(f)*jump(w)/avg(h))*dS
#l = k*w*dx

# Define bilinear form.
a_mass = f*phi*dx
a_int = - phi.dx(0)*f*v*dx
a_flux = jump(phi)*(vn('+')*f('+') - vn('-')*f('-'))*dS + phi*vn*f*ds # + (alpha*jump(f*vn)*jump(phi)/avg(h))*dS
a_source = - k*phi*dx

M = assemble(a_mass)
arhs = -dt*(a_int + a_flux + a_source)

# Define initial condition.
#f0 = Expression('exp(-pow(x[0] - 0.5, 2.0)/(2*sigmasq))/sqrt(2*pi*sigmasq)', sigmasq=0.001, degree=2)
#f0 = Expression('sin(5*pi*x[0])', degree=2)
#f0 = interpolate(f0, V_cg)

#f0 = Function(V_cg)
#vec = np.zeros_like(f0.vector().array())
#vec[np.int(4*m/10):np.int(6*m/10)] = 10
#f0.vector()[:] = vec

class MyExpression(Expression):
    def eval(self, value, x):
        value[0] = max(0, 0.1 - abs(x[0] - 0.5))
    def value_shape(self):
        return ()
f0 = MyExpression(degree=1)
f0 = interpolate(f0, V_cg)

class RectangleExpression(Expression):
    def eval(self, value, x):
        value[0] = 1 if x[0] >= 0.45 and x[0] <= 0.55 else 0
    def value_shape(self):
        return ()
#f0 = RectangleExpression(degree=1)
#f0 = interpolate(f0, V_cg)


df1 = Function(V_dg)
f1 = Function(V_dg)
df2 = Function(V_dg)
f2 = Function(V_dg)
df3 = Function(V_dg)

t = 0.0
T = 1
k = 0
dumpfreq = 1

# Initialise solution array.
f = interpolate(f0, V_dg)
initial_tracer_mass = assemble(f*dx)

iter = np.int(T/dt)
print('Running iterations ', iter)

fsol = np.zeros((np.int(iter/dumpfreq), m+1))
fsol[0, :] = f0.vector().array().reshape(1, m+1)

while(t < (T-dt/2)):    
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

    if(k % dumpfreq == 0):
        print('Iteration ', k)
        fproj = interpolate(f, V_cg)
        fsol[np.int(k/dumpfreq), :] = fproj.vector().array().reshape(1, m+1)
        
        #plot(f)
        #plt.show()

    t += dt
    k += 1

plt.figure()
plt.imshow(fsol, cmap=plt.cm.coolwarm)
plt.colorbar()
plt.show()

# Compute error.
conservation_error = assemble(f*dx) - initial_tracer_mass
print(conservation_error)

# Plot solution.
#p = plot(fsol)
#plt.colorbar(p)
#plt.show()

# Project for plotting.
#W = FunctionSpace(mesh, "CG", 1)
#fp = project(f, W)

# Plot solution.
#p = plot(fp)
#plt.colorbar(p)
##plt.show()

#f0 = Expression("x[0]", degree=2)
#f0 = project(f0, V)
#p = plot(f0)
#plt.colorbar(p)
#plt.show()