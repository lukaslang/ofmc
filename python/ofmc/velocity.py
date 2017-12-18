e #!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dolfin import *
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy import misc
from scipy import ndimage

# Create mesh.
m = 50
mesh = UnitIntervalMesh(m)

# Defining the function spaces
V_cg = FunctionSpace(mesh, "CG", 1)

a0 = 0.05
v0 = 1
tau0 = 1.0
tau1 = 0.2


def vel0(t):
    return v0 * np.exp(- t / tau0)

def vel(t, x):
    at = a0 + vel0(t) * t
    return vel0(t) * np.exp(- np.abs(x - at) / tau1) if x >= at else vel0(t) * x / at

T = 4
n = 100

v = np.zeros((n, 2*(m+1)-1))


rng = np.linspace(0, T, num=n)
for i in range(n):
    t = rng[i]
    
    class velt(Expression):
        def eval(self, value, x):
            value[0] = vel(t, x[0])
        def value_shape(self):
            return ()
    f0 = velt(degree=2)
    f0 = interpolate(f0, V_cg)
    f0_values = f0.vector().array()[:]
    
    
    
    v[i, :] = np.concatenate((-f0_values[0:-1], np.flip(f0_values, axis=0)), axis=0)
    
#    plt.plot(v[i, :])
#    plt.show()
#    print(t)

#v = np.flip(v, axis=1)

plt.plot(v[1, :])
plt.show()

plt.figure()
plt.imshow(v, cmap=plt.cm.coolwarm)
plt.colorbar()
plt.show()

# Create grid for streamlines.
#hx, hy = 1./(m-1), 1./(n-1)
hx = 1.0
hy = 1.0
Y, X = np.mgrid[0:v.shape[0]:1, 0:v.shape[1]:1]
V = np.ones_like(X)*hy

fig, ax = plt.subplots(figsize=(10, 10))
#plt.imshow(v, cmap=cm.gray)
strm = ax.streamplot(X, Y, v*hx, V, density=2, color=v, linewidth=1, cmap=cm.coolwarm)
ax.invert_yaxis()
fig.colorbar(strm.lines, orientation='horizontal')