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
import datetime
import glob
import imageio
import logging
import numpy as np
import os
import re
import warnings
import ofmc.external.tifffile as tiff
import ofmc.util.pyplothelpers as ph
import ofmc.util.roihelpers as rh
from read_roi import read_roi_zip
from scipy import ndimage
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
from ofmc.model.cmscr import cmscr1d_img
import math
import ofmc.mechanics.solver as solver
import scipy.stats as stats
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

ffc_logger = logging.getLogger('FFC')
ffc_logger.setLevel(logging.WARNING)
ufl_logger = logging.getLogger('UFL')
ufl_logger.setLevel(logging.WARNING)

# Set path with data.
datapath = ('/Users/lukaslang/'
            'Dropbox (Cambridge University)/Drosophila/Data from Elena')

# Set path where results are saved.
resultpath = 'results/{0}'.format(
        datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
if not os.path.exists(resultpath):
    os.makedirs(resultpath)

print('Processing {0}'.format(datapath))


def load_kymo(datfolder, dat):
    # Identify Kymograph and do sanity check.
    kymos = glob.glob('{0}/SUM_Reslice of {1}*.tif'.format(datfolder, dat))
    if len(kymos) != 1:
        warnings.warn("No Kymograph found!")

    # Extract name of kymograph and replace whitespaces.
    name = os.path.splitext(os.path.basename(kymos[0]))[0]
    name = re.sub(' ', '_', name)

    # Load kymograph.
    return imageio.imread(kymos[0]), name


def prepareimage(img: np.array) -> np.array:
    # Take image after cut.
    img = img[6:, :]

    # Filter image.
    img = ndimage.gaussian_filter(img, sigma=1.0)

    # Normalise to [0, 1].
    img = np.array(img, dtype=float)
    img = (img - img.min()) / (img.max() - img.min())
    return img


# Choose dataset.
gen = 'SqAX3_SqhGFP42_GAP43_TM6B'
# dat = '190216E8PSB1'
dat = '190216E5PSB2'



# Load kymograph.
datfolder = os.path.join(datapath, os.path.join(gen, dat))
img, name = load_kymo(datfolder, dat)

# Prepare image.
imgp = prepareimage(img)

#imgp = imgp[:, 40:125]

m, n = imgp.shape

# Set regularisation parameters for cmscr1d.
alpha0 = 5e-3
alpha1 = 5e-3
alpha2 = 1e-3
alpha3 = 1e-3
beta = 2.5e-3

# Compute velocity and source.
v, k, res, fun, converged = cmscr1d_img(imgp, alpha0, alpha1,
                                        alpha2, alpha3,
                                        beta, 'mesh')


slope, intercept, r_value, p_value, std_err = stats.linregress(imgp.flatten(),
                                                               k.flatten())


fig, ax = plt.subplots(figsize=(10, 5))
plt.scatter(imgp.flatten(), k.flatten(), s=1)
plt.plot(np.linspace(0, 1, 10), slope * np.linspace(0, 1, 10) + intercept,
         color='red')
ax.set_title('c vs k')
plt.xlabel('c')
plt.ylabel('k')
fig.tight_layout()
plt.show()
plt.close(fig)


resfolder = os.path.join(os.path.join(resultpath, gen), dat)

# Plot and save figures.
path = os.path.join(*[resultpath, 'cmscr1d', gen, dat])
if not os.path.exists(resultpath):
    os.makedirs(resultpath)

ph.saveimage(path, name, imgp)
ph.savevelocity(path, name, imgp, v)
ph.savesource(path, name, k)

# Create model and solver parameters.
mp = solver.ModelParams()
mp.t_cut = float('Inf')
mp.k_on = 0
mp.k_off = 0
eta = 0.5
xi = 0.1
chi = 1.5

sp = solver.SolverParams()
sp.n = n
sp.m = m
sp.T = 1
sp.dt = 1e-3

# Compute grid spacing.
dx = 1.0 / sp.n
dt = 1.0 / (sp.m - 1)

grid = np.linspace(dx / 2, 1 - dx / 2, num=sp.n)
ca_interp = interp1d(x=grid, y=imgp[0, :], kind='linear', bounds_error=False,
                     fill_value='extrapolate')


# Define initial values.
def ca_init(x: float) -> float:
    return ca_interp(x)


def rho_init(x: float) -> float:
    return 0


# Initialise tracers.
x = np.array(np.linspace(0, 1, num=25))

gridt = np.linspace(0, 1, sp.m)
gridx = np.linspace(dx / 2, 1 - dx / 2, num=sp.n)
v_int = RegularGridInterpolator(points=[gridt, gridx], values=v,
                                bounds_error=False, fill_value=0)

# X, Y = np.meshgrid(gridt, gridx, indexing='ij')
# plt.imshow(v_int((X, Y))), plt.show()

# plt.plot(v_int((0.1, gridx))), plt.show()


# Create velocity interpolation.
def vel(t: float, x: float) -> float:
    return v_int((t, x))


# Run solver.
rho, ca, v_, sigma, x, idx = solver.solve(mp, sp, rho_init, ca_init, x,
                                          vel=vel)

# Plot image.
fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(imgp, cmap=cm.viridis)
# plt.plot(x * sp.n, np.linspace(0, sp.m, sp.m + 1))
ax.set_title('imgp')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
fig.colorbar(im, cax=cax, orientation='vertical')
fig.tight_layout()
plt.show()
plt.close(fig)

# Plot image.
fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(ca, cmap=cm.viridis)
# plt.plot(x * sp.n, np.linspace(0, sp.m, sp.m + 1))
ax.set_title('ca')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
fig.colorbar(im, cax=cax, orientation='vertical')
fig.tight_layout()
plt.show()
plt.close(fig)

err = np.abs(ca[:-1, :] - imgp)

# Plot image.
fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(err, cmap=cm.viridis)
# plt.plot(x * sp.n, np.linspace(0, sp.m, sp.m + 1))
ax.set_title('error')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
fig.colorbar(im, cax=cax, orientation='vertical')
fig.tight_layout()
plt.show()
plt.close(fig)
