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
from matplotlib import cm
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ofmc.util.roihelpers import compute_error
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

# Set font style.
font = {'family': 'sans-serif',
        'serif': ['DejaVu Sans'],
        'weight': 'normal',
        'size': 30}
plt.rc('font', **font)
plt.rc('text', usetex=True)

# Set colormap.
cmap = cm.viridis

# Streamlines.
density = 1
linewidth = 2
arrowstyle = '-'

# Set output quality.
dpi = 100


def saveimage(path: str, name: str, img: np.array, title=None):
    """Takes a path string, a filename, and an array and saves the plotted
    array.

    Args:
        path (str): The path to save the image to.
        name (str): The filename.
        img (np.array): The 2D array.
        title(str): An optional title.

    Returns:
    """
    if not os.path.exists(path):
        os.makedirs(path)

    # Plot image.
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(img, cmap=cm.gray)
    if title is not None:
        ax.set_title(title)

    # Create colourbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # Save figure.
    fig.savefig(os.path.join(path, '{0}.png'.format(name)),
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def saveimage_nolegend(path: str, name: str, img: np.array):
    """Takes a path string, a filename, and an array and saves the plotted
    array without legend.

    Args:
        path (str): The path to save the image to.
        name (str): The filename.
        img (np.array): The 2D array.

    Returns:
    """
    if not os.path.exists(path):
        os.makedirs(path)

    # Plot image.
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(img, cmap=cm.gray)
    # plt.axis('off')

    # Save figure.
    fig.savefig(os.path.join(path, '{0}.png'.format(name)),
                dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def savesource(path: str, name: str, img: np.array):
    """Takes a path string, a filename, and an array and saves the plotted
    array.

    Args:
        path (str): The path to save the image to.
        name (str): The filename.
        img (np.array): The 2D array.

    Returns:
    """
    if not os.path.exists(path):
        os.makedirs(path)

    # Plot image.
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(img, cmap=cmap)
    # ax.set_title('Source')

    # Create colourbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # Save figure.
    fig.tight_layout()
    fig.savefig(os.path.join(path, '{0}-source.png'.format(name)),
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def savevelocity(path: str, name: str, img: np.array, vel: np.array, **kwargs):
    """Takes a path string, a filename, and an array and saves the plotted
    array.

    Args:
        path (str): The path to save the image to.
        name (str): The filename.
        img (np.array): A 2D array of intensities.
        vel (np.array): A 2D array with the velocity.
        T (float): Specifies the time interval [0, T] (optional).
    Returns:
    """
    T = kwargs.get('T', None)
    if T is None:
        T = 1.0

    if not os.path.exists(path):
        os.makedirs(path)

    maxvel = abs(vel).max()
    normi = mpl.colors.Normalize(vmin=-maxvel, vmax=maxvel)

    # Plot velocity.
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(vel, interpolation='nearest', norm=normi, cmap=cmap)
    # ax.set_title('Velocity')

    # Create colourbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # Save figure.
    fig.tight_layout()
    fig.savefig(os.path.join(path, '{0}-velocity.png'.format(name)),
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    m, n = vel.shape
    hx, hy = T / (m - 1), 1.0 / (n - 1)

    # Create grid for streamlines.
    Y, X = np.mgrid[0:m, 0:n]
    V = np.ones_like(X)

    # Plot streamlines.
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.imshow(img, cmap=cm.gray)
    # ax.set_title('Streamlines')
    strm = ax.streamplot(X, Y, vel * hx / hy, V, density=density,
                         arrowstyle=arrowstyle, color=vel, linewidth=linewidth,
                         norm=normi, cmap=cmap)
#    fig.colorbar(strm.lines, orientation='vertical')

    # Create colourbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(strm.lines, cax=cax, orientation='vertical')

    fig.tight_layout()
    fig.savefig(os.path.join(path, '{0}-streamlines.png'.format(name)),
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    # Save velocity profile after cut.
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.plot(vel[5])
    # ax.set_title('Velocity profile right after the cut')

    fig.tight_layout()
    fig.savefig(os.path.join(path, '{0}-profile.png'.format(name)),
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def savestrainrate(path: str, name: str, img: np.array, vel: np.array):
    """Takes a path string, a filename, and two arrays and saves the plotted
    array.

    Args:
        path (str): The path to save the image to.
        name (str): The filename.
        img (np.array): The 2D array.

    Returns:
    """
    if not os.path.exists(path):
        os.makedirs(path)

    m, n = img.shape
    hy = 1.0 / (n - 1)
    sr = np.gradient(vel, hy, axis=1)

    maxsr = abs(sr).max()
    normi = mpl.colors.Normalize(vmin=-maxsr, vmax=maxsr)

    # Plot velocity.
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(sr, interpolation='nearest', norm=normi, cmap=cmap)
    # ax.set_title('Strain rate')

    # Create colourbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # Save figure.
    fig.tight_layout()
    fig.savefig(os.path.join(path, '{0}-strainrate.png'.format(name)),
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def saveroi(path: str, name: str, img: np.array, roi):
    """Takes a path string, a filename, an array, and a roi, and saves the
    plotted array with splines superimposed.

    Args:
        path (str): The path to save the image to.
        name (str): The filename.
        img (np.array): The 2D array.

    Returns:
    """
    if not os.path.exists(path):
        os.makedirs(path)

    # Plot image.
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(img, cmap=cm.gray)
    # plt.axis('off')
    # ax.set_title('Manual tracks')

    for v in roi:
        plt.plot(roi[v]['x'], roi[v]['y'], 'C3', lw=2)

    # Create colourbar.
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.1)
    # fig.colorbar(im, cax=cax, orientation='vertical')

    # Save figure.
    fig.tight_layout()
    fig.savefig(os.path.join(path, '{0}-roi.png'.format(name)),
                dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def savespl(path: str, name: str, img: np.array, roi, spl):
    """Takes a path string, a filename, an array, and a roi, and saves the
    plotted array with splines superimposed.

    Args:
        path (str): The path to save the image to.
        name (str): The filename.
        img (np.array): The 2D array.

    Returns:
    """
    if not os.path.exists(path):
        os.makedirs(path)

    m, n = img.shape

    # Determine min/max velocity.
    maxvel = -np.inf
    for v in roi:
        y = roi[v]['y']
        derivspl = spl[v].derivative()
        maxvel = max(maxvel, max(abs(derivspl(y) * m / n)))

    # Determine colour coding.
    normi = mpl.colors.Normalize(vmin=-maxvel, vmax=maxvel)

    # Plot image.
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.imshow(img, cmap=cm.gray)
    # plt.axis('off')
    # ax.set_title('Velocity of spline')

    # Plot splines.
    for v in roi:
        # Compute derivative of spline.
        derivspl = spl[v].derivative()

        y = roi[v]['y']
        y = np.arange(y[0], y[-1] + 0.5, 0.5)

        points = np.array([spl[v](y), y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap=cmap, norm=normi)
        lc.set_array(derivspl(y) * m / n)
        lc.set_linewidth(2)
        plt.gca().add_collection(lc)

    # Create colourbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=normi)

    # Save figure.
    fig.tight_layout()
    fig.savefig(os.path.join(path, '{0}-spline.png'.format(name)),
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def saveerror(path: str, name: str, img: np.array, vel: np.array, roi, spl):
    """Takes a path string, a name, two arrays, a roi, and splines, and saves
    the plotted array with splines superimposed.

    Args:
        path (str): The path to save the image to.
        name (str): The filename.
        img (np.array): The image.
        vel (np.array): The velocity.
        roi: A roi instance.
        spl: Fitted spolines.

    Returns:
    """
    if not os.path.exists(path):
        os.makedirs(path)

    # Interpolate velocity.
    m, n = vel.shape

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.imshow(img, cmap=cm.gray)
    # ax.set_title('Error along tacks')

    # Evaluate velocity for each spline.
    error = compute_error(vel, roi, spl)
    # Compute maximum error.
    maxerror = 0
    for v in roi:
        maxerror = max(maxerror, max(abs(error[v])))

    # Determine colour coding.
    normi = mpl.colors.Normalize(vmin=0, vmax=maxerror)

    # Evaluate velocity for each spline.
    for v in roi:
        y = roi[v]['y']

        # Interpolate velocity.
        y = np.arange(y[0], y[-1] + 0.5, 0.5)
        # x = np.array(spl[v](y))

        points = np.array([spl[v](y), y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap=cmap, norm=normi)

        lc.set_array(error[v])
        lc.set_linewidth(2)
        plt.gca().add_collection(lc)

    # Create colourbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=normi)

    # Save figure.
    fig.tight_layout()
    fig.savefig(os.path.join(path, '{0}-error.png'.format(name)),
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def save_spl_streamlines(path: str, name: str, img: np.array, vel: np.array,
                         roi, spl):
    """Takes a path string, a name, two arrays, a roi, and splines, and saves
    the plotted array with splines superimposed.

    Args:
        path (str): The path to save the image to.
        name (str): The filename.
        img (np.array): The image.
        vel (np.array): The velocity.
        roi: A roi instance.
        spl: Fitted spolines.

    Returns:
    """
    if not os.path.exists(path):
        os.makedirs(path)

    m, n = img.shape

    # Determine max. velocity of splines.
    maxvel = -np.inf
    for v in roi:
        y = roi[v]['y']
        derivspl = spl[v].derivative()
        maxvel = max(maxvel, max(abs(derivspl(y) * m / n)))

    # Determine max. of splines and computed velocities.
    maxvel = max(maxvel, abs(vel).max())
    normi = mpl.colors.Normalize(vmin=-maxvel, vmax=maxvel)

    m, n = vel.shape
    hx, hy = 1.0 / (m - 1), 1.0 / (n - 1)

    # Create grid for streamlines.
    Y, X = np.mgrid[0:m, 0:n]
    V = np.ones_like(X)

    # Plot streamlines.
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.imshow(img, cmap=cm.gray)
    # ax.set_title('Streamlines and splines')

    # Plot splines and velocity.
    for v in roi:
        y = roi[v]['y']
        # Compute derivative of spline.
        derivspl = spl[v].derivative()

        points = np.array([spl[v](y), y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap=cmap, norm=normi)
        lc.set_array(derivspl(y) * m / n)
        lc.set_linewidth(2)
        plt.gca().add_collection(lc)

    # Plot streamlines.
    strm = ax.streamplot(X, Y, vel * hx / hy, V, density=density,
                         arrowstyle=arrowstyle, color=vel, linewidth=linewidth,
                         norm=normi, cmap=cmap)

    # Create colourbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(strm.lines, cax=cax, orientation='vertical', norm=normi,
                 cmap=cmap)

    fig.tight_layout()
    fig.savefig(os.path.join(path, '{0}-streamlines-splines.png'.format(name)),
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def save_roi_streamlines(path: str, name: str, img: np.array, vel: np.array,
                         roi):
    """Takes a path string, a filename, two arrays, and a roi, and saves the
    plotted array with tracks superimposed.

    Args:
        path (str): The path to save the image to.
        name (str): The filename.
        img (np.array): The image.
        vel (np.array): The velocity.
        roi: A roi instance.

    Returns:
    """
    if not os.path.exists(path):
        os.makedirs(path)

    # Determine max. of splines and computed velocities.
    maxvel = abs(vel).max()
    normi = mpl.colors.Normalize(vmin=-maxvel, vmax=maxvel)

    m, n = vel.shape
    hx, hy = 1.0 / (m - 1), 1.0 / (n - 1)

    # Create grid for streamlines.
    Y, X = np.mgrid[0:m, 0:n]
    V = np.ones_like(X)

    # Plot streamlines.
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.imshow(img, cmap=cm.gray)
    # ax.set_title('Streamlines and splines')

    # Plot splines and velocity.
    for v in roi:
        plt.plot(roi[v]['x'], roi[v]['y'], 'C3', lw=2)

    # Plot streamlines.
    strm = ax.streamplot(X, Y, vel * hx / hy, V, density=density,
                         arrowstyle=arrowstyle, color=vel, linewidth=linewidth,
                         norm=normi, cmap=cmap)

    # Create colourbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(strm.lines, cax=cax, orientation='vertical', norm=normi,
                 cmap=cmap)

    fig.tight_layout()
    fig.savefig(os.path.join(path, '{0}-streamlines-roi.png'.format(name)),
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def save_spl_curves(path: str, name: str, img: np.array,
                    roi, spl, curves):
    """Takes a path string, a name, two arrays, a roi, and splines, and saves
    the plotted array with splines superimposed.

    Args:
        path (str): The path to save the image to.
        name (str): The filename.
        img (np.array): The image.
        vel (np.array): The velocity.
        roi: A roi instance.
        spl: Fitted spolines.

    Returns:
    """
    if not os.path.exists(path):
        os.makedirs(path)

    # Plot image.
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.imshow(img, cmap=cm.gray)
    # ax.set_title('Splines and characteristics.')

    # Plot splines.
    for v in roi:
        # Plot roi.
        y = roi[v]['y']
        points = np.array([spl[v](y), y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments)
        lc.set_linewidth(2)
        lc.set_color('C3')
        plt.gca().add_collection(lc)

        # Plot characteristics.
        y = np.arange(y[0], y[-1] + 1, 1)
        c = curves[v]
        points = np.array([c, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments)
        lc.set_linewidth(2)
        lc.set_color('C0')
        plt.gca().add_collection(lc)

    fig.tight_layout()
    fig.savefig(os.path.join(path, '{0}-splines-curves.png'.format(name)),
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)
