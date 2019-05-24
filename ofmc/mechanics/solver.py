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
import numpy as np
import scipy.ndimage.filters
import scipy.sparse
import scipy.sparse.linalg
import sys


# Model parameters.
class ModelParams:
    # Viscosity of the cortex.
    eta: float = 0.5
    # Friction with the membrane.
    xi: float = 0.1
    # Contractility modulus.
    chi: float = 1.5
    # Define time of simulated laser ablation.
    t_cut = 0.025
    # Define width of cut. Interval will be [0.5-k/2, 0.5+k/2].
    k = 0.05
    # Rate of myosin adsorption.
    k_on: float = 200.0
    # Rate of myosin desorption.
    k_off: float = 10.0


# Solver parameters.
class SolverParams:
    # Number of grid points of cell-centred grid.
    n: int = 300
    # Number of output time steps (not including initial conditions).
    m: int = 100
    # Maximum time to run the time stepping.
    T: float = 0.1
    # Courant number (CFL-condition) that determines maximum step size in time.
    CFL: float = 0.98
    # Define step size.
    dt: float = 2.5e-5
    # Define parameter for smoothing of absolute value function.
    delta: float = 1e-4


def stress_matrix(n: int, dx: float, eta: float, xi: float):
    """Computes the stress in 1D for given concentration.

    Creates system matrix for the differential equation:

    (1 - \\frac{\\eta \\partial_{xx}}{\\xi}) \\sigma = \\chi c_{a},

    with zero-Neumann boundary conditions.

    Args:
        n (int): Number of grid points.
        dx (float): Mesh size dx > 0.
        eta (float): Parameter eta > 0.
        xi (float): Parameter xi > 0.

    Returns:
        csr_matrix: A sparse matrix of size (n, n).

    """
    # Create entries for diagonals.
    d = np.ones((n, 3)) * np.array([-1, 2 + xi * dx**2 / eta, -1])

    # Create sparse matrix.
    A = scipy.sparse.spdiags(d.transpose(), [-1, 0, 1], n, n)
    A = scipy.sparse.csr_matrix(A)

    # Incorporate zero-Neumann boundary conditions.
    A[0, 0] = 1 + xi * dx**2 / eta
    A[n - 1, n - 1] = 1 + xi * dx**2 / eta
    return A


def solve_stress_velocity(A, c: np.array, dx: float, chi: float, eta: float,
                          xi: float) -> (np.array, np.array):
    """Computes stress and velocity.

    The stress sigma obeys zero Neumann boundary conditions.
    The velocity v obeys zero Dirichlet boundary conditions.

    Args:
       A: A system matrix.
       c (np.array): An array of shape (1, n) of concentration.
       dx (float): Mesh size.
       chi (float): Parameter chi > 0.
       eta (float): Parameter eta > 0.
       xi (float): Parameter xi > 0.

    Returns:
        np.array: Arrays of shape (1, n + 2) with stress.
        np.array: Arrays of shape (1, n + 1) with velocity.

    """
    # Compute right-hand side.
    b = xi * dx**2 * chi * c / eta
    # Compute stress according to linear system.
    sigma = scipy.sparse.linalg.spsolve(A, b).transpose()
    # Respect boundary conditions by repeating boundary values.
    sigma = np.pad(sigma, 1, 'edge')

    # Compute velocity according to \\partial_{x} \\sigma = \\xi v.
    v = np.diff(sigma) / (dx * xi)
    return sigma, v


def check_cfl(v: np.array, dt: float, dx: float, CFL: float):
    """Checks if CFL condition for explicit time stepping is violated.

    Note that if violated sys.exit() is called!

    Args:
       v  (np.array): An array with velocities.
       dt (float): Step size.
       dx (float): Mesh size.
       CFL (float): Constant such that np.max(np.abs(v)) * dt / dx <= CFL.

    """
    cmax = CFL * dx / np.max(np.abs(v))
    if cmax < dt:
        print('CFL condition violated as dt={0} > {1}!\n'.format(dt, cmax))
        sys.exit()


def sabs(x: np.array, delta: float):
    """Returns smoothed absolute value function for each element in an array.

    Args:
       x  (np.array): An array.
       delta (float): A float > 0.

    Returns:
        np.array: An array same size as x.

    """
    return np.where(np.abs(x) < delta,
                    x**2 / (2 * delta) + delta / 2, np.abs(x))


def flux(v: np.array, c: np.array, delta: float):
    """Computes the flux.

    Takes velocities v living on staggered grid [0, dx, ..., 1] and a
    concentration c that lives on cell-centred grid [dx/2, ..., 1 - dx/2] and
    returns the flux, which also lives on the staggered grid. Since v is zero
    on the boundaries the flux is also zero on the boundary nodes at 0 and 1.

    The flux is computed as

    f(i) = 0.5 * v(i) (c(i) + c(i - 1))
            - 0.5 * sabs(v(i), delta) (c(i) - c(i - 1)).

    Args:
        v (np.array): Velocity vector of shape (1, n + 1).
        c (np.array): Concentration vector of shape (1, n).
        delta (float): A parameter > 0 used to smooth the abs function.

    Returns:
        np.array: An array of shape (1, n + 1) with the transport.

    """
    f = 0.5 * (v[1:-1] * (c[1:] + c[0:-1]) -
               sabs(v[1:-1], delta) * (c[1:] - c[0:-1]))
    return np.pad(f, 1, 'constant', constant_values=0)


def dump(var_dump: np.array, var: np.array, idx: int):
    """Stores a row in an array at a given index.

    Args:
        var_dump (np.array): An array of shape (m, n).
        var (np.array): An array of shape (n,).
        idx (int): An index idx < n.
    """
    var_dump[idx, :] = var


def solve(mp: ModelParams, sp: SolverParams, rho_init, ca_init, x: np.array,
          **kwargs):
    """Solves the following system of PDEs with explicit Euler time-stepping on
    a staggered grid.

    TODO: Add description and PDE system.

    Args:
        mp (ModelParams): An instance of ModelParams.
        sp (SolverParams): An instance of SolverParams.
        rho_init: A function fun(x) that gives the initial concentration at x.
        ca_init: A function fun(x) that gives the initial concentration at x.
        x (np.array): An array of shape (k,) of initial tracer positions.
        vel: A function vel(t, x) that gives the velocity (optional).
    Returns:
        np.array: A 2D array of size (m + 1, n) with rho.
        np.array: A 2D array of size (m + 1, n) with ca.
        np.array: A 2D array of size (m + 1, n + 1) with v.
        np.array: A 2D array of size (m + 1, n + 2) with sigma.
        np.array: A 2D array of size (m + 1, k) with tracer positions.
        int: The array index at which the artificial cut was applied.
    """
    vel = kwargs.get('vel', None)

    # Define start time.
    t = 0.0

    # Define iteration number and dump number.
    k = 0
    d = 1

    # Compute dump frequency.
    dumpfreq = max(np.int(sp.T / sp.dt / sp.m), 1)

    # Initialise matrix for tracers.
    x_dump = np.zeros((sp.m + 1, x.shape[0]))
    x_dump[0, :] = x

    # Compute grid spacing.
    dx = 1.0 / sp.n

    # Create cell-centred grid.
    X = np.linspace(dx / 2, 1 - dx / 2, num=sp.n)

    # Create staggered grid.
    Xs = np.linspace(0, 1, num=sp.n + 1)

    # Define simulated laser ablation on cell-centred grid.
    cut = [0 if x <= 0.5 + mp.k / 2 and x >= 0.5 - mp.k / 2 else 1 for x in X]
    # Smoothen cut.
    # cut = scipy.ndimage.filters.gaussian_filter1d(np.array(cut, dtype=float),
    #                                              sigma=1.5)
    cut_idx = 0

    # Concentration of actin.
    rho = np.vectorize(rho_init, otypes=[float])(X)
    rho_dump = np.zeros((sp.m + 1, sp.n))
    dump(rho_dump, rho, 0)

    # Concentration of attached myosin.
    ca = np.vectorize(ca_init, otypes=[float])(X) if mp.t_cut > t \
        else np.vectorize(ca_init, otypes=[float])(X) * cut
    ca_dump = np.zeros((sp.m + 1, sp.n))
    dump(ca_dump, ca, 0)

    # Create stress and velocity.
    sigma_dump = np.zeros((sp.m + 1, sp.n + 2))
    v_dump = np.zeros((sp.m + 1, sp.n + 1))

    # Create matrix for stress computation.
    A = stress_matrix(sp.n, dx, mp.eta, mp.xi)

    # Compute initial stress and velocity.
    if vel is None:
        sigma, v = solve_stress_velocity(A, ca, dx, mp.chi, mp.eta, mp.xi)
    else:
        v = np.vectorize(vel, otypes=[float])(t, Xs)
        sigma = np.zeros((1, sp.n + 2))
    dump(sigma_dump, sigma, 0)
    dump(v_dump, v, 0)

    # Check if CFL condition is violated.
    check_cfl(v, sp.dt, dx, sp.CFL)

    # Iterate.
    while(d <= sp.m):
        # Compute flux.
        fca = flux(v, ca, sp.delta)
        frho = flux(v, rho, sp.delta)

        # Add cut and save dump index.
        if t >= mp.t_cut - sp.dt / 2 and t < mp.t_cut + sp.dt / 2 \
                and mp.t_cut > 0:
            ca = ca * cut
            cut_idx = d

        # Perform explicit forward Euler step.
        rho_new = rho - sp.dt * np.diff(frho) / dx
        # ca = ca - sp.dt * (np.diff(fca) / dx - mp.k_on + mp.k_off * ca * rho)
        ca = ca - sp.dt * (np.diff(fca) / dx - mp.k_on + mp.k_off * ca)

        # Update variable rho.
        rho = rho_new

        # Compute stress and velocity.
        if vel is None:
            sigma, v = solve_stress_velocity(A, ca, dx, mp.chi, mp.eta, mp.xi)
        else:
            v = np.vectorize(vel, otypes=[float])(t, Xs)

        # Update position of tracers.
        x = x + sp.dt * np.interp(x, Xs, v)

        # Dump state.
        if(k % dumpfreq == 0 and k > 0):
            print('Iteration {0}, dump {1}.'.format(k, d))
            dump(rho_dump, rho, d)
            dump(ca_dump, ca, d)
            dump(sigma_dump, sigma, d)
            dump(v_dump, v, d)
            dump(x_dump, x, d)
            d += 1

        # Increase time and iteration number.
        t += sp.dt
        k += 1

        # Check if CFL condition is violated.
        check_cfl(v, sp.dt, dx, sp.CFL)

    # Compute conservation error.
    print_conservation_error(rho_dump, 'rho', dx)
    print_conservation_error(ca_dump, 'ca', dx)

    return rho_dump, ca_dump, v_dump, sigma_dump, x_dump, cut_idx


def print_conservation_error(var: np.array, name: str, dx: float):
    """Computes and prints the difference in the integrated mass between the
    first and the last time instant.

    Args:
        var (np.array): An array of shape (m, n).
        name (str): Name of the variabe.
    """
    conservation_error = dx * np.abs(np.sum(var[-1, :]) - np.sum(var[0, :]))
    print('Initial mass {0}: {1}'.format(name, dx * np.sum(var[0, :])))
    print('Conservation error {0}: {1}'.format(name, conservation_error))
