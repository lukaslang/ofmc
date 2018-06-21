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
from dolfin import assemble
from dolfin import dx
from dolfin import solve
from dolfin import Expression
from dolfin import Function
from dolfin import TrialFunction
from dolfin import TestFunction
from dolfin import FunctionSpace
from dolfin import UnitSquareMesh
import ofmc.util.dolfinhelpers as dh
import ofmc.util.numpyhelpers as nh


def cm1d_weak_solution(V: FunctionSpace,
                       f: Function, ft: Function, fx: Function,
                       alpha0: float, alpha1: float) -> Function:
    """Solves the weak formulation of the Euler-Lagrange equations of the L2-H1
    mass conserving flow functional with spatio-temporal regularisation
    for a 1D image sequence, i.e.

    f_x * (f_t + f_x * v + f * v_x) - (f * (f_t + f_x * v + f * v_x))_x
        - alpha0 * v_xx - alpha1 * v_tt = 0

    with natural boundary conditions.

    Takes a function space, a one-dimensional image sequence f and its
    partial derivatives with respect to time and space, and returns a solution.

    Args:
        f (Function): A 1D image sequence.
        ft (Function): Partial derivative of f wrt. time.
        fx (Function): Partial derivative of f wrt. space.
        alpha0 (float): The spatial regularisation parameter.
        alpha1 (float): The temporal regularisation parameter.

    Returns:
        v (Function): The velocity.
    """
    # Define trial and test functions.
    v = TrialFunction(V)
    w = TestFunction(V)

    # Define weak formulation.
    A = (-(fx * v + f * v.dx(1)) * (fx * w + f * w.dx(1))
         - alpha0 * v.dx(1) * w.dx(1)
         - alpha1 * v.dx(0) * w.dx(0)) * dx
    b = ft * (fx * w + f * w.dx(1)) * dx

    # Compute and return solution.
    v = Function(V)
    solve(A == b, v)

    # Evaluate and print residual and functional value.
    res = abs(ft + fx * v + f * v.dx(1))
    func = 0.5 * (res ** 2 + alpha0 * v.dx(1) ** 2 + alpha1 * v.dx(0) ** 2)
    print('Res={0}, Func={1}'.format(assemble(res * dx),
                                     assemble(func * dx)))
    return v


def cm1d_exp(m: int, n: int,
             f: Expression, ft: Expression, fx: Expression,
             alpha0: float, alpha1: float) -> np.array:
    """Computes the L2-H1 mass conserving flow for a 1D image sequence.

    Args:
        m (int): Number of temporal sampling points.
        n (int): Number of spatial sampling points.
        f (Expression): 1D image sequence.
        ft (Expression): Partial derivative of f wrt. time.
        fx (Expression): Partial derivative of f wrt. space.
        alpha0 (float): Spatial regularisation parameter.
        alpha1 (float): Temporal regularisation parameter.

    Returns:
        v (np.array): A velocity array of shape (m, n).

    """
    # Define mesh and function space.
    mesh = UnitSquareMesh(m - 1, n - 1)
    V = dh.create_function_space(mesh, 'default')

    # Compute velocity.
    v = cm1d_weak_solution(V, f, ft, fx, alpha0, alpha1)

    # Convert to array and return.
    return dh.funvec2img(v.vector().get_local(), m, n)


def cm1d_exp_pb(m: int, n: int,
                f: Expression, ft: Expression, fx: Expression,
                alpha0: float, alpha1: float) -> np.array:
    """Computes the L2-H1 mass conserving flow for a 1D image sequence with
    periodic boundary.

    Args:
        m (int): Number of temporal sampling points.
        n (int): Number of spatial sampling points.
        f (Expression): 1D image sequence.
        ft (Expression): Partial derivative of f wrt. time.
        fx (Expression): Partial derivative of f wrt. space.
        alpha0 (float): Spatial regularisation parameter.
        alpha1 (float): Temporal regularisation parameter.

    Returns:
        v (np.array): A velocity array of shape (m, n).

    """
    # Define mesh and function space.
    mesh = UnitSquareMesh(m - 1, n - 1)
    V = dh.create_function_space(mesh, 'periodic')

    # Compute velocity.
    v = cm1d_weak_solution(V, f, ft, fx, alpha0, alpha1)

    # Convert to array and return.
    return dh.funvec2img_pb(v.vector().get_local(), m, n)


def cm1d_img(img: np.array, alpha0: float, alpha1: float, deriv) -> np.array:
    """Computes the L2-H1 mass conserving flow for a 1D image sequence.

    Allows to specify how to approximate partial derivatives of f numerically.

    Args:
        img (np.array): 1D image sequence of shape (m, n), where m is the
                        number of time steps and n is the number of pixels.
        alpha0 (float): Spatial regularisation parameter.
        alpha1 (float): Temporal regularisation parameter.
        deriv (str): Specifies how to approximate pertial derivatives.
                     When set to 'mesh' it uses FEniCS built in function.
                     When set to 'fd' it uses finite differences.

    Returns:
        v (np.array): A velocity array of shape (m, n).

    """
    # Check for valid arguments.
    valid = {'mesh', 'fd'}
    if deriv not in valid:
        raise ValueError("Argument 'deriv' must be one of %r." % valid)

    # Create mesh.
    m, n = img.shape
    mesh = UnitSquareMesh(m - 1, n - 1)

    # Define function space.
    V = dh.create_function_space(mesh, 'default')

    # Convert array to function.
    f = Function(V)
    f.vector()[:] = dh.img2funvec(img)

    # Compute partial derivatives.
    if deriv is 'mesh':
        ft, fx = f.dx(0), f.dx(1)
    if deriv is 'fd':
        imgt, imgx = nh.partial_derivatives(img)
        ft, fx = Function(V), Function(V)
        ft.vector()[:] = dh.img2funvec(imgt)
        fx.vector()[:] = dh.img2funvec(imgx)

    # Compute velocity.
    v = cm1d_weak_solution(V, f, ft, fx, alpha0, alpha1)

    # Convert to array and return.
    return dh.funvec2img(v.vector().get_local(), m, n)


def cm1d_img_pb(img: np.array, alpha0: float, alpha1: float,
                deriv='mesh') -> np.array:
    """Computes the L2-H1 mass conserving flow for a 1D image sequence with
    periodic spatial boundary.

    Allows to specify how to approximate partial derivatives of f numerically.

    Note that the last column of img is ignored.

    Args:
        img (np.array): 1D image sequence of shape (m, n), where m is the
                        number of time steps and n is the number of pixels.
        alpha0 (float): Spatial regularisation parameter.
        alpha1 (float): Temporal regularisation parameter.
        deriv (str): Specifies how to approximate pertial derivatives.
                     When set to 'mesh' it uses FEniCS built in function.

    Returns:
        v (np.array): A velocity array of shape (m, n).

    """
    # Check for valid arguments.
    valid = {'mesh'}
    if deriv not in valid:
        raise ValueError("Argument 'deriv' must be one of %r." % valid)

    # Create mesh.
    m, n = img.shape
    mesh = UnitSquareMesh(m - 1, n - 1)

    # Define function space.
    V = dh.create_function_space(mesh, 'periodic')

    # Convert array to function.
    f = Function(V)
    f.vector()[:] = dh.img2funvec_pb(img)

    # Compute partial derivatives.
    ft, fx = f.dx(0), f.dx(1)

    # Compute velocity.
    v = cm1d_weak_solution(V, f, ft, fx, alpha0, alpha1)

    # Convert to array and return.
    return dh.funvec2img_pb(v.vector().get_local(), m, n)


def cm1dsource(img: np.array, k: np.array, alpha0: float,
               alpha1: float) -> np.array:
    """Computes the L2-H1 mass conserving flow for a 1D image sequence and a
    given source.

    Takes a one-dimensional image sequence and a source, and returns a
    minimiser of the L2-H1 mass conservation functional with spatio-temporal
    regularisation.

    Args:
        img (np.array): A 1D image sequence of shape (m, n), where m is the
                        number of time steps and n is the number of pixels.
        k (np.array):   A 1D image sequence of shape (m, n).
        alpha0 (float): The spatial regularisation parameter.
        alpha1 (float): The temporal regularisation parameter.

    Returns:
        v: A velocity array of shape (m, n).

    """
    # Create mesh.
    [m, n] = img.shape
    mesh = UnitSquareMesh(m-1, n-1)

    # Define function space and functions.
    V = FunctionSpace(mesh, 'CG', 1)
    v = TrialFunction(V)
    w = TestFunction(V)

    # Convert image to function.
    f = Function(V)
    f.vector()[:] = dh.img2funvec(img)

    # Convert source to function.
    g = Function(V)
    g.vector()[:] = dh.img2funvec(k)

    # Define derivatives of data.
    ft = Function(V)
    ftv = np.diff(img, axis=0) * (m - 1)
    ftv = np.concatenate((ftv, ftv[-1, :].reshape(1, n)), axis=0)
    ft.vector()[:] = dh.img2funvec(ftv)

    fx = Function(V)
    fxv = np.gradient(img, 1 / (n - 1), axis=1)
    fx.vector()[:] = dh.img2funvec(fxv)

    ft = f.dx(0)
    fx = f.dx(1)

    # Define weak formulation.
    A = - (fx*v + f*v.dx(1)) * (fx*w + f*w.dx(1))*dx \
        - alpha0*v.dx(1)*w.dx(1)*dx - alpha1*v.dx(0)*w.dx(0)*dx
    b = ft*(fx*w + f*w.dx(1))*dx - g*(fx*w + f*w.dx(1))*dx

    # Compute solution.
    v = Function(V)
    solve(A == b, v)

    # Convert back to array.
    vel = dh.funvec2img(v.vector().get_local(), m, n)
    return vel


def cm1dvelocity(img: np.array, vel: np.array, alpha0: float,
                 alpha1: float) -> np.array:
    """Computes the source for a L2-H1 mass conserving flow for a 1D image
    sequence and a given velocity.

    Takes a one-dimensional image sequence and a velocity, and returns a
    minimiser of the L2-H1 mass conservation functional with spatio-temporal
    regularisation.

    Args:
        img (np.array): A 1D image sequence of shape (m, n), where m is the
                        number of time steps and n is the number of pixels.
        vel (np.array): A 1D image sequence of shape (m, n).
        alpha0 (float): The spatial regularisation parameter.
        alpha1 (float): The temporal regularisation parameter.

    Returns:
        k: A source array of shape (m, n).

    """
    # Create mesh.
    [m, n] = img.shape
    mesh = UnitSquareMesh(m-1, n-1)

    # Define function space and functions.
    V = FunctionSpace(mesh, 'CG', 1)
    k = TrialFunction(V)
    w = TestFunction(V)

    # Convert image to function.
    f = Function(V)
    f.vector()[:] = dh.img2funvec(img)

    # Convert velocity to function.
    v = Function(V)
    v.vector()[:] = dh.img2funvec(vel)

    # Define derivatives of data.
    ft = Function(V)
    ftv = np.diff(img, axis=0) * (m - 1)
    ftv = np.concatenate((ftv, ftv[-1, :].reshape(1, n)), axis=0)
    ft.vector()[:] = dh.img2funvec(ftv)

    fx = Function(V)
    fxv = np.gradient(img, 1 / (n - 1), axis=1)
    fx.vector()[:] = dh.img2funvec(fxv)

    # Define weak formulation.
    A = k*w*dx + alpha0*k.dx(1)*w.dx(1)*dx + alpha1*k.dx(0)*w.dx(0)*dx
    b = (ft + v.dx(1)*f + v*fx)*w*dx

    # Compute solution.
    k = Function(V)
    solve(A == b, k)

    # Convert back to array.
    k = dh.funvec2img(k.vector().get_local(), m, n)
    return k
