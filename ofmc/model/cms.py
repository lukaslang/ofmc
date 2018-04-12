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
from dolfin import triangle
from dolfin import Expression
from dolfin import Function
from dolfin import FiniteElement
from dolfin import TrialFunctions
from dolfin import TestFunctions
from dolfin import FunctionSpace
from dolfin import UnitSquareMesh
import ofmc.util.dolfinhelpers as dh
import ofmc.util.numpyhelpers as nh

# Note that for the periodic case v and k should be of shape (m, n-1).
# There seems to be a problem with the handling of periodic vector-valued
# spaces, see:
# https://bitbucket.org/fenics-project/dolfin/issues/405
# https://fenicsproject.org/qa/6462/split-mixed-spaces-does-not-preserve-number-degrees-freedom/


def cms1d_weak_solution(V: FunctionSpace,
                        f: Function, ft: Function, fx: Function,
                        alpha0: float, alpha1: float,
                        alpha2: float, alpha3: float) -> (Function, Function):
    """Solves the weak formulation of the Euler-Lagrange equations of the L2-H1
    mass conserving flow functional with source and with spatio-temporal
    regularisation for a 1D image sequence, i.e.

    f_x * (f_t + f_x * v + f * v_x - k) \
        - (f * (f_t + f_x * v + f * v_x - k))_x \
        - alpha0 * v_xx - alpha1 * v_tt = 0
    - (f_t + f_x * v + f * v_x - k) - alpha2 * k_xx - alpha3 * k_tt = 0

    with natural boundary conditions.

    Takes a function space, a one-dimensional image sequence f and its
    partial derivatives with respect to time and space, and returns a solution.

    Args:
        f (Function): A 1D image sequence.
        ft (Function): Partial derivative of f wrt. time.
        fx (Function): Partial derivative of f wrt. space.
        alpha0 (float): The spatial regularisation parameter for v.
        alpha1 (float): The temporal regularisation parameter for v.
        alpha2 (float): The spatial regularisation parameter for k.
        alpha3 (float): The temporal regularisation parameter for k.

    Returns:
        v (Function): The velocity.
        k (Function): The source.
    """
    # Define trial and test functions.
    v, k = TrialFunctions(V)
    w1, w2 = TestFunctions(V)

    # Define weak formulation.
    A = (- (fx * v + f * v.dx(1) - k) * (fx * w1 + f * w1.dx(1))
         - alpha0 * v.dx(1) * w1.dx(1)
         - alpha1 * v.dx(0) * w1.dx(0)) * dx \
        + ((fx * v + f * v.dx(1) - k) * w2
           - alpha2 * k.dx(1) * w2.dx(1)
           - alpha3 * k.dx(0) * w2.dx(0)) * dx
    b = ft * (fx * w1 + f * w1.dx(1) - w2) * dx

    # Compute solution.
    v = Function(V)
    solve(A == b, v)

    # Recover solution.
    v, k = v.split(deepcopy=True)

    # Evaluate and print residual and functional value.
    res = abs(ft + fx * v + f * v.dx(1) - k)
    func = 0.5 * (res ** 2 + alpha0 * v.dx(1) ** 2 + alpha1 * v.dx(0) ** 2
                  + alpha2 * k.dx(1) ** 2 + alpha3 * k.dx(0) ** 2)
    print('Res={0}, Func={1}\n'.format(assemble(res * dx),
                                       assemble(func * dx)))
    return v, k


def cms1d_exp(m: int, n: int,
              f: Expression, ft: Expression, fx: Expression,
              alpha0: float, alpha1: float,
              alpha2: float, alpha3: float) -> (np.array, np.array):
    """Computes the L2-H1 mass conserving flow with source for a 1D image sequence.

    Args:
        m (int): Number of temporal sampling points.
        n (int): Number of spatial sampling points.
        f (Expression): 1D image sequence.
        ft (Expression): Partial derivative of f wrt. time.
        fx (Expression): Partial derivative of f wrt. space.
        alpha0 (float): The spatial regularisation parameter for v.
        alpha1 (float): The temporal regularisation parameter for v.
        alpha2 (float): The spatial regularisation parameter for k.
        alpha3 (float): The temporal regularisation parameter for k.

    Returns:
        v (np.array): A velocity array of shape (m, n).
        k (np.array): A source array of shape (m, n).

    """
    # Define mesh and function space.
    mesh = UnitSquareMesh(m - 1, n - 1)
    V = dh.create_vector_function_space(mesh, 'default')

    # Compute velocity and source.
    v, k = cms1d_weak_solution(V, f, ft, fx, alpha0, alpha1, alpha2, alpha3)

    # Convert back to array and return.
    v = dh.funvec2img(v.vector().get_local(), m, n)
    k = dh.funvec2img(k.vector().get_local(), m, n)
    return v, k


def cms1d_exp_pb(m: int, n: int,
                 f: Expression, ft: Expression, fx: Expression,
                 alpha0: float, alpha1: float,
                 alpha2: float, alpha3: float) -> (np.array, np.array):
    """Computes the L2-H1 mass conserving flow with source for a 1D image
    sequence with periodic boundary.

    Args:
        m (int): Number of temporal sampling points.
        n (int): Number of spatial sampling points.
        f (Expression): 1D image sequence.
        ft (Expression): Partial derivative of f wrt. time.
        fx (Expression): Partial derivative of f wrt. space.
        alpha0 (float): The spatial regularisation parameter for v.
        alpha1 (float): The temporal regularisation parameter for v.
        alpha2 (float): The spatial regularisation parameter for k.
        alpha3 (float): The temporal regularisation parameter for k.

    Returns:
        v (np.array): A velocity array of shape (m, n).
        k (np.array): A source array of shape (m, n).

    """
    # Define mesh and function space.
    mesh = UnitSquareMesh(m - 1, n - 1)
    V = dh.create_vector_function_space(mesh, 'periodic')

    # Compute velocity.
    v, k = cms1d_weak_solution(V, f, ft, fx, alpha0, alpha1, alpha2, alpha3)

    # Convert back to array and return.
    v = dh.funvec2img(v.vector().get_local(), m, n)
    k = dh.funvec2img(k.vector().get_local(), m, n)
    return v, k


def cms1d_img(img: np.array,
              alpha0: float, alpha1: float,
              alpha2: float, alpha3: float,
              deriv) -> (np.array, np.array):
    """Computes the L2-H1 mass conserving flow with source for a 1D image
    sequence.

    Allows to specify how to approximate partial derivatives of f numerically.

    Args:
        img (np.array): 1D image sequence of shape (m, n), where m is the
                        number of time steps and n is the number of pixels.
        alpha0 (float): The spatial regularisation parameter for v.
        alpha1 (float): The temporal regularisation parameter for v.
        alpha2 (float): The spatial regularisation parameter for k.
        alpha3 (float): The temporal regularisation parameter for k.
        deriv (str): Specifies how to approximate pertial derivatives.
                     When set to 'mesh' it uses FEniCS built in function.
                     When set to 'fd' it uses finite differences.

    Returns:
        v (np.array): A velocity array of shape (m, n).
        k (np.array): A source array of shape (m, n).

    """
    # Check for valid arguments.
    valid = {'mesh', 'fd'}
    if deriv not in valid:
        raise ValueError("Argument 'deriv' must be one of %r." % valid)

    # Create mesh.
    m, n = img.shape
    mesh = UnitSquareMesh(m - 1, n - 1)

    # Define function spaces.
    V = dh.create_function_space(mesh, 'default')
    W = dh.create_vector_function_space(mesh, 'default')

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
    v, k = cms1d_weak_solution(W, f, ft, fx, alpha0, alpha1, alpha2, alpha3)

    # Convert back to array and return.
    v = dh.funvec2img(v.vector().get_local(), m, n)
    k = dh.funvec2img(k.vector().get_local(), m, n)
    return v, k


def cms1d_img_pb(img: np.array,
                 alpha0: float, alpha1: float,
                 alpha2: float, alpha3: float,
                 deriv='mesh') -> np.array:
    """Computes the L2-H1 mass conserving flow with source for a 1D image
    sequence with periodic spatial boundary.

    Args:
        img (np.array): 1D image sequence of shape (m, n), where m is the
                        number of time steps and n is the number of pixels.
        alpha0 (float): The spatial regularisation parameter for v.
        alpha1 (float): The temporal regularisation parameter for v.
        alpha2 (float): The spatial regularisation parameter for k.
        alpha3 (float): The temporal regularisation parameter for k.
        deriv (str): Specifies how to approximate pertial derivatives.
                     When set to 'mesh' it uses FEniCS built in function.

    Returns:
        v (np.array): A velocity array of shape (m, n).
        k (np.array): A source array of shape (m, n).

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
    W = dh.create_vector_function_space(mesh, 'periodic')

    # Convert array to function.
    f = Function(V)
    f.vector()[:] = dh.img2funvec_pb(img)

    # Compute partial derivatives.
    ft, fx = f.dx(0), f.dx(1)

    # Compute velocity.
    v, k = cms1d_weak_solution(W, f, ft, fx, alpha0, alpha1, alpha2, alpha3)

    # Convert back to array and return.
    v = dh.funvec2img(v.vector().get_local(), m, n)
    k = dh.funvec2img(k.vector().get_local(), m, n)
    return v, k


def cms1dl2(img: np.array, alpha0: float, alpha1: float,
            alpha2: float) -> (np.array, np.array):
    """Computes the L2-H1-L2 mass conserving flow with source for a 1D image
    sequence.

    Takes a one-dimensional image sequence and returns a minimiser of the
    L2-H1-L2 mass conservation functional with source term with spatio-temporal
    regularisation for the velocity and L2 regularisation for the source.

    Args:
        img (np.array): A 1D image sequence of shape (m, n), where m is the
                        number of time steps and n is the number of pixels.
        alpha0 (float): The spatial regularisation parameter for v.
        alpha1 (float): The temporal regularisation parameter for v.
        alpha2 (float): The regularisation parameter for k.

    Returns:
        v: A velocity array of shape (m, n).
        k: A source array of shape (m, n).

    """
    # Create mesh.
    [m, n] = img.shape
    mesh = UnitSquareMesh(m-1, n-1)

    # Define function space and functions.
    P = FiniteElement('P', triangle, 1)
    W = FunctionSpace(mesh, P * P)
    v, k = TrialFunctions(W)
    w1, w2 = TestFunctions(W)

    # Convert image to function.
    V = FunctionSpace(mesh, 'CG', 1)
    f = Function(V)
    f.vector()[:] = dh.img2funvec(img)

    # Define derivatives of data.
    ft = Function(V)
    ftv = np.diff(img, axis=0) * (m - 1)
    ftv = np.concatenate((ftv, ftv[-1, :].reshape(1, n)), axis=0)
    ft.vector()[:] = dh.img2funvec(ftv)

    fx = Function(V)
    fxv = np.gradient(img, 1.0 / (n - 1), axis=1)
    fx.vector()[:] = dh.img2funvec(fxv)

    # Define weak formulation.
    A = - (fx*v + f*v.dx(1) - k) * (fx*w1 + f*w1.dx(1))*dx \
        - alpha0*v.dx(1)*w1.dx(1)*dx - alpha1*v.dx(0)*w1.dx(0)*dx \
        + (fx*v + f*v.dx(1) - k)*w2*dx - alpha2*k*w2*dx
    b = ft*(fx*w1 + f*w1.dx(1))*dx - ft*w2*dx

    # Compute solution.
    v = Function(W)
    solve(A == b, v)

    # Recover solution.
    v, k = v.split(deepcopy=True)

    # Convert back to array.
    vel = dh.funvec2img(v.vector().get_local(), m, n)
    k = dh.funvec2img(k.vector().get_local(), m, n)
    return vel, k
