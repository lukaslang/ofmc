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
from dolfin import assemble
from dolfin import dx
from dolfin import dof_to_vertex_map
from dolfin import solve
from dolfin import Expression
from dolfin import Function
from dolfin import FunctionSpace
from dolfin import TestFunction
from dolfin import TestFunctions
from dolfin import TrialFunction
from dolfin import TrialFunctions
from dolfin import UnitCubeMesh
from dolfin import UnitSquareMesh
from dolfin import VectorFunctionSpace
import numpy as np
import ofmc.util.dolfinhelpers as dh
import ofmc.util.numpyhelpers as nh


def of1d_weak_solution(V: FunctionSpace,
                       f: Function, ft: Function, fx: Function,
                       alpha0: float, alpha1: float) \
                       -> (Function, float, float):
    """Solves the weak formulation of the Euler-Lagrange equations of the L2-H1
    Horn-Schunck optical flow functional with spatio-temporal regularisation
    for a 1D image sequence, i.e.

    f_x * (f_t + f_x * v) - alpha0 * v_xx - alpha1 * v_tt = 0

    with zero Neumann boundary conditions v_x = 0 and v_t = 0.

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
        res (float): The residual.
        fun (float): The function value.
    """
    # Define trial and test functions.
    v = TrialFunction(V)
    w = TestFunction(V)

    # Define weak formulation.
    A = (fx * fx * v * w
         + alpha0 * v.dx(1) * w.dx(1)
         + alpha1 * v.dx(0) * w.dx(0)) * dx
    b = -fx * ft * w * dx

    # Compute and return solution.
    v = Function(V)
    solve(A == b, v)

    # Evaluate and print residual and functional value.
    res = abs(ft + fx*v)
    fun = 0.5 * (res ** 2 + alpha0 * v.dx(1) ** 2 + alpha1 * v.dx(0) ** 2)
    print('Res={0}, Func={1}'.format(assemble(res * dx),
                                     assemble(fun * dx)))
    return v, assemble(res * dx), assemble(fun * dx)


def of1d_exp(m: int, n: int,
             f: Expression, ft: Expression, fx: Expression,
             alpha0: float, alpha1) -> (np.array, float, float):
    """Computes the L2-H1 optical flow for a 1D image sequence.

    Takes a one-dimensional image sequence and partial derivatives, and returns
    a minimiser of the Horn-Schunck functional with spatio-temporal
    regularisation.

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
        res (float): The residual.
        fun (float): The function value.
    """
    # Define mesh and function space.
    mesh = UnitSquareMesh(m - 1, n - 1)
    V = dh.create_function_space(mesh, 'default')

    # Compute velocity.
    v, res, fun = of1d_weak_solution(V, f, ft, fx, alpha0, alpha1)

    # Convert to array and return.
    return dh.funvec2img(v.vector().get_local(), m, n), res, fun


def of1d_exp_pb(m: int, n: int,
                f: Expression, ft: Expression, fx: Expression,
                alpha0: float, alpha1) -> (np.array, float, float):
    """Computes the L2-H1 optical flow for a 1D image sequence with periodic
    boundary in space.

    Takes a one-dimensional image sequence and partial derivatives, and returns
    a minimiser of the Horn-Schunck functional with spatio-temporal
    regularisation.

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
        res (float): The residual.
        fun (float): The function value.
    """
    # Define mesh and function space.
    mesh = UnitSquareMesh(m - 1, n - 1)
    V = dh.create_function_space(mesh, 'periodic')

    # Compute velocity.
    v, res, fun = of1d_weak_solution(V, f, ft, fx, alpha0, alpha1)

    # Convert to array and return.
    return dh.funvec2img_pb(v.vector().get_local(), m, n), res, fun


def of1d_img(img: np.array, alpha0: float, alpha1: float, deriv) \
             -> (np.array, float, float):
    """Computes the L2-H1 optical flow for a 1D image sequence.

    Takes a one-dimensional image sequence and returns a minimiser of the
    Horn-Schunck functional with spatio-temporal regularisation.

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
        res (float): The residual.
        fun (float): The function value.
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
    v, res, fun = of1d_weak_solution(V, f, ft, fx, alpha0, alpha1)

    # Convert to array and return.
    return dh.funvec2img(v.vector().get_local(), m, n), res, fun


def of1d_img_pb(img: np.array, alpha0: float, alpha1: float,
                deriv='mesh') -> (np.array, float, float):
    """Computes the L2-H1 optical flow for a 1D image sequence with periodic
    boundary in space.

    Takes a one-dimensional image sequence and returns a minimiser of the
    Horn-Schunck functional with spatio-temporal regularisation.

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
        res (float): The residual.
        fun (float): The function value.
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
    v, res, fun = of1d_weak_solution(V, f, ft, fx, alpha0, alpha1)

    # Convert to array and return.
    return dh.funvec2img_pb(v.vector().get_local(), m, n), res, fun


def of2dmcs(img1: np.array, img2: np.array, alpha0: float, alpha1: float,
            beta0: float, beta1: float) -> (np.array, np.array):
    """Computes the L2-H1 optical flow for a 2D two-channel image sequence and
    source for the second channel (optical flow 2d multi-channel with source).

    Takes a two-dimensional image sequence and returns a minimiser of the
    Horn-Schunck functional with source and with spatio-temporal
    regularisation for both velocity and source.

    Args:
        img1 (np.array): A 2D image sequence of shape (t, m, n), where t is the
                         number of time steps and (n, n) is the number of
                         pixels.
        img2 (np.array): A 2D image sequence of shape (t, m, n), where t is the
                         number of time steps and (n, n) is the number of
                         pixels.
        alpha0 (float):  The spatial regularisation parameter.
        alpha1 (float):  The temporal regularisation parameter.

    Returns:
        v (np.array): A velocity array of shape (t, m, n, 2).
        k (np.array): A source array of shape (t, m, n).

    """
    # Create mesh.
    [t, m, n] = img1.shape
    mesh = UnitCubeMesh(t-2, m-1, n-1)

    # Define function space and functions.
    V = FunctionSpace(mesh, 'CG', 1)
    W = VectorFunctionSpace(mesh, 'CG', 1, dim=3)
    v1, v2, k = TrialFunctions(W)
    w1, w2, w3 = TestFunctions(W)

    # Convert image to function.
    f1, f2 = Function(V), Function(V)
    f1.vector()[:] = dh.imgseq2funvec(img1[0:-1])
    f2.vector()[:] = dh.imgseq2funvec(img2[0:-1])

    # Define function to compute temporal derivative.
    def time_deriv(img: np.array) -> Function:
        # Evaluate function at vertices.
        mc = mesh.coordinates().reshape((-1, 3))
        hx, hy, hz = 1./(t-2), 1./(m-1), 1./(n-1)
        x = np.array(mc[:, 0]/hx, dtype=int)
        y = np.array(mc[:, 1]/hy, dtype=int)
        z = np.array(mc[:, 2]/hz, dtype=int)

        # Map pixel values to vertices.
        d2v = dof_to_vertex_map(V)

        # Compute derivative wrt. time.
        imgt = img[1:] - img[0:-1]
        ftv = imgt[x, y, z]

        # Create function.
        ft = Function(V)
        ft.vector()[:] = ftv[d2v]
        return ft

    # Compute temporal derivatives.
    f1t = time_deriv(img1)
    f2t = time_deriv(img2)

    # Define derivatives of data.
    f1x, f1y = f1.dx(1), f1.dx(2)
    f2x, f2y = f2.dx(1), f2.dx(2)

    # Define weak formulation.
    A = f1x*(f1x*v1 + f1y*v2)*w1*dx + f2x*(f2x*v1 + f2y*v2 - k)*w1*dx \
        + f1y*(f1x*v1 + f1y*v2)*w2*dx + f2y*(f2x*v1 + f2y*v2 - k)*w2*dx \
        - (f2x*v1 + f2y*v2 - k)*w3*dx \
        + alpha0*v1.dx(1)*w1.dx(1)*dx + alpha0*v1.dx(2)*w1.dx(2)*dx \
        + alpha1*v1.dx(0)*w1.dx(0)*dx \
        + alpha0*v2.dx(1)*w2.dx(1)*dx + alpha0*v2.dx(2)*w2.dx(2)*dx \
        + alpha1*v2.dx(0)*w2.dx(0)*dx \
        + beta0*k.dx(1)*w3.dx(1)*dx + beta0*k.dx(2)*w3.dx(2)*dx \
        + beta1*k.dx(0)*w3.dx(0)*dx
    b = - f1x*f1t*w1*dx - f2x*f2t*w1*dx \
        - f1y*f1t*w2*dx - f2y*f2t*w2*dx \
        + f2t*w3*dx

    # Compute solution.
    v = Function(W)
    solve(A == b, v, [], solver_parameters={"linear_solver": "cg"})

    # Split solution into functions.
    v1, v2, k = v.split(deepcopy=True)

    # Convert back to arrays.
    v1 = dh.funvec2imgseq(v1.vector().get_local(), t-1, m, n)
    v2 = dh.funvec2imgseq(v2.vector().get_local(), t-1, m, n)
    k = dh.funvec2imgseq(k.vector().get_local(), t-1, m, n)
    return (np.stack((v1, v2), axis=3), k)
