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
from dolfin import Constant
from dolfin import derivative
from dolfin import dx
from dolfin import DirichletBC
from dolfin import Function
from dolfin import FunctionSpace
from dolfin import Expression
from dolfin import solve
from dolfin import split
from dolfin import NonlinearVariationalProblem
from dolfin import NonlinearVariationalSolver
from dolfin import TestFunctions
from dolfin import TrialFunctions
from dolfin import UnitSquareMesh
from dolfin import VectorFunctionSpace
import numpy as np
import ofmc.util.dolfinhelpers as dh
import ofmc.util.numpyhelpers as nh

# Note that for the periodic case v and k should be of shape (m, n-1).
# There seems to be a problem with the handling of periodic vector-valued
# spaces, see:
# https://bitbucket.org/fenics-project/dolfin/issues/405
# https://fenicsproject.org/qa/6462/split-mixed-spaces-does-not-preserve-number-degrees-freedom/


def cmscr1d_weak_solution(V: VectorFunctionSpace,
                          f: Function, ft: Function, fx: Function,
                          alpha0: float, alpha1: float,
                          alpha2: float, alpha3: float,
                          beta: float, bcs=[]) \
                          -> (Function, Function, float, float, bool):
    """Solves the weak formulation of the Euler-Lagrange equations of the L2-H1
    mass conserving flow functional with source and with spatio-temporal and
    convective regularisation for a 1D image sequence with natural boundary
    conditions.

    Args:
        V (VectorFunctionSpace): The function space.
        f (Function): A 1D image sequence.
        ft (Function): Partial derivative of f wrt. time.
        fx (Function): Partial derivative of f wrt. space.
        alpha0 (float): The spatial regularisation parameter for v.
        alpha1 (float): The temporal regularisation parameter for v.
        alpha2 (float): The spatial regularisation parameter for k.
        alpha3 (float): The temporal regularisation parameter for k.
        beta (float): The convective regularisation parameter for.
        bcs: boundary conditions (optional).

    Returns:
        v (Function): The velocity.
        k (Function): The source.
        res (float): The residual.
        fun (float): The function value.
        status (bool): True if Newton's method converged.
    """
    # Define trial and test functions.
    w = Function(V)
    v, k = split(w)
    w1, w2 = TestFunctions(V)

    # Define weak formulation.
    A = (- (ft + fx * v + f * v.dx(1) - k) * (fx * w1 + f * w1.dx(1))
         - alpha0 * v.dx(1) * w1.dx(1)
         - alpha1 * v.dx(0) * w1.dx(0)
         - beta * k.dx(1) * (k.dx(0) + k.dx(1) * v) * w1) * dx \
        + ((ft + fx * v + f * v.dx(1) - k) * w2
           - alpha2 * k.dx(1) * w2.dx(1)
           - alpha3 * k.dx(0) * w2.dx(0)
           - beta * (k.dx(0) * w2.dx(0) + k.dx(1) * v * w2.dx(0)
                     + k.dx(0) * v * w2.dx(1)
                     + k.dx(1) * v * v * w2.dx(1))) * dx

    # Compute Gateaux derivative.
    DA = derivative(A, w)

    # Set up solver.
    problem = NonlinearVariationalProblem(A, w, bcs, DA)
    solver = NonlinearVariationalSolver(problem)

    # Set solver parameters.
    prm = solver.parameters
    prm['newton_solver']['error_on_nonconvergence'] = False
    prm['newton_solver']['maximum_iterations'] = 15
    prm['newton_solver']['absolute_tolerance'] = 1e-10
    prm['newton_solver']['relative_tolerance'] = 1e-10

    # Compute solution via Newton method.
    itern, status = solver.solve()
    if not bool(status):
        print("Newton's method did not converge within {0} ".format(itern) +
              "iterations for parameters alpha0={0}, ".format(alpha0) +
              "alpha1={0}, alpha3={1}, ".format(alpha1, alpha2) +
              "alpha4={0}, beta={1}".format(alpha3, beta))

    # Recover solution.
    v, k = w.split(deepcopy=True)

    # Evaluate and print residual and functional value.
    res = abs(ft + fx * v + f * v.dx(1) - k)
    fun = 0.5 * (res ** 2 + alpha0 * v.dx(1) ** 2 + alpha1 * v.dx(0) ** 2
                 + alpha2 * k.dx(1) ** 2 + alpha3 * k.dx(0) ** 2
                 + beta * (k.dx(0) + k.dx(1) * v) ** 2)
    print('Res={0}, Func={1}'.format(assemble(res * dx),
                                     assemble(fun * dx)))
    return v, k, assemble(res * dx), assemble(fun * dx), bool(status)


def cmscr1d_exp(m: int, n: int,
                f: Expression, ft: Expression, fx: Expression,
                alpha0: float, alpha1: float,
                alpha2: float, alpha3: float,
                beta: float) -> (np.array, np.array, float, float, bool):
    """Computes the L2-H1 mass conserving flow with source for a 1D image
    sequence with spatio-temporal and convective regularisation.

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
        beta (float): The convective regularisation parameter.

    Returns:
        v (np.array): A velocity array of shape (m, n).
        k (np.array): A source array of shape (m, n).
        res (float): The residual.
        fun (float): The function value.
        converged (bool): True if Newton's method converged.

    """
    # Define mesh and function space.
    mesh = UnitSquareMesh(m - 1, n - 1)
    V = dh.create_vector_function_space(mesh, 'default')

    # Compute velocity and source.
    v, k, res, fun, converged = cmscr1d_weak_solution(V, f, ft, fx,
                                                      alpha0, alpha1,
                                                      alpha2, alpha3, beta)

    # Convert back to array and return.
    v = dh.funvec2img(v.vector().get_local(), m, n)
    k = dh.funvec2img(k.vector().get_local(), m, n)
    return v, k, res, fun, converged


def cmscr1d_exp_pb(m: int, n: int,
                   f: Expression, ft: Expression, fx: Expression,
                   alpha0: float, alpha1: float,
                   alpha2: float, alpha3: float,
                   beta: float) -> (np.array, np.array, bool):
    """Computes the L2-H1 mass conserving flow with source for a 1D image
    sequence with spatio-temporal and convective regularisation with periodic
    spatial boundary.

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
        beta (float): The convective regularisation parameter.

    Returns:
        v (np.array): A velocity array of shape (m, n).
        k (np.array): A source array of shape (m, n).
        res (float): The residual.
        fun (float): The function value.
        converged (bool): True if Newton's method converged.

    """
    # Define mesh and function space.
    mesh = UnitSquareMesh(m - 1, n - 1)
    V = dh.create_vector_function_space(mesh, 'periodic')

    # Compute velocity.
    v, k, res, fun, converged = cmscr1d_weak_solution(V, f, ft, fx,
                                                      alpha0, alpha1,
                                                      alpha2, alpha3, beta)

    # Convert back to array and return.
    v = dh.funvec2img(v.vector().get_local(), m, n)
    k = dh.funvec2img(k.vector().get_local(), m, n)
    return v, k, res, fun, converged


def cmscr1d_img(img: np.array,
                alpha0: float, alpha1: float,
                alpha2: float, alpha3: float,
                beta: float, deriv, mesh=None, bc='natural') \
                -> (np.array, np.array, float, float, bool):
    """Computes the L2-H1 mass conserving flow with source for a 1D image
    sequence with spatio-temporal and convective regularisation.

    Allows to specify how to approximate partial derivatives of f numerically.

    Args:
        img (np.array): 1D image sequence of shape (m, n), where m is the
                        number of time steps and n is the number of pixels.
        alpha0 (float): The spatial regularisation parameter for v.
        alpha1 (float): The temporal regularisation parameter for v.
        alpha2 (float): The spatial regularisation parameter for k.
        alpha3 (float): The temporal regularisation parameter for k.
        beta (float): The convective regularisation parameter.
        deriv (str): Specifies how to approximate pertial derivatives.
                     When set to 'mesh' it uses FEniCS built in function.
                     When set to 'fd' it uses finite differences.
        mesh: A custom mesh (optional). Must have (m - 1, n - 1) cells.
        bc (str): One of {'natural', 'zero', 'zerospace'} for boundary
                    conditions for the velocity v (optional).

    Returns:
        v (np.array): A velocity array of shape (m, n).
        k (np.array): A source array of shape (m, n).
        res (float): The residual.
        fun (float): The function value.
        converged (bool): True if Newton's method converged.

    """
    # Check for valid arguments.
    valid = {'mesh', 'fd'}
    if deriv not in valid:
        raise ValueError("Argument 'deriv' must be one of %r." % valid)

    # Create mesh.
    m, n = img.shape
    if mesh is None:
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

    # Check for valid arguments.
    valid = {'natural', 'zero', 'zerospace'}
    if bc not in valid:
        raise ValueError("Argument 'bc' must be one of %r." % valid)

    # Define boundary conditions for velocity.
    if bc is 'natural':
        bc = []
    if bc is 'zero':
        bc = DirichletBC(W.sub(0), Constant(0), dh.DirichletBoundary())
    if bc is 'zerospace':
        bc = DirichletBC(W.sub(0), Constant(0), dh.DirichletBoundarySpace())

    # Compute velocity.
    v, k, res, fun, converged = cmscr1d_weak_solution(W, f, ft, fx,
                                                      alpha0, alpha1,
                                                      alpha2, alpha3, beta,
                                                      bcs=bc)

    # Convert back to array and return.
    v = dh.funvec2img(v.vector().get_local(), m, n)
    k = dh.funvec2img(k.vector().get_local(), m, n)
    return v, k, res, fun, converged


def cmscr1d_img_pb(img: np.array,
                   alpha0: float, alpha1: float,
                   alpha2: float, alpha3: float,
                   beta: float, deriv='mesh') \
                   -> (np.array, np.array, float, float, bool):
    """Computes the L2-H1 mass conserving flow with source for a 1D image
    sequence with spatio-temporal and convective regularisation with periodic
    spatial boundary.

    Args:
        img (np.array): 1D image sequence of shape (m, n), where m is the
                        number of time steps and n is the number of pixels.
        alpha0 (float): The spatial regularisation parameter for v.
        alpha1 (float): The temporal regularisation parameter for v.
        alpha2 (float): The spatial regularisation parameter for k.
        alpha3 (float): The temporal regularisation parameter for k.
        beta (float): The convective regularisation parameter.
        deriv (str): Specifies how to approximate pertial derivatives.
                     When set to 'mesh' it uses FEniCS built in function.

    Returns:
        v (np.array): A velocity array of shape (m, n).
        k (np.array): A source array of shape (m, n).
        res (float): The residual.
        fun (float): The function value.
        converged (bool): True if Newton's method converged.

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
    v, k, res, fun, converged = cmscr1d_weak_solution(W, f, ft, fx,
                                                      alpha0, alpha1,
                                                      alpha2, alpha3, beta)

    # Convert back to array and return.
    v = dh.funvec2img(v.vector().get_local(), m, n)
    k = dh.funvec2img(k.vector().get_local(), m, n)
    return v, k, res, fun, converged


def cmscr1dnewton(img: np.array, alpha0: float, alpha1: float, alpha2: float,
                  alpha3: float, beta: float) \
                  -> (np.array, np.array, float, float, bool):
    """Same as cmscr1d but doesn't use FEniCS Newton method.

    Args:
        img (np.array): A 1D image sequence of shape (m, n), where m is the
                        number of time steps and n is the number of pixels.
        alpha0 (float): The spatial regularisation parameter for v.
        alpha1 (float): The temporal regularisation parameter for v.
        alpha2 (float): The spatial regularisation parameter for k.
        alpha3 (float): The temporal regularisation parameter for k.
        beta (float): The parameter for convective regularisation.

    Returns:
        v: A velocity array of shape (m, n).
        k: A source array of shape (m, n).
        res (float): The residual.
        fun (float): The function value.
        converged (bool): True if Newton's method converged.

    """
    # Create mesh.
    [m, n] = img.shape
    mesh = UnitSquareMesh(m - 1, n - 1)

    # Define function space and functions.
    W = VectorFunctionSpace(mesh, 'CG', 1, dim=2)
    w = Function(W)
    v, k = split(w)
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
    fxv = np.gradient(img, 1 / (n - 1), axis=1)
    fx.vector()[:] = dh.img2funvec(fxv)

    # Define weak formulation.
    F = - (ft + fx*v + f*v.dx(1) - k) * (fx*w1 + f*w1.dx(1))*dx \
        - alpha0*v.dx(1)*w1.dx(1)*dx - alpha1*v.dx(0)*w1.dx(0)*dx \
        - beta*k.dx(1)*(k.dx(0) + k.dx(1)*v)*w1*dx \
        + (ft + fx*v + f*v.dx(1) - k)*w2*dx \
        - alpha2*k.dx(1)*w2.dx(1)*dx - alpha3*k.dx(0)*w2.dx(0)*dx \
        - beta*(k.dx(0)*w2.dx(0) + k.dx(1)*v*w2.dx(0)
                + k.dx(0)*v*w2.dx(1) + k.dx(1)*v*v*w2.dx(1))*dx

    # Define derivative.
    dv, dk = TrialFunctions(W)
    J = - (fx*dv + f*dv.dx(1) - dk)*(fx*w1 + f*w1.dx(1))*dx \
        - alpha0*dv.dx(1)*w1.dx(1)*dx - alpha1*dv.dx(0)*w1.dx(0)*dx \
        - beta*(k.dx(1)*(dk.dx(0) + dk.dx(1)*v) + k.dx(1)**2*dv
                + dk.dx(1)*(k.dx(0) + k.dx(1)*v))*w1*dx \
        + (fx*dv + f*dv.dx(1) - dk)*w2*dx \
        - alpha2*dk.dx(1)*w2.dx(1)*dx - alpha3*dk.dx(0)*w2.dx(0)*dx \
        - beta*(dv*k.dx(1) + dk.dx(0) + v*dk.dx(1))*w2.dx(0)*dx \
        - beta*(dv*k.dx(0) + 2*v*dv*k.dx(1) + v*dk.dx(0)
                + v*v*dk.dx(1))*w2.dx(1)*dx

    # Alternatively, use automatic differentiation.
    # J = derivative(F, w)

    # Define algorithm parameters.
    tol = 1e-10
    maxiter = 100

    # Define increment.
    dw = Function(W)

    # Run Newton iteration.
    niter = 0
    eps = 1
    res = 1
    while res > tol and niter < maxiter:
        niter += 1

        # Solve for increment.
        solve(J == -F, dw)

        # Update solution.
        w.assign(w + dw)

        # Compute norm of increment.
        eps = dw.vector().norm('l2')

        # Compute norm of residual.
        a = assemble(F)
        res = a.norm('l2')

        # Print statistics.
        print("Iteration {0} eps={1} res={2}".format(niter, eps, res))

    # Split solution.
    v, k = w.split(deepcopy=True)

    # Evaluate and print residual and functional value.
    res = abs(ft + fx * v + f * v.dx(1) - k)
    fun = 0.5 * (res ** 2 + alpha0 * v.dx(1) ** 2 + alpha1 * v.dx(0) ** 2
                 + alpha2 * k.dx(1) ** 2 + alpha3 * k.dx(0) ** 2
                 + beta * (k.dx(0) + k.dx(1) * v) ** 2)
    print('Res={0}, Func={1}'.format(assemble(res * dx),
                                     assemble(fun * dx)))

    # Convert back to array.
    vel = dh.funvec2img(v.vector().get_local(), m, n)
    k = dh.funvec2img(k.vector().get_local(), m, n)
    return vel, k, assemble(res * dx), assemble(fun * dx), res > tol
