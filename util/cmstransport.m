% Copyright 2017 Lukas Lang
%
% This file is part of OFMC.
%
%    OFMC is free software: you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.
%
%    OFMC is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with OFMC.  If not, see <http://www.gnu.org/licenses/>.
function [A, b] = cmstransport(f, v, k, h, ht)
%CMSTRANSPORT Creates a linear system for the 1D mass conservation transport
%with source/sink term using a second-order Crank-Nicolson scheme with
%homogenous Neumann boundary conditions.
%
%   [A, b] = CMSTRANSPORT(f, v, k, h, ht) takes a vector f of image 
%   intensities, a vector v of velocities, a source vector k, spatial and 
%   temporal scaling parameters h and ht, and creates a linear system of
%   the form Ax = b.
%
%   f, v, and k are vectors of length n.
%   h, ht are positive scalars.
%   A is a matrix of size [n, n].
%   b is a vector of length n.

% Get image size.
n = length(f);

% Compute gradient of f.
fx = gradient(f, h);

% Compute gradient of v.
vx = gradient(v, h);

% Create system matrix.
A = deriv1d(n, 1, h).*v/2;
d = spdiags(A, 0);
d = d + ones(n, 1)/ht + vx/2;
A = spdiags(d, 0, A);

% Create right-hand side.
b = k + f/ht - v.*fx/2 - vx.*f/2;

end