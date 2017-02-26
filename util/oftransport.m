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
function [A, b] = oftransport(f, v, h, ht)
%OFTRANSPORT Creates a linear system for the 1D optical flow transport
%using a second-order Crank-Nicolson scheme.
%
%   [A, b] = OFTRANSPORT(f, v, k, h, ht) takes a vector f of image 
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

% Create system matrix.
A = deriv1d(n, 1, h).*v/2;
A = spdiags(ones(n, 1)/ht, 0, A);

% Create right-hand side.
b = f/ht - v.*fx/2;

end