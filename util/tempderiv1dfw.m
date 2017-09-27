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
function L = tempderiv1dfw(n, t, ht)
%TEMPDERIV1DFW Creates first difference matrix for 1D in time.
%
%   L = TEMPDERIV1D(n, t, h) takes number of pixels n, number of
%   time instants t, and a scaling parameter ht, and creates first order
%   forward difference matrix for first derivative in time with zero
%   Neumann boundary conditions.
%
%   n, t are integers.
%   ht is a scalar.
%   L is a matrix of size [n*t, n*t].
%
%   Note that n > 1 and t >= 3.

v1 = [-ones(n*(t-1), 1)/(2*ht); zeros(n, 1)];
v2 = [zeros(n, 1); ones(n*(t-1), 1)/(2*ht)];
L = spdiags([v1, v2], [0, n], n*t, n*t);

end