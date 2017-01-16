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
function L = deriv1d(n, t, h)
%DERIV1D Creates first difference matrix for 1D.
%
%   L = DERIV1D(n, t, h) takes number of pixels n, number of
%   time instants t, and a scaling parameter h, and creates first order
%   central difference matrix.
%
%   n, t are integers.
%   h is a scalar.
%   L is a matrix of size [n*t, n*t].
%
%   Note that n > 1 and t >= 1.

v1 = -ones(n, 1)/(2*h);
v1(n-1) = 0;
v2 = ones(n, 1)/(2*h);
v2(2) = 0;
L = spdiags([v1, v2], [-1, 1], n, n);
L = kron(eye(t), L);

end