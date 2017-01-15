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
function L = templaplacian1d(n, t, ht)
%TEMPLAPLACIAN1D Creates temporal second difference matrix for 1D.
%
%   L = TEMPLAPLACIAN1D(n, t, ht) takes number of pixels n, number of
%   time instants t, and a scaling parameter ht, and creates second order
%   central difference matrix.
%
%   n, t are integers.
%   ht is a scalar.
%   L is a matrix of size [n*t, n*t].
%
%   Note that n >= 1 and t >= 3.

v1 = ones(n*t, 1)/(ht^2);
v1((t-1)*n) = 2/(ht^2);
v2 = -2*ones(t*n, 1)/(ht^2);
v3 = ones(t*n, 1)/(ht^2);
v3(n+1) = 2/(ht^2);
L = spdiags([v1, v2, v3], [-n, 0, n], t*n, t*n);

end