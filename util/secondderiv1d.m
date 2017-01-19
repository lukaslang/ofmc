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
function L = secondderiv1d(n, t, h, ht)
%SECONDDERIV1D Creates mixed second derivative difference matrix for 1D.
%
%   L = SECONDDERIV1D(n, t, h, ht) takes number of pixels n, number of
%   time instants t, and scaling parameters h and ht, and creates second
%   order central difference matrix.
%
%   n, t are integers.
%   h, ht are scalars.
%   L is a matrix of size [n*t, n*t].
%
%   Note that n >= 3 and t >= 3.

D = deriv1d(n, t-2, 2*h*ht);
L1 = [sparse(n, t*n); -D, sparse(n*(t-2), 2*n); sparse(n, t*n)];
L2 = [sparse(n, t*n); sparse(n*(t-2), 2*n), D; sparse(n, t*n)];
L = L1 + L2;

end