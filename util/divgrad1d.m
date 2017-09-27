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
function L = divgrad1d(d, h)
%DIVGRAD1D Creates finite difference matrix of div(grad v / d) for 1D.
%
%   L = DIVGRAD1D(d, h) takes a matrix d of size [t, n], a number of 
%   pixels n and a scaling parameter h, and creates second-order finite 
%   difference matrix.
%
%   h is a scalar.
%   L is a matrix of size [n*t, n*t].
%
%   Note that n > 1 and t >= 1.
%   Note that d is interpolated between grid points.
%
%   See Aubert & Kornprobst: Mathematical Problems in Image Processing,
%   2006, eqn. (A.73) for details on the discretisation.

% Set interpolation/extrapolation method.
method = 'linear';

[t, n] = size(d);

% Create grid.
X = ndgrid(0:h:1);

L = cell(t, 1);
for k=1:t
    % Interpolate between grid points.
    F = griddedInterpolant(X, 1./d(k, :), method);
    v1 = [F(h/2:h:1-h-h/2), F(1-h/2) + F(1+h/2), 0]';
    v2 = (F(-h/2:h:1-h/2) + F(h/2:h:1+h/2))';
    v3 = [0, F(-h/2) + F(h/2), F(h+h/2:h:1-h/2)]';

    L{k} = spdiags([v1/(h^2), -v2/(h^2), v3/(h^2)], [-1, 0, 1], n, n);
end

% Create block diagonal matrix.
L = blkdiag(L{:});

end