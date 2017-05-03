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
function fw = computecmstransport(f, v, k, h, ht, iterSolver, tolSolver)
%COMPUTECMSTRANSPORT Computes transport of an image along a vector field.
%
%   fw = COMPUTECMSTRANSPORT(f, v, k, h, ht) takes a vector f of image 
%   intensities, a vector v of velocities, a source k, and spatial and 
%   temporal scaling parameters h and ht, the maximum number of solver 
%   iterations, the solver tolerance, and transports the first time instant
%   of f along v.
%
%   Note that this function uses GMRES for solving the linear system.
%
%   f, v, and k are matrices of size [t, n].
%   h, ht are positive scalars.
%   iterSolver > 0 is an integer.
%   tolSolver > 0 is a scalar.
%   fw is a matrix of size [t, n].

% Get image size.
[t, n] = size(f);

% Initialise first time instant.
fw = zeros(t, n);
fw(1, :) = f(1, :);

% Compute transport.
for j=2:t
    % Create linear system for transport.
    [At, bt] = cmstransport(fw(j-1, :)', v(j-1, :)', k(j-1, :)', h, ht);

    % Solve system.
    [xt, ~, relres, iter] = gmres(At, bt, [], tolSolver, min(iterSolver, size(At, 1)));
    fprintf('GMRES iter %i, relres %e\n', iter(1)*iter(2), relres);

    % Save result.
    fw(j, :) = xt';
end
end