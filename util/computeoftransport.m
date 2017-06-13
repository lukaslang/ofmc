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
function fw = computeoftransport(f, v, h, ht)
%COMPUTEOFTRANSPORT Computes transport of an image along a vector field.
%
%   fw = COMPUTEOFTRANSPORT(f, v, h, ht) takes a vector f of image 
%   intensities, a vector v of velocities, and spatial and temporal scaling
%   parameters h and ht, and transports the first time instant of f along v.
%
%   f and v are matrices of size [t, n].
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
    [At, bt] = oftransport(fw(j-1, :)', v(j-1, :)', h, ht);

    % Solve system.
    xt = At \ bt;
    
    % Save result.
    fw(j, :) = xt';
end
end