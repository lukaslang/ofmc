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
function r = cmsresidual(f, v, k, h, ht)
%CMSRESIDUAL Computes the residual of mass conservation with source term.
%
%   v = CMSRESIDUAL(f, v, k, h, ht) takes an image f, a velocity field v, 
%   a source k, and scaling parameters h and ht, and returns the residual 
%   of the mass conservation equation with source/sink term.
%
%   f, v, k are matrices of size [m, n].
%   h, ht are scalars.
%   r is a matrix of size [m, n].

% Compute partial derivatives.
[fx, ft] = gradient(f, h, ht);
[vx, ~] = gradient(v, h, ht);

% Compute residual.
r = ft + fx.*v + f.*vx - k;

end