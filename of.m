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
function [A, B, C, b] = of(f, h, ht)
%OF Creates a linear system for the 1D optical flow problem with 
%spatio-temporal regularisation.
%
%   [A, B, C, b] = OF(f, h, ht) takes matrix f of image intensities, and 
%   spatial and temporal scaling parameters h and ht, and creates a linear
%   system of the form
%
%   A + alpha*B + beta*C = b.
%
%   f is a matrix of size [m, n] where m is the number of time steps and n
%   the number of pixels.
%   A, B, C are matrices of size [m*n, m*n].
%   b is a vector of length m*n.

% Get image size.
[t, n] = size(f);

% Compute partial derivatives with central differences.
[fx, ft] = gradient(f, h, ht);

% Transpose data.
fx = fx';
fx = fx(:);
ft = ft';
ft = ft(:);

% Create matrix A.
A = spdiags(-fx.^2, 0, t*n, t*n);

% Create spatial regularisation matrix for v.
B = laplacian1d(n, t, h);

% Create temporal regularisation matrix for v.
C = templaplacian1d(n, t, ht);

% Create right-hand side.
b = ft.*fx;

end