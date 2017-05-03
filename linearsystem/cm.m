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
function [A, B, C, b] = cm(f, h, ht)
%CM Creates a linear system for the 1D mass preservation flow problem with 
%spatio-temporal regularisation.
%
%   [A, B, C, b] = CM(f, h, ht) takes matrix f of image intensities, and 
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

% Compute partial derivatives.
[fx, ft] = gradient(f, h, ht);
[fxx, ~] = gradient(fx, h, ht);
[fxt, ~] = gradient(ft, h, ht);

% Transpose data.
f = img2vec(f);
fx = img2vec(fx);
fxx = img2vec(fxx);
fxt = img2vec(fxt);

% Create matrix A.
A = bsxfun(@times, f.^2, laplacian1d(n, t, h)) + spdiags(fxx.*f, 0, t*n, t*n) + bsxfun(@times, 2*fx.*f, deriv1d(n, t, h));

% Create spatial regularisation matrix for v.
B = laplacian1d(n, t, h);

% Create temporal regularisation matrix for v.
C = templaplacian1d(n, t, ht);

% Create right-hand side.
b = -fxt.*f;

end