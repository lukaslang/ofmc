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
function [A, B, C, D, b] = cms(f, h, ht)
%CM Creates a linear system for the 1D mass preservation flow problem with
%source terms with spatio-temporal regularisation.
%
%   [A, B, C, D, b] = CMS(f, h, ht) takes matrix f of image intensities, 
%   and spatial and temporal scaling parameters h and ht, and creates a 
%   linear system of the form
%
%   A + alpha*B + beta*C + gamma*D = b.
%
%   f is a matrix of size [m, n] where m is the number of time steps and n
%   the number of pixels.
%   A, B, C, D are matrices of size [m*n, m*n].
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
ft = img2vec(ft);
fxx = img2vec(fxx);
fxt = img2vec(fxt);

% Create matrix A.
A1 = bsxfun(@times, f.^2, laplacian1d(n, t, h)) + spdiags(fxx.*f, 0, t*n, t*n) + bsxfun(@times, 2*fx.*f, deriv1d(n, t, h));
A2 = bsxfun(@times, -f, deriv1d(n, t, h));
A3 = bsxfun(@times, f, deriv1d(n, t, h)) + spdiags(fx, 0, t*n, t*n);
A4 = spdiags(-ones(t*n, 1), 0, t*n, t*n);
A = [A1, A2; A3, A4];

% Create spatial regularisation matrix for v.
B = [laplacian1d(n, t, h), sparse(t*n, t*n); sparse(t*n, 2*t*n)];

% Create temporal regularisation matrix for v.
C = [templaplacian1d(n, t, ht), sparse(t*n, t*n); sparse(t*n, 2*t*n)];

% Create regularisation matrix for k.
D = -spdiags([zeros(t*n, 1); ones(t*n, 1)], 0, 2*t*n, 2*t*n);

% Create right-hand side.
b = [-fxt.*f; -ft];

end