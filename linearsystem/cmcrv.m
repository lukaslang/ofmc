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
function [A, B, C, D, b, c] = cmcrv(f, k, h, ht)
%CMCRV Creates a linear system for the 1D mass preservation flow problem 
%with spatio-temporal and convective regularisation.
%
%   [A, B, C, D, b, c] = CMCRV(f, k, h, ht) takes matrix f of image 
%   intensities, a source term, and spatial and temporal scaling parameters
%   h and ht, and creates a linear system of the form
%
%   A + alpha*B + beta*C + gamma*D = b + gamma*c.
%
%   f and k are matrices of size [m, n] where m is the number of time steps
%   and n the number of pixels.
%   A, B, C, D are matrices of size [m*n, m*n].
%   b and c are vectors of length m*n.

% Get image size.
[t, n] = size(f);

% Compute partial derivatives.
[fx, ft] = gradient(f, h, ht);
[fxx, ~] = gradient(fx, h, ht);
[fxt, ~] = gradient(ft, h, ht);
[kx, kt] = gradient(k, h, ht);

% Transpose data.
f = img2vec(f);
fx = img2vec(fx);
fxx = img2vec(fxx);
fxt = img2vec(fxt);
kx = img2vec(kx);
kt = img2vec(kt);

% Create matrix A.
A = bsxfun(@times, f.^2, laplacian1d(n, t, h)) + spdiags(fxx.*f, 0, t*n, t*n) + bsxfun(@times, 2*fx.*f, deriv1d(n, t, h));

% Create spatial regularisation matrix for v.
B = laplacian1d(n, t, h);

% Create temporal regularisation matrix for v.
C = templaplacian1d(n, t, ht);

% Create convective regularisation term.
D = spdiags(kx.^2, 0, t*n, t*n);

% Create right-hand side.
b = -fxt.*f;

% Create second term on right-hand side.
c = -kt.*kx;

end