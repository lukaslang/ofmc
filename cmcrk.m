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
function [A, B, b] = cmcrk(f, v, h, ht)
%CMCRK Creates a linear system for the 1D mass preservation flow problem with
%source terms with spatio-temporal regularisation.
%
%   [A, B, b] = CMCRK(f, v, h, ht) takes matrix f of image intensities, 
%   and spatial and temporal scaling parameters h and ht, and creates a 
%   linear system of the form
%
%   A + gamma*B = b.
%
%   f and v are matrices of size [m, n] where m is the number of time steps
%   and n the number of pixels.
%   A, B are matrices of size [m*n, m*n].
%   b is a vector of length m*n.

% Get image size.
[t, n] = size(f);

% Compute partial derivatives.
[fx, ft] = gradient(f, h, ht);
[vx, vt] = gradient(v, h, ht);

% Transpose data.
f = img2vec(f);
fx = img2vec(fx);
ft = img2vec(ft);
v = img2vec(v);
vx = img2vec(vx);
vt = img2vec(vt);

% Create matrix A.
A = -speye(t*n, t*n);

% Create convective regularisation matrix for k.
B = bsxfun(@times, v.^2, laplacian1d(n, t, h)) + bsxfun(@times, 2*vx.*v + vt, deriv1d(n, t, h)) + bsxfun(@times, vx, tempderiv1d(n, t, ht)) + bsxfun(@times, 2*v, secondderiv1d(n, t, h, ht)) + templaplacian1d(n, t, ht);

% Create right-hand side.
b = -ft - fx.*v - f.*vx;

end