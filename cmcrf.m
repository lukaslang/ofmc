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
function [A, B, C, D, b, c] = cmcrf(f, v, k, h, ht)
%CMCRF Creates a linear system for the 1D mass preservation flow problem 
%with spatio-temporal and convective regularisation.
%
%   [A, B, C, D, b, c] = CMCRV(f, k, h, ht) takes matrix f of image 
%   intensities, a source term, and spatial and temporal scaling parameters
%   h and ht, and creates a linear system of the form
%
%   A + kappa*B + lambda*C + mu*D = kappa*b + c.
%
%   f, v, and k are matrices of size [m, n] where m is the number of time 
%   steps and n the number of pixels.
%   A, B, C, D are matrices of size [m*n, m*n].
%   b and c are vectors of length m*n.

% Get image size.
[t, n] = size(f);

% Compute partial derivatives.
[vx, vt] = gradient(v, h, ht);
[vxx, vxt] = gradient(vx, h, ht);
[kx, kt] = gradient(k, h, ht);

% Transpose data.
f = img2vec(f);
v = img2vec(v);
kx = img2vec(kx);
kt = img2vec(kt);
vx = img2vec(vx);
vt = img2vec(vt);
vxx = img2vec(vxx);
vxt = img2vec(vxt);

% Create matrix associated with mass conservation.
A = bsxfun(@times, v.^2, laplacian1d(n, t, h)) + templaplacian1d(n, t, ht) + bsxfun(@times, 2*v, secondderiv1d(n, t, h, ht)) + bsxfun(@times, vt + 2*vx.*v, deriv1d(n, t, h)) + bsxfun(@times, vx, tempderiv1d(n, t, ht)) + spdiags(vxt + vxx.*v, 0, t*n, t*n);

% Create matrix accosicated with data term.
B = spdiags(-ones(t*n, 1), 0, t*n, t*n);

% Create spatial regularisation matrix for f.
C = laplacian1d(n, t, h);

% Create temporal regularisation matrix for f.
D = templaplacian1d(n, t, ht);

% Create right-hand side.
b = -f;

% Create second term on right-hand side.
c = kx.*v + kt;

end