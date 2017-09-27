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
function [A, b] = cmrtv(f, v, alpha0, alpha1, h, ht, epsilon)
%CM Creates a linear system for the 1D mass preservation flow problem with 
%spatial epsilon-TV regularisation and temporal regularisation.
%
%   [A, b] = CMRTV(f, v, alpha0, alpha1, h, ht) takes matrix f of image 
%   intensities, a velocity field v, spatial and temporal regularisation 
%   parameters alpha0 and alpha1, and spatial and temporal scaling parameters 
%   h and ht, and a positive scalar epsilon for regularised norm 
%   computation, and creates a linear system of the form A*x = b.
%
%   f and v are matrices of size [m, n] where m is the number of time steps
%   and n the number of pixels.
%   A is a matricx of size [m*n, m*n].
%   b is a vector of length m*n.
%   alpha0, alpha1 > 0.
%   epsilon > 0.

% Get image size.
[t, n] = size(f);

% Compute partial derivatives.
[fx, ft] = gradient(f, h, ht);
[fxx, ~] = gradient(fx, h, ht);
[fxt, ~] = gradient(ft, h, ht);
[vx, ~] = gradient(v, h, ht);

% Compute regularised norm of vx.
nvx = rnorm(vx, epsilon);

% Create div(grad v / rnorm(v)) matrix.
B = divgrad1d(nvx, h);

% Create vectors.
f = img2vec(f);
fx = img2vec(fx);
ft = img2vec(ft);
fxx = img2vec(fxx);
fxt = img2vec(fxt);

% Create matrix A.
A = f.^2 .* laplacian1d(n, t, h) + alpha0 * B + 2*fx.*f .* deriv1d(n, t, h) + spdiags(fxx.*f, 0, t*n, t*n) + alpha1 * templaplacian1d(n, t, ht);

% Create right-hand side.
b = -fxt.*f;

% Incorporate boundary conditions for left boundary.
I = 1:n:n*t;
d = spdiags(A, 0);
d(I) = d(I) + 2*f(I).*fx(I)/h - 2*(f(I).*fx(I)).^2 ./ (f(I).^2 + alpha0);
A = spdiags(d, 0, A);
b(I) = b(I) - 2*f(I).*ft(I)/h + (2*fx(I).*ft(I).*f(I).^2) ./ (f(I).^2 + alpha0);

% Incorporate boundary conditions for right boundary.
I = n:n:n*t;
d = spdiags(A, 0);
d(I) = d(I) - 2*f(I).*fx(I)/h - 2*(f(I).*fx(I)).^2 ./ (f(I).^2 + alpha0);
A = spdiags(d, 0, A);
b(I) = b(I) + 2*f(I).*ft(I)/h + (2*fx(I).*ft(I).*f(I).^2) ./ (f(I).^2 + alpha0);

end