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
function [A, b] = cm(f, alpha, beta, h, ht)
%CM Creates a linear system for the 1D mass preservation flow problem with 
%spatio-temporal regularisation.
%
%   [A, b] = CM(f, alpha, beta, h, ht) takes matrix f of image 
%   intensities, spatial and temporal regularisation parameters alpha and
%   beta, and spatial and temporal scaling parameters h and ht, and creates
%   a linear system of the form A*x = b.
%
%   f is a matrix of size [m, n] where m is the number of time steps and n
%   the number of pixels.
%   A is a matricx of size [m*n, m*n].
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
A = (f.^2 + alpha) .* laplacian1d(n, t, h) + 2*fx.*f .* deriv1d(n, t, h) + spdiags(fxx.*f, 0, t*n, t*n) + beta * templaplacian1d(n, t, ht);

% Create right-hand side.
b = -fxt.*f;

% Incorporate boundary conditions for left boundary.
I = 1:n:n*t;
d = spdiags(A, 0);
d(I) = d(I) + 2*f(I).*fx(I)/h - 2*(f(I).*fx(I)).^2 ./ (f(I).^2 + alpha);
A = spdiags(d, 0, A);
b(I) = b(I) - 2*f(I).*ft(I)/h + (2*fx(I).*ft(I).*f(I).^2) ./ (f(I).^2 + alpha);

% Incorporate boundary conditions for right boundary.
I = n:n:n*t;
d = spdiags(A, 0);
d(I) = d(I) - 2*f(I).*fx(I)/h - 2*(f(I).*fx(I)).^2 ./ (f(I).^2 + alpha);
A = spdiags(d, 0, A);
b(I) = b(I) + 2*f(I).*ft(I)/h + (2*fx(I).*ft(I).*f(I).^2) ./ (f(I).^2 + alpha);

end