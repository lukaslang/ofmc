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
function [A, b] = cmsrtv(f, v, alpha0, alpha1, beta0, beta1, h, ht, epsilon)
%CMSRTV Creates a linear system for the 1D mass preservation flow problem
%with source terms with spatial epsilon-TV regularisation and temporal
%regularisation.
%
%   [A, b] = CMSRTV(f, v, alpha0, alpha1, beta0, beta1, h, ht, epsilon) 
%   takes matrix f of image intensities, a velocity field v, regularisation
%   parameters alpha0, alpha1, beta0, and beta1, spatial and temporal 
%   scaling parameters h and ht, and a positive scalar epsilon for 
%   regularised norm computation, and creates a linear system of the 
%   form A*(v, k)^T= b.
%
%   f and v are matrices of size [m, n] where m is the number of time steps
%   and n the number of pixels.
%   alpha0, alpha1, beta0, beta1 > 0 are scalars.
%   h, ht, and epsilon > 0 are scalars.
%   A is a matrix of size [2*m*n, 2*m*n].
%   b is a vector of length 2*m*n.
%   epsilon > 0 is a scalar.

% Get image size.
[t, n] = size(f);

% Compute partial derivatives.
[fx, ft] = gradient(f, h, ht);
[fxx, ~] = gradient(fx, h, ht);
[fxt, ~] = gradient(ft, h, ht);
[vx, ~] = gradient(v, h, ht);

% Compute regularised norm of vx.
nvx = rnorm(vx, epsilon);

% Compute div-grad matrix.
B = divgrad1d(nvx, h);

% Transpose data.
f = img2vec(f);
fx = img2vec(fx);
ft = img2vec(ft);
fxx = img2vec(fxx);
fxt = img2vec(fxt);

% Create matrix A.
A1 = f.^2 .* laplacian1d(n, t, h) + spdiags(fxx.*f, 0, t*n, t*n) + 2*fx.*f .* deriv1d(n, t, h);
A2 = -f .* deriv1d(n, t, h);
A3 = f .* deriv1d(n, t, h) + spdiags(fx, 0, t*n, t*n);
A4 = spdiags(-ones(t*n, 1), 0, t*n, t*n);

% Create right-hand side.
b1 = -fxt.*f;
b2 = -ft;

% Incorporate boundary conditions for left boundary for first equation.
I = 1:n:n*t;
d = spdiags(A1, 0);
d(I) = d(I) + 2*f(I).*fx(I)/h - 2*(f(I).*fx(I)).^2 ./ (f(I).^2 + alpha0);
A1 = spdiags(d, 0, A1);
d = spdiags(A2, 0);
d(I) = d(I) - 2*f(I)/h + 2*(fx(I).*f(I).^2)./(f(I).^2 + alpha0);
A2 = spdiags(d, 0, A2);
b1(I) = b1(I) - 2*f(I).*ft(I)/h + (2*fx(I).*ft(I).*f(I).^2) ./ (f(I).^2 + alpha0);

% Incorporate boundary conditions for right boundary for first equation.
I = n:n:n*t;
d = spdiags(A1, 0);
d(I) = d(I) - 2*f(I).*fx(I)/h - 2*(f(I).*fx(I)).^2 ./ (f(I).^2 + alpha0);
A1 = spdiags(d, 0, A1);
d = spdiags(A2, 0);
d(I) = d(I) + 2*f(I)/h + 2*(fx(I).*f(I).^2)./(f(I).^2 + alpha0);
A2 = spdiags(d, 0, A2);
b1(I) = b1(I) + 2*f(I).*ft(I)/h + (2*fx(I).*ft(I).*f(I).^2) ./ (f(I).^2 + alpha0);

% Incorporate boundary conditions for left boundary for second equation.
I = 1:n:n*t;
d = spdiags(A3, 0);
d(I) = d(I) - (fx(I).*f(I).^2)./(f(I).^2 + alpha0);
A3 = spdiags(d, 0, A3);
d = spdiags(A4, 0);
d(I) = d(I) + (f(I).^2)./(f(I).^2 + alpha0);
A4 = spdiags(d, 0, A4);
b2(I) = b2(I) + (ft(I).*f(I).^2)./(f(I).^2 + alpha0);

% Incorporate boundary conditions for right boundary for second equation.
I = n:n:n*t;
d = spdiags(A3, 0);
d(I) = d(I) - (fx(I).*f(I).^2)./(f(I).^2 + alpha0);
A3 = spdiags(d, 0, A3);
d = spdiags(A4, 0);
d(I) = d(I) + (f(I).^2)./(f(I).^2 + alpha0);
A4 = spdiags(d, 0, A4);
b2(I) = b2(I) + (ft(I).*f(I).^2)./(f(I).^2 + alpha0);

% Create spatial regularisation matrix for v.
B = [B, sparse(t*n, t*n); sparse(t*n, 2*t*n)];

% Create temporal regularisation matrix for v.
C = [templaplacian1d(n, t, ht), sparse(t*n, t*n); sparse(t*n, 2*t*n)];

% Create spatial regularisation matrix for k.
D = [sparse(t*n, 2*t*n); sparse(t*n, t*n), laplacian1d(n, t, h)];

% Create temporal regularisation matrix for k.
E = [sparse(t*n, 2*t*n); sparse(t*n, t*n), templaplacian1d(n, t, ht)];

% Assemble linear system.
A = [A1, A2; A3, A4] + alpha0*B + alpha1*C + beta0*D + beta1*E;
b = [b1; b2];

end