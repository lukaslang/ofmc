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
function tests = cmTest
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    cd('../');
end

function teardownOnce(testCase)
    cd('test');
end

function resultTest(testCase)

n = 4;
t = 4;

% Create empty image with three time steps and n pixels.
f = zeros(t, n);

% Compute linear system.
[A, B, C, b] = cm(f, 1, 1);
verifyEqual(testCase, size(A), [t*n, t*n]);
verifyEqual(testCase, size(B), [t*n, t*n]);
verifyEqual(testCase, size(C), [t*n, t*n]);
verifyEqual(testCase, size(b), [t*n, 1]);

[x, flag] = gmres(A + B + C, b, [], 1e-3, size(A, 1));
verifyEqual(testCase, flag, 0);
verifyEqual(testCase, x, zeros(t*n, 1));

end

function travellingGaussianTest(testCase)

% Set regularisation parameters.
alpha = 0;
beta = 0;

% Set time and space resolution.
n = 100;
t = 20;

% Set scaling parameters.
h = 1/(n-1);
ht = 1/(t-1);

% Create travelling heaviside pattern.
sigma = 0.05;
x = repmat(0:h:1, t, 1);
y = repmat((0:ht:1)', 1, n);
f = normpdf(x, 0.5+y/10, sigma);

% Compute linear system.
[A, B, C, b] = cm(f, h, ht);

% Solve system.
[x, ~, relres, iter] = gmres(A + alpha*B + beta*C, b, [], 1e-3, 1000);
fprintf('GMRES iter %i, relres %e\n', iter(1)*iter(2), relres);

% Recover flow.
v = reshape(x, n, t)';

figure(1);
imagesc(0:h:1, 0:ht:1, f);
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
[X, Y] = meshgrid(0:h:1, 0:ht:1);
streamline(X, Y, ht*v, ht*ones(t, n), 0:2*h:1, ht*zeros(1, ceil(n/2)));
title('Input image with streamlines superimposed.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);

figure(2);
imagesc(0:h:1, 0:ht:1, v);
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
title('Velocity field for mass conservation', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);

figure(3);
imagesc(0:h:1, 0:ht:1, ofresidual(f, v, h, ht));
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
title('Residual.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);

end