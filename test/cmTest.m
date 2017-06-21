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

% Set regularisation parameters.
alpha = 1;
beta = 1;

% Compute linear system.
[A, b] = cm(f, alpha, beta, 1, 1);
verifyEqual(testCase, size(A), [t*n, t*n]);
verifyEqual(testCase, size(b), [t*n, 1]);

x = A \ b;
verifyEqual(testCase, x, zeros(t*n, 1));

end

function travellingGaussianTest(testCase)

% Set regularisation parameters.
alpha = 1;
beta = 1;

% Set time and space resolution.
n = 100;
t = 20;

% Set scaling parameters.
h = 1/(n-1);
ht = 1/(t-1);

% Create travelling Gaussian pattern.
sigma = 0.05;
x = repmat(0:h:1, t, 1);
y = repmat((0:ht:1)', 1, n);
f = normpdf(x, 0.5+y/10, sigma);

% Compute linear system.
[A, b] = cm(f, alpha, beta, h, ht);

% Solve system.
x = A \ b;

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

function membraneTest(testCase)

% Set regularisation parameters.
alpha = 0.005;
beta = 0.005;

% Set time and space resolution.
n = 100;
t = 30;

% Set scaling parameters.
h = 1/(n-1);
ht = 1/(t-1);

% Set initial distribution.
finit = repmat(sin(5*(0:2*pi*h:2*pi)), t, 1);
finit(:, [1:10, end-10:end]) = 0;

% Create multiplicator.
mult = 0.5*sin(2*(0:2*pi/(t-1):2*pi))';

% Create vector field.
k = 0.5;
v = 1 ./ (1 + exp(-2*k*(-n/2:1:n/2-1))) - 0.5;
vgt = mult .* repmat(v, t, 1);

% Create image.
f = computecmstransport(finit, vgt, zeros(t, n), h, ht);

% Cut boundaries.
%f = f(:, 11:end-10);
%vgt = vgt(:, 11:end-10);
[t, n] = size(f);

% Set scaling parameters.
h = 1/(n-1);
ht = 1/(t-1);

% Compute linear system.
[A, b] = cm(f, alpha, beta, h, ht);

% Solve system.
x = A \ b;

% Recover flow.
v = reshape(x, n, t)';

figure(1);
imagesc(0:h:1, 0:ht:1, f);
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
[X, Y] = meshgrid(0:h:1, 0:ht:1);
%streamline(X, Y, ht*v, ht*ones(t, n), 0:2*h:1, ht*zeros(1, ceil(n/2)));
streamline(X, Y, ht*v, ht*ones(t, n), 0:h:1, ht*zeros(1, n));
title('Input image with streamlines superimposed.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);

figure(2);
imagesc(0:h:1, 0:ht:1, v);
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
title('Velocity field for mass conservation.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);

figure(3);
imagesc(0:h:1, 0:ht:1, vgt);
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
title('Ground truth velocity field.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);

figure(4);
imagesc(0:h:1, 0:ht:1, cmresidual(f, v, h, ht));
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
title('Residual.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);

np = 10;
fp = padarray(f, [0, np], 'replicate');
vp = padarray(v, [0, np], 'replicate');
fw = computecmstransport(fp, vp, zeros(t, n+2*np), h, ht);
fw = fw(:, np+1:end-np);

figure(5);
imagesc(0:h:1, 0:ht:1, fw);
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
title('Transport.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);

diff = abs(f - fw);
figure(6);
imagesc(0:h:1, 0:ht:1, diff);
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
title('Absolute difference between image and transported image.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);

end