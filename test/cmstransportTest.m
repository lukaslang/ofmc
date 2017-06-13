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
function tests = cmstransportTest
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    cd('../');
end

function teardownOnce(testCase)
    cd('test');
end

function resultTest(testCase)

n = 3;
t = 5;
h = 1;
ht = 1;

% Create empty image with three time steps and n pixels.
f = rand(n, 1);

% Create velocity field.
v = zeros(t, n);

% Create source term.
k = zeros(t, n);

fw = zeros(t, n);
fw(1, :) = f';
for j=2:t
    % Create linear system for transport.
    [A, b] = cmstransport(fw(j-1, :)', v(j-1, :)', k(j-1, :)', h, ht);

    % Solve system.
    x = A \ b;
    
    fw(j, :) = x';
end
verifyEqual(testCase, fw, repmat(f', t, 1), 'AbsTol', 1e-15);

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

% Create travelling pattern.
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

% Create source.
k = zeros(t, n);

% Compute transport.
fw = zeros(t, n);
fw(1, :) = f(1, :);
for j=2:t
    % Create linear system for transport.
    [A, b] = cmstransport(fw(j-1, :)', v(j-1, :)', k(j-1, :)', h, ht);

    % Solve system.
    x = A \ b;
    
    % Store result.
    fw(j, :) = x';
end

verifyEqual(testCase, fw, f, 'AbsTol', 1e-1);
fprintf('Max. pointwise warping error is %.2f\n', max(abs(fw(:) - f(:))));

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
imagesc(0:h:1, 0:ht:1, fw);
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
title('Warped image.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);

figure(3);
imagesc(0:h:1, 0:ht:1, f - fw);
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
title('Difference between original and warped image.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);

end