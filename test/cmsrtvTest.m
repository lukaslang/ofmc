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
function tests = cmsrtvTest
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
alpha0 = 1;
alpha1 = 1;
beta0 = 1;
beta1 = 1;

% Set scaling parameters.
h = 1/(n-1);
ht = 1/(t-1);

% Compute linear system.
[A, b] = cmsrtv(f, zeros(t, n), alpha0, alpha1, beta0, beta1, h, ht, 1e-3);
verifyEqual(testCase, size(A), [2*t*n, 2*t*n]);
verifyEqual(testCase, size(b), [2*t*n, 1]);

x = A \ b;
verifyEqual(testCase, x, zeros(2*t*n, 1));

% Create constant image.
n = 4;
t = 3;

% Set scaling parameters.
h = 1/(n-1);
ht = 1/(t-1);

% Create constant Gaussian pattern.
sigma = 0.05;
x = repmat(0:h:1, t, 1);
f = normpdf(x, 0.5, sigma);

% Set regularisation parameters.
alpha0 = 1;
alpha1 = 1;
beta0 = 1;
beta1 = 1;

% Compute linear system.
[A, b] = cmsrtv(f, zeros(t, n), alpha0, alpha1, beta0, beta1, h, ht, 1e-3);
verifyEqual(testCase, size(A), [2*t*n, 2*t*n]);
verifyEqual(testCase, size(b), [2*t*n, 1]);

x = A \ b;
verifyEqual(testCase, x, zeros(2*t*n, 1));

end