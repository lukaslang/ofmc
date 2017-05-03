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

% Clean up.
clear;
close all;
clc;

% Define data.
name = 'E2PSB1PMT-560-C1';
path = 'data';

% Create start date and time.
startdate = datestr(now, 'yyyy-mm-dd-HH-MM-SS');

% Define and create output folder.
outputPath = fullfile('results', startdate);
mkdir(outputPath);

% Set parameters for linear system solver.
iterSolver = 1000;
tolSolver = 1e-3;

% Set parameters for solving transport problem.
iterSolverTransport = 1000;
tolSolverTransport = 1e-15;

% Read data.
g = im2double(imread(fullfile(path, sprintf('%s.png', name))));

% Remove cut.
fdelta = g([1:5, 7:end], :);

% Get image size.
[t, n] = size(fdelta);

% Set scaling parameters.
h = 1/(n-1);
ht = 1/(t-1);

% Filter image.
f = imfilter(fdelta, fspecial('gaussian', 5, 5), 'replicate');

%% Mass conservation.

% Spatial and temporal reguarisation of v.
alpha = 0.025;
beta = 0.01;

% Create output folder. 
alg = 'cm';
mkdir(fullfile(outputPath, alg));

% Create linear system for mass conservation.
[A, B, C, b] = cm(f, h, ht);

% Solve system.
[x, ~, relres, iter] = gmres(A + alpha*B + beta*C, b, [], tolSolver, iterSolver);
fprintf('GMRES iter %i, relres %e\n', iter(1)*iter(2), relres);

% Recover flow.
v = reshape(x, n, t)';

% Visualise flow.
plotstreamlines(1, 'Input image with streamlines superimposed.', 'gray', f, v, h, ht);
export_fig(gcf, fullfile(outputPath, alg, sprintf('%s-input.png', name)), '-png', '-q300', '-a1', '-transparent');

plotdata(2, 'Velocity.', 'default', v, h, ht);
export_fig(gcf, fullfile(outputPath, alg, sprintf('%s-velocity.png', name)), '-png', '-q300', '-a1', '-transparent');

res = cmresidual(f, v, h, ht);
plotdata(3, 'Residual.', 'default', res, h, ht);
export_fig(gcf, fullfile(outputPath, alg, sprintf('%s-residual.png', name)), '-png', '-q300', '-a1', '-transparent');

warp = warpcms(f, v, zeros(t, n), h, ht);
plotdata(4, 'Warped image.', 'gray', warp, h, ht);
export_fig(gcf, fullfile(outputPath, alg, sprintf('%s-warp.png', name)), '-png', '-q300', '-a1', '-transparent');

fw = computecmstransport(f, v, zeros(t, n), h, ht, iterSolverTransport, tolSolverTransport);
plotdata(5, 'Transport.', 'gray', fw, h, ht);
export_fig(gcf, fullfile(outputPath, alg, sprintf('%s-transport.png', name)), '-png', '-q300', '-a1', '-transparent');

diff = abs(f - fw);
plotdata(6, 'Absolute difference between image and transported image.', 'default', diff, h, ht);
export_fig(gcf, fullfile(outputPath, alg, sprintf('%s-diff.png', name)), '-png', '-q300', '-a1', '-transparent');