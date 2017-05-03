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

% Number of iterations.
niter = 15;

% Number of inner iterations.
niterinner = 15;

% Set norm regularisation parameter.
epsilon = 1e-3;

% Read data.
g = im2double(imread(fullfile(path, sprintf('%s.png', name))));

% Set number of inpainting steps. 
ninterp = 0;

% Interpolate data.
lininterp = (1 - linspace(0, 1, ninterp+2)') * double(g(5, :)) + linspace(0, 1, ninterp+2)' * double(g(7, :));
f = cat(1, double(g(1:4, :)), lininterp, double(g(8:end, :)));
[t, n] = size(f);

% Create inpainting mask.
M = ones(t, n);
M(5+1:5+ninterp, :) = 0;
m = img2vec(M);

% Set scaling parameters.
h = 1/(n-1);
ht = 1/(t-1);

% Filter image.
f = imfilter(fdelta, fspecial('gaussian', 5, 5), 'replicate');

%% Joint image, velocity, and source estimation with convective regularisation, regularised TV, and inpainting.

% Spatial and temporal reguarisation of v.
alpha = 0.02;
beta = 0.005;
% Norm of k.
gamma = 0.001;
% Spatial and temporal regularisation of k.
delta = 0;
eta = 0;
% Convective regularisation of k.
theta = 0.001;
% Data term for f.
kappa = 100;
% Spatial and temporal regularisation for f.
lambda = 0.005;
mu = 0.005;

% Create output folder. 
alg = 'cmjertv';
mkdir(fullfile(outputPath, alg));

% Create linear system.
[A, B, C, D, E, F, b] = cms(f, h, ht);

% Solve system for mass conservation with source/sink term.
[x, ~, relres, iter] = gmres(A + alpha*B + beta*C + gamma*D + delta*E + eta*F, b, [], tolSolver, iterSolver);
fprintf('GMRES iter %i, relres %e\n', iter(1)*iter(2), relres);

% Recover flow.
v = reshape(x(1:t*n), n, t)';
k = reshape(x(t*n+1:end), n, t)';

plotstreamlines(1, 'Input image with streamlines superimposed.', 'gray', f, v, h, ht);
export_fig(gcf, fullfile(outputPath, alg, sprintf('%s-input-000.png', name)), '-png', '-q300', '-a1', '-transparent');

plotdata(2, 'Velocity.', 'default', v, h, ht);
export_fig(gcf, fullfile(outputPath, alg, sprintf('%s-velocity-000.png', name)), '-png', '-q300', '-a1', '-transparent');

plotdata(3, 'Source.', 'default', k, h, ht);
export_fig(gcf, fullfile(outputPath, alg, sprintf('%s-source-000.png', name)), '-png', '-q300', '-a1', '-transparent');

res = cmsresidual(fdelta, v, k, h, ht);
plotdata(4, 'Residual.', 'default', res, h, ht);
export_fig(gcf, fullfile(outputPath, alg, sprintf('%s-residual-000.png', name)), '-png', '-q300', '-a1', '-transparent');

warp = warpcms(fdelta, v, k, h, ht);
plotdata(5, 'Warped image.', 'gray', warp, h, ht);
export_fig(gcf, fullfile(outputPath, alg, sprintf('%s-warp-000.png', name)), '-png', '-q300', '-a1', '-transparent');

% Compute transport.
fw = zeros(t, n);
fw(1, :) = fdelta(1, :);
for l=2:t
    % Create linear system for transport.
    [At, bt] = cmstransport(fw(l-1, :)', v(l-1, :)', k(l-1, :)', h, ht);

    % Solve system.
    [xt, ~, relres, iter] = gmres(At, bt, [], tolSolverTransport, min(iterSolverTransport, size(At, 1)));
    fprintf('GMRES iter %i, relres %e\n', iter(1)*iter(2), relres);

    % Save result.
    fw(l, :) = xt';
end

plotdata(6, 'Transport.', 'gray', fw, h, ht);
export_fig(gcf, fullfile(outputPath, alg, sprintf('%s-transport-000.png', name)), '-png', '-q300', '-a1', '-transparent');

diff = abs(fdelta - fw);
plotdata(7, 'Absolute difference between image and transported image.', 'default', diff, h, ht);
export_fig(gcf, fullfile(outputPath, alg, sprintf('%s-diff-000.png', name)), '-png', '-q300', '-a1', '-transparent');

f = fdelta;
for j=1:niter

    % Create linear system for k.
    [A, B, b] = cmcrk(f, v, h, ht);

    % Solve system.
    [x, ~, relres, iter] = gmres(A + theta*B, b, [], tolSolver, iterSolver);
    fprintf('GMRES iter %i, relres %e\n', iter(1)*iter(2), relres);

    % Recover source.
    k = reshape(x, n, t)';
    
    % Create linear system for f.
    [A, B, C, D, b, c] = cmcrf(fdelta, v, k, h, ht);

    % Solve system.
    [x, ~, relres, iter] = gmres(A + kappa*B.*m + lambda*C + mu*D, kappa*b.*m + c, [], tolSolver, iterSolver);
    fprintf('GMRES iter %i, relres %e\n', iter(1)*iter(2), relres);

    % Recover image.
    f = reshape(x, n, t)';
    
    % Create linear system for v.
    [A, ~, C, D, b, c] = cmcrv(f, k, h, ht);

    for l=1:niterinner
    
        % Create div(grad v / rnorm(v)) matrix.
        [vx, ~] = gradient(v, h, ht);
        B = divgrad1d(rnorm(vx, epsilon), h);
        
        % Solve system.
        [x, ~, relres, iter] = gmres(A + alpha*B + beta*C + theta*D, b + theta*c, [], tolSolver, iterSolver);
        fprintf('GMRES iter %i, relres %e\n', iter(1)*iter(2), relres);

        % Recover flow.
        v = reshape(x, n, t)';
    end
    
    plotstreamlines(1, 'Input image with streamlines superimposed.', 'gray', fdelta, v, h, ht);
    export_fig(gcf, fullfile(outputPath, alg, sprintf('%s-input-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    
    plotdata(2, 'Velocity field for MC with source/sink term and convective regularisation.', 'default', v, h, ht);
    export_fig(gcf, fullfile(outputPath, alg, sprintf('%s-velocity-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');

    plotdata(3, 'Source/sink term with convective regularisation.', 'default', k, h, ht);
    export_fig(gcf, fullfile(outputPath, alg, sprintf('%s-source-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    
    res = cmsresidual(f, v, k, h, ht);
    plotdata(4, 'Residual.', 'default', res, h, ht);
    export_fig(gcf, fullfile(outputPath, alg, sprintf('%s-residual-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    
    warp = warpcms(f, v, k, h, ht);
    plotdata(5, 'Warped image.', 'gray', warp, h, ht);
    export_fig(gcf, fullfile(outputPath, alg, sprintf('%s-warp-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');

    fw = computecmstransport(f, v, k, h, ht, iterSolverTransport, tolSolverTransport);
    plotdata(6, 'Transport.', 'gray', fw, h, ht);
    export_fig(gcf, fullfile(outputPath, alg, sprintf('%s-transport-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');

    diff = abs(fdelta - fw);
    plotdata(7, 'Absolute difference between image and transported image.', 'default', diff, h, ht);
    export_fig(gcf, fullfile(outputPath, alg, sprintf('%s-diff-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    
    plotdata(8, 'Denoised image.', 'gray', f, h, ht);
    export_fig(gcf, fullfile(outputPath, alg, sprintf('%s-image-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    drawnow();
end