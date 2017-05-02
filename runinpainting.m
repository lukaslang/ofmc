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
niter = 30;

% Number of inner iterations.
niterinner = 10;

% Set norm regularisation parameter.
epsilon = 1e-3;

% Read data.
g = imread(fullfile(path, sprintf('%s.png', name)));

% Remove cut.
%f = padarray(double(g([1:5, 7:end], :)), [0, 10]);
%f = padarray(double(g), [0, 10]);
f = double(g);
f(6, :) = f(5, :);

% Remove cut and everything before.
%f = padarray(double(g(7:end, :)), [0, 10]);
[t, n] = size(f);

% Create mask.
M = ones(t, n);
M(6, :) = 0;
m = img2vec(M);

% Set scaling parameters.
h = 1/(n-1);
ht = 1/(t-1);

% Scale image to [0, 1].
fdelta = (f - min(f(:))) / max(f(:) - min(f(:)));

% Filter image.
%f = imfilter(fdelta, fspecial('gaussian', 5, 5), 'replicate');

%% Joint image, velocity, and source estimation with convective regularisation and regularised TV.

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
mkdir(fullfile(outputPath, 'cmjertv'));

% Create linear system.
[A, B, C, D, E, F, b] = cms(fdelta, h, ht);

% Solve system for mass conservation with source/sink term.
[x, ~, relres, iter] = gmres(A + alpha*B + beta*C + gamma*D + delta*E + eta*F, b, [], tolSolver, iterSolver);
fprintf('GMRES iter %i, relres %e\n', iter(1)*iter(2), relres);

% Recover flow.
v = reshape(x(1:t*n), n, t)';
k = reshape(x(t*n+1:end), n, t)';

figure(1);
imagesc(0:h:1, 0:ht:1, fdelta);
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
colormap gray;
[X, Y] = meshgrid(0:h:1, 0:ht:1);
streamline(X, Y, ht*v, ht*ones(t, n), 0:2*h:1, ht*zeros(ceil(n/2), 1));
title('Input image with streamlines superimposed.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'cmjertv', sprintf('%s-input-000.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(2);
imagesc(0:h:1, 0:ht:1, v);
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
colormap default;
title('Velocity field for MC with source/sink term and convective regularisation.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'cmjertv', sprintf('%s-velocity-000.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(3);
imagesc(0:h:1, 0:ht:1, k);
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
colormap default;
title('Source/sink term with convective regularisation.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'cmjertv', sprintf('%s-source-000.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(4);
imagesc(0:h:1, 0:ht:1, cmsresidual(fdelta, v, k, h, ht));
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
colormap default;
title('Residual.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'cmjertv', sprintf('%s-residual-000.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(5);
imagesc(0:h:1, 0:ht:1, warpcms(fdelta, v, k, h, ht));
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
colormap gray;
title('Warped image.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'cmjertv', sprintf('%s-warp-000.png', name)), '-png', '-q300', '-a1', '-transparent');

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

figure(6);
imagesc(0:h:1, 0:ht:1, fw);
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
colormap gray;
title('Transport.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'cmjertv', sprintf('%s-transport-000.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(7);
imagesc(0:h:1, 0:ht:1, abs(fdelta - fw));
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
colormap default;
title('Absolute difference between image and transported image.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'cmjertv', sprintf('%s-diff-000.png', name)), '-png', '-q300', '-a1', '-transparent');

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
    
    figure(1);
    imagesc(0:h:1, 0:ht:1, fdelta);
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    colormap gray;
    [X, Y] = meshgrid(0:h:1, 0:ht:1);
    streamline(X, Y, ht*v, ht*ones(t, n), 0:2*h:1, ht*zeros(ceil(n/2), 1));
    title('Input image with streamlines superimposed.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'cmjertv', sprintf('%s-input-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    
    figure(2);
    imagesc(0:h:1, 0:ht:1, v);
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    colormap default;
    title('Velocity field for MC with source/sink term and convective regularisation.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'cmjertv', sprintf('%s-velocity-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    
    figure(3);
    imagesc(0:h:1, 0:ht:1, k);
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    colormap default;
    title('Source/sink term with convective regularisation.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'cmjertv', sprintf('%s-source-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    
    figure(4);
    imagesc(0:h:1, 0:ht:1, cmsresidual(f, v, k, h, ht));
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    colormap default;
    title('Residual.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'cmjertv', sprintf('%s-residual-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    
    figure(5);
    imagesc(0:h:1, 0:ht:1, warpcms(f, v, k, h, ht));
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    colormap gray;
    title('Warped image.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'cmjertv', sprintf('%s-warp-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    
    % Compute transport.
    fw = zeros(t, n);
    fw(1, :) = f(1, :);
    for l=2:t
        % Create linear system for transport.
        [At, bt] = cmstransport(fw(l-1, :)', v(l-1, :)', k(l-1, :)', h, ht);

        % Solve system.
        [xt, ~, relres, iter] = gmres(At, bt, [], tolSolverTransport, min(iterSolverTransport, size(At, 1)));
        fprintf('GMRES iter %i, relres %e\n', iter(1)*iter(2), relres);

        % Save result.
        fw(l, :) = xt';
    end

    figure(6);
    imagesc(0:h:1, 0:ht:1, fw);
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    colormap gray;
    title('Transport.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'cmjertv', sprintf('%s-transport-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');

    figure(7);
    imagesc(0:h:1, 0:ht:1, abs(f - fw));
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    colormap default;
    title('Absolute difference between image and transported image.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'cmjertv', sprintf('%s-diff-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    
    figure(8);
    imagesc(f);
    axis image;
    colorbar;
    colormap gray;
    title('Denoised image.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'cmjertv', sprintf('%s-image-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    drawnow();
end