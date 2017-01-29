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

% Number of iterations for convective regularisation.
niter = 10;

% Read data.
f = imread(fullfile(path, sprintf('%s.png', name)));

% Remove cut and everything before.
f = double(f(7:end, 1:140));
[t, n] = size(f);

% Set scaling parameters.
h = 1/n;
ht = 1/t;

% Scale image to [0, 1].
fdelta = (f - min(f(:))) / max(f(:) - min(f(:)));

% Filter image.
f = imfilter(fdelta, fspecial('gaussian', 5, 5), 'replicate');

%% Optical flow.

% Spatial and temporal reguarisation of v.
alpha = 0.01;
beta = 0.001;

% Create linear system.
[A, B, C, b] = of(f, h, ht);

% Solve system.
[x, flag, relres, iter] = gmres(A + alpha*B + beta*C, b, [], tolSolver, iterSolver);
fprintf('GMRES iter %i, relres %e\n', iter(1)*iter(2), relres);

% Recover flow.
v = reshape(x, n, t)';

% Visualise flow.
figure(1);
imagesc(f);
axis image;
colorbar;
colormap gray;
streamline(v, ones(t, n), 1:2:n, ones(ceil(n/2), 1));
title('Input image with streamlines superimposed.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, sprintf('%s-of-input.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(2);
imagesc(v);
axis image;
colorbar;
title('Velocity field for optical flow.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, sprintf('%s-of-velocity.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(3);
imagesc(ofresidual(f, v, h, ht));
axis image;
colorbar;
title('Residual.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, sprintf('%s-of-residual.png', name)), '-png', '-q300', '-a1', '-transparent');

%% Mass conservation.

% Spatial and temporal reguarisation of v.
alpha = 0.01;
beta = 0.001;

% Create linear system for mass conservation.
[A, B, C, b] = cm(f, h, ht);

% Solve system.
[x, ~, relres, iter] = gmres(A + alpha*B + beta*C, b, [], tolSolver, iterSolver);
fprintf('GMRES iter %i, relres %e\n', iter(1)*iter(2), relres);

% Recover flow.
v = reshape(x, n, t)';

figure(4);
imagesc(f);
axis image;
colorbar;
colormap gray;
streamline(v, ones(t, n), 1:2:n, ones(ceil(n/2), 1));
title('Mass conservation without source/sink term.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, sprintf('%s-cm-input.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(5);
imagesc(v);
axis image;
colorbar;
title('Velocity field for mass conservation without source/sink term.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
set(gca, 'FontName', 'Helvetica');
set(gca, 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, sprintf('%s-cm-velocity.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(6);
imagesc(cmresidual(f, v, h, ht));
axis image;
colorbar;
title('Residual.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, sprintf('%s-cm-residual.png', name)), '-png', '-q300', '-a1', '-transparent');

%% Mass conservation with source/sink term.

% Spatial and temporal reguarisation of v.
alpha = 0.01;
beta = 0.001;
% Norm of k.
gamma = 0.1;
% Spatial and temporal regularisation of k.
delta = 0.001;
eta = 0.001;

% Create linear system.
[A, B, C, D, E, F, b] = cms(f, h, ht);

% Solve system for mass conservation with source/sink term.
[x, ~, relres, iter] = gmres(A + alpha*B + beta*C + gamma*D + delta*E + eta*F, b, [], tolSolver, iterSolver);
fprintf('GMRES iter %i, relres %e\n', iter(1)*iter(2), relres);

% Recover flow.
v = reshape(x(1:t*n), n, t)';
k = reshape(x(t*n+1:end), n, t)';

% Visualise flow.
figure(7);
imagesc(f);
axis image;
colorbar;
colormap gray;
streamline(v, ones(t, n), 1:2:n, ones(ceil(n/2), 1));
title('Input image with streamlines superimposed.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, sprintf('%s-cms-input.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(8);
imagesc(v);
axis image;
colorbar;
title('Velocity field for mass conservation with source/sink term.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, sprintf('%s-cms-velocity.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(9);
imagesc(k);
axis image;
colorbar;
title('Source/sink term.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, sprintf('%s-cms-source.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(10);
imagesc(cmsresidual(f, v, k, h, ht));
axis image;
colorbar;
title('Residual.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, sprintf('%s-cms-residual.png', name)), '-png', '-q300', '-a1', '-transparent');

%% Mass conservation with convective regularisation.

% Spatial and temporal reguarisation of v.
alpha = 0.01;
beta = 0.001;
% Norm of k.
gamma = 0.001;
% Spatial and temporal regularisation of k.
delta = 0;
eta = 0;
% Convective regularisation of k.
theta = 0.001;

% Create linear system.
[A, B, C, D, E, F, b] = cms(f, h, ht);

% Solve system for mass conservation with source/sink term.
[x, ~, relres, iter] = gmres(A + alpha*B + beta*C + gamma*D + delta*E + eta*F, b, [], tolSolver, iterSolver);
fprintf('GMRES iter %i, relres %e\n', iter(1)*iter(2), relres);

% Recover flow.
v = reshape(x(1:t*n), n, t)';
k = reshape(x(t*n+1:end), n, t)';

figure(11);
imagesc(f);
axis image;
colorbar;
colormap gray;
streamline(v, ones(t, n), 1:2:n, ones(ceil(n/2), 1));
title('Input image with streamlines superimposed.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, sprintf('%s-cmcr-input-000.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(12);
imagesc(v);
axis image;
colorbar;
title('Velocity field for MC with source/sink term and convective regularisation.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, sprintf('%s-cmcr-velocity-000.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(13);
imagesc(k);
axis image;
colorbar;
title('Source/sink term with convective regularisation.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, sprintf('%s-cmcr-source-000.png', name)), '-png', '-q300', '-a1', '-transparent');

for j=1:niter

    % Create linear system for k.
    [A, B, b] = cmcrk(f, v, h, ht);

    % Solve system.
    [x, ~, relres, iter] = gmres(A + theta*B, b, [], tolSolver, iterSolver);
    fprintf('GMRES iter %i, relres %e\n', iter(1)*iter(2), relres);

    % Recover source.
    k = reshape(x, n, t)';
    
    % Create linear system for v.
    [A, B, C, D, b, c] = cmcrv(f, k, h, ht);

    % Solve system.
    [x, ~, relres, iter] = gmres(A + alpha*B + beta*C + theta*D, b + theta*c, [], tolSolver, iterSolver);
    fprintf('GMRES iter %i, relres %e\n', iter(1)*iter(2), relres);

    % Recover flow.
    v = reshape(x, n, t)';
    
    figure(14);
    imagesc(f);
    axis image;
    colorbar;
    colormap gray;
    streamline(v, ones(t, n), 1:2:n, ones(ceil(n/2), 1));
    title('Input image with streamlines superimposed.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, sprintf('%s-cmcr-input-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    
    figure(15);
    imagesc(v);
    axis image;
    colorbar;
    title('Velocity field for MC with source/sink term and convective regularisation.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, sprintf('%s-cmcr-velocity-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    
    figure(16);
    imagesc(k);
    axis image;
    colorbar;
    title('Source/sink term with convective regularisation.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, sprintf('%s-cmcr-source-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    
    figure(17);
    imagesc(cmsresidual(f, v, k, h, ht));
    axis image;
    colorbar;
    title('Residual.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, sprintf('%s-cmcr-residual-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    drawnow();
end

%% Joint image, velocity, and source estimation with convective regularisation.

% Spatial and temporal reguarisation of v.
alpha = 0.1;
beta = 0.01;
% Norm of k.
gamma = 0.001;
% Spatial and temporal regularisation of k.
delta = 0;
eta = 0;
% Convective regularisation of k.
theta = 0.001;
% Data term for f.
kappa = 1000;
% Spatial and temporal regularisation for f.
lambda = 0.01;
mu = 0.01;

% Create linear system.
[A, B, C, D, E, F, b] = cms(fdelta, h, ht);

% Solve system for mass conservation with source/sink term.
[x, ~, relres, iter] = gmres(A + alpha*B + beta*C + gamma*D + delta*E + eta*F, b, [], tolSolver, iterSolver);
fprintf('GMRES iter %i, relres %e\n', iter(1)*iter(2), relres);

% Recover flow.
v = reshape(x(1:t*n), n, t)';
k = reshape(x(t*n+1:end), n, t)';

figure(18);
imagesc(fdelta);
axis image;
colorbar;
colormap gray;
streamline(v, ones(t, n), 1:2:n, ones(ceil(n/2), 1));
title('Input image with streamlines superimposed.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, sprintf('%s-cmje-input-000.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(19);
imagesc(v);
axis image;
colorbar;
title('Velocity field for MC with source/sink term and convective regularisation.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, sprintf('%s-cmje-velocity-000.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(20);
imagesc(k);
axis image;
colorbar;
title('Source/sink term with convective regularisation.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, sprintf('%s-cmje-source-000.png', name)), '-png', '-q300', '-a1', '-transparent');

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
    [x, ~, relres, iter] = gmres(A + kappa*B + lambda*C + mu*D, kappa*b + c, [], tolSolver, iterSolver);
    fprintf('GMRES iter %i, relres %e\n', iter(1)*iter(2), relres);

    % Recover image.
    f = reshape(x, n, t)';
    
    % Create linear system for v.
    [A, B, C, D, b, c] = cmcrv(f, k, h, ht);

    % Solve system.
    [x, ~, relres, iter] = gmres(A + alpha*B + beta*C + theta*D, b + theta*c, [], tolSolver, iterSolver);
    fprintf('GMRES iter %i, relres %e\n', iter(1)*iter(2), relres);

    % Recover flow.
    v = reshape(x, n, t)';
    
    figure(18);
    imagesc(fdelta);
    axis image;
    colorbar;
    colormap gray;
    streamline(v, ones(t, n), 1:2:n, ones(ceil(n/2), 1));
    title('Input image with streamlines superimposed.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, sprintf('%s-cmje-input-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    
    figure(21);
    imagesc(f);
    axis image;
    colorbar;
    colormap gray;
    title('Denoised image.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, sprintf('%s-cmje-image-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    
    figure(19);
    imagesc(v);
    axis image;
    colorbar;
    title('Velocity field for MC with source/sink term and convective regularisation.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, sprintf('%s-cmje-velocity-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    
    figure(20);
    imagesc(k);
    axis image;
    colorbar;
    title('Source/sink term with convective regularisation.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, sprintf('%s-cmje-source-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    
    figure(25);
    imagesc(cmsresidual(f, v, k, h, ht));
    axis image;
    colorbar;
    title('Residual.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, sprintf('%s-cmje-residual-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    drawnow();
end