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

% Spatial and temporal reguarisation of v.
alpha = 0.05;
beta = 0.05;
% Norm of k.
gamma = 0.1;
% Spatial and temporal regularisation of k.
delta = 0.001;
eta = 0.001;
% Convective regularisation of k.
theta = 0.005;
% Data term for f.
kappa = 3000;
% Spatial and temporal regularisation for f.
lambda = 0.0001;
mu = 0.0001;

% Number of iterations for convective regularisation.
niter = 5;

% Read data.
f = imread(fullfile(path, sprintf('%s.png', name)));

% Remove cut and everything before.
f = double(f(7:end, 1:140));
[t, n] = size(f);

% Set scaling parameters.
h = 1/n;
ht = 1/t;

% Set streamline scaling factor (determined experimentally).
hs = 1;

% Scale image to [0, 1].
fdelta = (f - min(f(:))) / max(f(:) - min(f(:)));

% Filter image.
%f = imfilter(f, fspecial('gaussian', 5, 10), 'replicate');

%% Mass conservation with source/sink term.

% Create linear system.
[A, B, C, D, E, F, b] = cms(fdelta, h, ht);

% Solve system for mass conservation with source/sink term.
[x, ~, relres, iter] = gmres(A + alpha*B + beta*C + gamma*D + delta*E + eta*F, b, [], 1e-3, 2000);
fprintf('GMRES iter %i, relres %e\n', iter(1)*iter(2), relres);

% Recover flow.
v = reshape(x(1:t*n), n, t)';
k = reshape(x(t*n+1:end), n, t)';

% Visualise flow.
figure(1);
imagesc(fdelta);
axis image;
colorbar;
colormap gray;
streamline(v*hs, ones(t, n), 1:n, ones(n, 1));
title('Mass conservation with source/sink term.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, sprintf('%s-cms-input.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(2);
imagesc(v);
axis image;
colorbar;
title('Velocity field for mass conservation with source/sink term.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, sprintf('%s-cms-velocity.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(3);
imagesc(k);
axis image;
colorbar;
title('Source/sink term.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, sprintf('%s-cms-source.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(4);
imagesc(cmsresidual(fdelta, v, k, h, ht));
axis image;
colorbar;
title('Residual.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, sprintf('%s-cms-residual.png', name)), '-png', '-q300', '-a1', '-transparent');

%% Mass conservation.

% Create linear system for mass conservation.
[A, B, C, b] = cm(fdelta, h, ht);

% Solve system.
[x, ~, relres, iter] = gmres(A + alpha*B + beta*C, b, [], 1e-3, 2000);
fprintf('GMRES iter %i, relres %e\n', iter(1)*iter(2), relres);

% Recover flow.
v = reshape(x, n, t)';

figure(5);
imagesc(fdelta);
axis image;
colorbar;
colormap gray;
streamline(v*hs, ones(t, n), 1:n, ones(n, 1));
title('Mass conservation without source/sink term.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, sprintf('%s-cm-input.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(6);
imagesc(v);
axis image;
colorbar;
title('Velocity field for mass conservation without source/sink term.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
set(gca, 'FontName', 'Helvetica');
set(gca, 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, sprintf('%s-cm-velocity.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(7);
imagesc(cmresidual(fdelta, v, h, ht));
axis image;
colorbar;
title('Residual.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, sprintf('%s-cm-residual.png', name)), '-png', '-q300', '-a1', '-transparent');

%% Convective regularisation.

% Create linear system.
[A, B, C, D, E, F, b] = cms(fdelta, h, ht);

% Solve system for mass conservation with source/sink term.
[x, ~, relres, iter] = gmres(A + alpha*B + beta*C + gamma*D, b, [], 1e-3, 2000);
fprintf('GMRES iter %i, relres %e\n', iter(1)*iter(2), relres);

% Recover flow.
v = reshape(x(1:t*n), n, t)';

f = fdelta;
for j=1:niter

    % Create linear system for k.
    [A, B, b] = cmcrk(f, v, h, ht);

    % Solve system.
    [x, ~, relres, iter] = gmres(A + theta*B, b, [], 1e-3, 1000);
    fprintf('GMRES iter %i, relres %e\n', iter(1)*iter(2), relres);

    % Recover source.
    k = reshape(x, n, t)';
    
    % Create linear system for f.
    [A, B, C, D, b, c] = cmcrf(fdelta, v, k, h, ht);

    % Solve system.
    [x, ~, relres, iter] = gmres(A + kappa*B + lambda*C + mu*D, kappa*b + c, [], 1e-3, 1000);
    fprintf('GMRES iter %i, relres %e\n', iter(1)*iter(2), relres);

    % Recover flow.
    f = reshape(x, n, t)';
    
    % Create linear system for v.
    [A, B, C, D, b, c] = cmcrv(f, k, h, ht);

    % Solve system.
    [x, ~, relres, iter] = gmres(A + alpha*B + beta*C + theta*D, b + theta*c, [], 1e-3, 1000);
    fprintf('GMRES iter %i, relres %e\n', iter(1)*iter(2), relres);

    % Recover flow.
    v = reshape(x, n, t)';
    
    figure(8);
    imagesc(fdelta);
    axis image;
    colorbar;
    colormap gray;
    streamline(v*hs, ones(t, n), 1:n, ones(n, 1));
    title('MC with source/sink term and convective regularisation.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, sprintf('%s-cmcr-input-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    
    figure(9);
    imagesc(f);
    axis image;
    colorbar;
    colormap gray;
    title('MC with source/sink term and convective regularisation.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, sprintf('%s-cmcr-image-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    
    figure(10);
    imagesc(v);
    axis image;
    colorbar;
    title('Velocity field for MC with source/sink term and convective regularisation.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, sprintf('%s-cmcr-velocity-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    
    figure(11);
    imagesc(k);
    axis image;
    colorbar;
    title('Source/sink term with convective regularisation.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, sprintf('%s-cmcr-source-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    
    figure(12);
    imagesc(cmsresidual(f, v, k, h, ht));
    axis image;
    colorbar;
    title('Residual.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, sprintf('%s-cmcr-residual-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    drawnow();
end