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

% Set norm regularisation parameter.
epsilon = 1e-3;

% Read data.
g = imread(fullfile(path, sprintf('%s.png', name)));

% Remove cut.
f = padarray(double(g([1:5, 7:end], :)), [0, 10]);

% Remove cut and everything before.
%f = padarray(double(g(7:end, :)), [0, 10]);
[t, n] = size(f);

% Set scaling parameters.
h = 1/(n-1);
ht = 1/(t-1);

% Scale image to [0, 1].
fdelta = (f - min(f(:))) / max(f(:) - min(f(:)));

% Filter image.
f = imfilter(fdelta, fspecial('gaussian', 5, 5), 'replicate');

%% Optical flow.

% Spatial and temporal reguarisation of v.
alpha = 0.01;
beta = 0.001;

% Create output folder. 
mkdir(fullfile(outputPath, 'of'));

% Create linear system.
[A, B, C, b] = of(f, h, ht);

% Solve system.
[x, flag, relres, iter] = gmres(A + alpha*B + beta*C, b, [], tolSolver, iterSolver);
fprintf('GMRES iter %i, relres %e\n', iter(1)*iter(2), relres);

% Recover flow.
v = reshape(x, n, t)';

% Visualise flow.
figure(1);
imagesc(0:h:1, 0:ht:1, f);
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
colormap gray;
[X, Y] = meshgrid(0:h:1, 0:ht:1);
streamline(X, Y, ht*v, ht*ones(t, n), 0:2*h:1, ht*zeros(ceil(n/2), 1));
title('Input image with streamlines superimposed.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'of', sprintf('%s-input.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(2);
imagesc(0:h:1, 0:ht:1, v);
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
title('Velocity field for optical flow.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'of', sprintf('%s-velocity.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(3);
imagesc(0:h:1, 0:ht:1, ofresidual(f, v, h, ht));
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
title('Residual.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'of', sprintf('%s-residual.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(4);
imagesc(0:h:1, 0:ht:1, warpof(f, v, h, ht));
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
colormap gray;
title('Warped image.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'of', sprintf('%s-warp.png', name)), '-png', '-q300', '-a1', '-transparent');

% Compute transport.
fw = zeros(t, n);
fw(1, :) = f(1, :);
for k=2:t
    % Create linear system for transport.
    [At, bt] = oftransport(fw(k-1, :)', v(k-1, :)', h, ht);

    % Solve system.
    [xt, ~, relres, iter] = gmres(At, bt, [], tolSolverTransport, min(iterSolverTransport, size(At, 1)));
    fprintf('GMRES iter %i, relres %e\n', iter(1)*iter(2), relres);

    % Save result.
    fw(k, :) = xt';
end

figure(5);
imagesc(0:h:1, 0:ht:1, fw);
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
colormap gray;
title('Transport.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'of', sprintf('%s-transport.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(6);
imagesc(0:h:1, 0:ht:1, abs(f - fw));
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
title('Absolute difference between image and transported image.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'of', sprintf('%s-diff.png', name)), '-png', '-q300', '-a1', '-transparent');

%% Optical flow with regularised TV.

% Spatial and temporal reguarisation of v.
alpha = 0.01;
beta = 0.001;

% Create output folder. 
mkdir(fullfile(outputPath, 'ofrtv'));

% Create initial guess.
v = zeros(t, n);

% Create linear system.
[A, ~, C, b] = of(f, h, ht);

for j=1:niter

    % Create div(grad v / rnorm(v)) matrix.
    [vx, ~] = gradient(v, h, ht);
    B = divgrad1d(rnorm(vx, epsilon), h);
    
    % Solve system.
    [x, flag, relres, iter] = gmres(A + alpha*B + beta*C, b, [], tolSolver, iterSolver);
    fprintf('GMRES iter %i, relres %e\n', iter(1)*iter(2), relres);

    % Recover flow.
    v = reshape(x, n, t)';

    % Visualise flow.
    figure(1);
    imagesc(0:h:1, 0:ht:1, f);
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    colormap gray;
    [X, Y] = meshgrid(0:h:1, 0:ht:1);
    streamline(X, Y, ht*v, ht*ones(t, n), 0:2*h:1, ht*zeros(ceil(n/2), 1));
    title('Input image with streamlines superimposed.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'ofrtv', sprintf('%s-input-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');

    figure(2);
    imagesc(0:h:1, 0:ht:1, v);
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    title('Velocity field for optical flow.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'ofrtv', sprintf('%s-velocity-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');

    figure(3);
    imagesc(0:h:1, 0:ht:1, ofresidual(f, v, h, ht));
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    title('Residual.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'ofrtv', sprintf('%s-residual-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');

    figure(4);
    imagesc(0:h:1, 0:ht:1, warpof(f, v, h, ht));
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    colormap gray;
    title('Warped image.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'ofrtv', sprintf('%s-warp-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    
    % Compute transport.
    fw = zeros(t, n);
    fw(1, :) = f(1, :);
    for k=2:t
        % Create linear system for transport.
        [At, bt] = oftransport(fw(k-1, :)', v(k-1, :)', h, ht);

        % Solve system.
        [xt, ~, relres, iter] = gmres(At, bt, [], tolSolverTransport, min(iterSolverTransport, size(At, 1)));
        fprintf('GMRES iter %i, relres %e\n', iter(1)*iter(2), relres);

        % Save result.
        fw(k, :) = xt';
    end

    figure(5);
    imagesc(0:h:1, 0:ht:1, fw);
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    colormap gray;
    title('Transport.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'ofrtv', sprintf('%s-transport-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');

    figure(6);
    imagesc(0:h:1, 0:ht:1, abs(f - fw));
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    title('Absolute difference between image and transported image.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'ofrtv', sprintf('%s-diff-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    
    figure(7);
    imagesc(0:h:1, 0:ht:1, rnorm(vx, epsilon));
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    title('Regularised norm of $\partial_x v$.', 'FontName', 'Helvetica', 'FontSize', 14, 'Interpreter', 'latex');
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'ofrtv', sprintf('%s-rnorm-vx-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');  
end

%% Mass conservation.

% Spatial and temporal reguarisation of v.
alpha = 0.025;
beta = 0.01;

% Create output folder. 
mkdir(fullfile(outputPath, 'cm'));

% Create linear system for mass conservation.
[A, B, C, b] = cm(f, h, ht);

% Solve system.
[x, ~, relres, iter] = gmres(A + alpha*B + beta*C, b, [], tolSolver, iterSolver);
fprintf('GMRES iter %i, relres %e\n', iter(1)*iter(2), relres);

% Recover flow.
v = reshape(x, n, t)';

figure(1);
imagesc(0:h:1, 0:ht:1, f);
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
colormap gray;
[X, Y] = meshgrid(0:h:1, 0:ht:1);
streamline(X, Y, ht*v, ht*ones(t, n), 0:2*h:1, ht*zeros(ceil(n/2), 1));
title('Mass conservation without source/sink term.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'cm', sprintf('%s-input.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(2);
imagesc(0:h:1, 0:ht:1, v);
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
title('Velocity field for mass conservation without source/sink term.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
set(gca, 'FontName', 'Helvetica');
set(gca, 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'cm', sprintf('%s-velocity.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(3);
imagesc(0:h:1, 0:ht:1, cmresidual(f, v, h, ht));
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
title('Residual.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'cm', sprintf('%s-residual.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(4);
imagesc(0:h:1, 0:ht:1, warpcms(f, v, zeros(t, n), h, ht));
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
colormap gray;
title('Warped image.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'cm', sprintf('%s-warp.png', name)), '-png', '-q300', '-a1', '-transparent');

% Compute transport.
fw = zeros(t, n);
fw(1, :) = f(1, :);
for k=2:t
    % Create linear system for transport.
    [At, bt] = cmstransport(fw(k-1, :)', v(k-1, :)', zeros(n, 1), h, ht);

    % Solve system.
    [xt, ~, relres, iter] = gmres(At, bt, [], tolSolverTransport, min(iterSolverTransport, size(At, 1)));
    fprintf('GMRES iter %i, relres %e\n', iter(1)*iter(2), relres);

    % Save result.
    fw(k, :) = xt';
end

figure(5);
imagesc(0:h:1, 0:ht:1, fw);
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
colormap gray;
title('Transport.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'cm', sprintf('%s-transport.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(6);
imagesc(0:h:1, 0:ht:1, abs(f - fw));
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
title('Absolute difference between image and transported image.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'cm', sprintf('%s-diff.png', name)), '-png', '-q300', '-a1', '-transparent');

%% Mass conservation with regularised TV.

% Spatial and temporal reguarisation of v.
alpha = 0.025;
beta = 0.01;

% Create output folder. 
mkdir(fullfile(outputPath, 'cmrtv'));

% Create initial guess.
v = zeros(t, n);

% Create linear system for mass conservation.
[A, ~, C, b] = cm(f, h, ht);

for j=1:niter

    % Create div(grad v / rnorm(v)) matrix.
    [vx, ~] = gradient(v, h, ht);
    B = divgrad1d(rnorm(vx, epsilon), h);
    
    % Solve system.
    [x, flag, relres, iter] = gmres(A + alpha*B + beta*C, b, [], tolSolver, iterSolver);
    fprintf('GMRES iter %i, relres %e\n', iter(1)*iter(2), relres);

    % Recover flow.
    v = reshape(x, n, t)';

    % Visualise flow.
    figure(1);
    imagesc(0:h:1, 0:ht:1, f);
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    colormap gray;
    [X, Y] = meshgrid(0:h:1, 0:ht:1);
    streamline(X, Y, ht*v, ht*ones(t, n), 0:2*h:1, ht*zeros(ceil(n/2), 1));
    title('Mass conservation without source/sink term.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'cmrtv', sprintf('%s-input-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');

    figure(2);
    imagesc(0:h:1, 0:ht:1, v);
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    title('Velocity field for mass conservation without source/sink term.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'cmrtv', sprintf('%s-velocity-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');

    figure(3);
    imagesc(0:h:1, 0:ht:1, ofresidual(f, v, h, ht));
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    title('Residual.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'cmrtv', sprintf('%s-residual-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');

    figure(4);
    imagesc(0:h:1, 0:ht:1, warpof(f, v, h, ht));
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    colormap gray;
    title('Warped image.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'cmrtv', sprintf('%s-warp-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    
    % Compute transport.
    fw = zeros(t, n);
    fw(1, :) = f(1, :);
    for k=2:t
        % Create linear system for transport.
        [At, bt] = cmstransport(fw(k-1, :)', v(k-1, :)', zeros(n, 1), h, ht);

        % Solve system.
        [xt, ~, relres, iter] = gmres(At, bt, [], tolSolverTransport, min(iterSolverTransport, size(At, 1)));
        fprintf('GMRES iter %i, relres %e\n', iter(1)*iter(2), relres);

        % Save result.
        fw(k, :) = xt';
    end

    figure(5);
    imagesc(0:h:1, 0:ht:1, fw);
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    colormap gray;
    title('Transport.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'cmrtv', sprintf('%s-transport-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');

    figure(6);
    imagesc(0:h:1, 0:ht:1, abs(f - fw));
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    title('Absolute difference between image and transported image.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'cmrtv', sprintf('%s-diff-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    
    figure(7);
    imagesc(0:h:1, 0:ht:1, rnorm(vx, epsilon));
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    title('Regularised norm of $\partial_x v$.', 'FontName', 'Helvetica', 'FontSize', 14, 'Interpreter', 'latex');
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'cmrtv', sprintf('%s-rnorm-vx-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');

end

%% Mass conservation with source/sink term.

% Spatial and temporal reguarisation of v.
alpha = 0.025;
beta = 0.01;
% Norm of k.
gamma = 0.01;
% Spatial and temporal regularisation of k.
delta = 0.001;
eta = 0.001;

% Create output folder. 
mkdir(fullfile(outputPath, 'cms'));

% Create linear system.
[A, B, C, D, E, F, b] = cms(f, h, ht);

% Solve system for mass conservation with source/sink term.
[x, ~, relres, iter] = gmres(A + alpha*B + beta*C + gamma*D + delta*E + eta*F, b, [], tolSolver, iterSolver);
fprintf('GMRES iter %i, relres %e\n', iter(1)*iter(2), relres);

% Recover flow.
v = reshape(x(1:t*n), n, t)';
k = reshape(x(t*n+1:end), n, t)';

% Visualise flow.
figure(1);
imagesc(0:h:1, 0:ht:1, f);
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
colormap gray;
[X, Y] = meshgrid(0:h:1, 0:ht:1);
streamline(X, Y, ht*v, ht*ones(t, n), 0:2*h:1, ht*zeros(ceil(n/2), 1));
title('Input image with streamlines superimposed.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'cms', sprintf('%s-input.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(2);
imagesc(0:h:1, 0:ht:1, v);
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
title('Velocity field for mass conservation with source/sink term.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'cms', sprintf('%s-velocity.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(3);
imagesc(0:h:1, 0:ht:1, k);
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
title('Source/sink term.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'cms', sprintf('%s-source.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(4);
imagesc(0:h:1, 0:ht:1, cmsresidual(f, v, k, h, ht));
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
title('Residual.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'cms', sprintf('%s-residual.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(5);
imagesc(0:h:1, 0:ht:1, warpcms(f, v, k, h, ht));
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
colormap gray;
title('Warped image.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'cms', sprintf('%s-warp.png', name)), '-png', '-q300', '-a1', '-transparent');

% Compute transport.
fw = zeros(t, n);
fw(1, :) = f(1, :);
for j=2:t
    % Create linear system for transport.
    [At, bt] = cmstransport(fw(j-1, :)', v(j-1, :)', k(j-1, :)', h, ht);

    % Solve system.
    [xt, ~, relres, iter] = gmres(At, bt, [], tolSolverTransport, min(iterSolverTransport, size(At, 1)));
    fprintf('GMRES iter %i, relres %e\n', iter(1)*iter(2), relres);

    % Save result.
    fw(j, :) = xt';
end

figure(5);
imagesc(0:h:1, 0:ht:1, fw);
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
colormap gray;
title('Transport.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'cms', sprintf('%s-transport.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(6);
imagesc(0:h:1, 0:ht:1, abs(f - fw));
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
title('Absolute difference between image and transported image.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'cms', sprintf('%s-diff.png', name)), '-png', '-q300', '-a1', '-transparent');

%% Mass conservation with source/sink term and regularised TV.

% Spatial and temporal reguarisation of v.
alpha = 0.025;
beta = 0.01;
% Norm of k.
gamma = 0.01;
% Spatial and temporal regularisation of k.
delta = 0.001;
eta = 0.001;

% Create output folder. 
mkdir(fullfile(outputPath, 'cmsrtv'));

% Create initial guess.
v = zeros(t, n);

% Create linear system for mass conservation with source/sink term.
[A, ~, C, D, E, F, b] = cms(f, h, ht);

for j=1:niter

    % Create div(grad v / rnorm(v)) matrix.
    [vx, ~] = gradient(v, h, ht);
    B = [divgrad1d(rnorm(vx, epsilon), h), sparse(t*n, t*n); sparse(t*n, 2*t*n)];

    % Solve system for mass conservation with source/sink term.
    [x, ~, relres, iter] = gmres(A + alpha*B + beta*C + gamma*D + delta*E + eta*F, b, [], tolSolver, iterSolver);
    fprintf('GMRES iter %i, relres %e\n', iter(1)*iter(2), relres);

    % Recover flow.
    v = reshape(x(1:t*n), n, t)';
    k = reshape(x(t*n+1:end), n, t)';

    % Visualise flow.
    figure(1);
    imagesc(0:h:1, 0:ht:1, f);
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    colormap gray;
    [X, Y] = meshgrid(0:h:1, 0:ht:1);
    streamline(X, Y, ht*v, ht*ones(t, n), 0:2*h:1, ht*zeros(ceil(n/2), 1));
    title('Input image with streamlines superimposed.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'cmsrtv', sprintf('%s-input-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');

    figure(2);
    imagesc(0:h:1, 0:ht:1, v);
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    title('Velocity field for mass conservation with source/sink term.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'cmsrtv', sprintf('%s-velocity-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');

    figure(3);
    imagesc(0:h:1, 0:ht:1, k);
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    title('Source/sink term.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'cmsrtv', sprintf('%s-source-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');

    figure(4);
    imagesc(0:h:1, 0:ht:1, cmsresidual(f, v, k, h, ht));
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    title('Residual.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'cmsrtv', sprintf('%s-residual-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');

    figure(5);
    imagesc(0:h:1, 0:ht:1, warpcms(f, v, k, h, ht));
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    colormap gray;
    title('Warped image.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'cmsrtv', sprintf('%s-warp-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');

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

    figure(5);
    imagesc(0:h:1, 0:ht:1, fw);
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    colormap gray;
    title('Transport.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'cmsrtv', sprintf('%s-transport-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');

    figure(6);
    imagesc(0:h:1, 0:ht:1, abs(f - fw));
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    title('Absolute difference between image and transported image.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'cmsrtv', sprintf('%s-diff-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    
    figure(7);
    imagesc(0:h:1, 0:ht:1, rnorm(vx, epsilon));
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    title('Regularised norm of $\partial_x v$.', 'FontName', 'Helvetica', 'FontSize', 14, 'Interpreter', 'latex');
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'cmsrtv', sprintf('%s-rnorm-vx-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');

end

%% Mass conservation with source/sink and convective regularisation.

% Spatial and temporal reguarisation of v.
alpha = 0.025;
beta = 0.01;
% Norm of k.
gamma = 0.001;
% Spatial and temporal regularisation of k.
delta = 0;
eta = 0;
% Convective regularisation of k.
theta = 0.001;

% Create output folder. 
mkdir(fullfile(outputPath, 'cmcr'));

% Create linear system.
[A, B, C, D, E, F, b] = cms(f, h, ht);

% Solve system for mass conservation with source/sink term.
[x, ~, relres, iter] = gmres(A + alpha*B + beta*C + gamma*D + delta*E + eta*F, b, [], tolSolver, iterSolver);
fprintf('GMRES iter %i, relres %e\n', iter(1)*iter(2), relres);

% Recover flow.
v = reshape(x(1:t*n), n, t)';
k = reshape(x(t*n+1:end), n, t)';

figure(1);
imagesc(0:h:1, 0:ht:1, f);
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
colormap gray;
[X, Y] = meshgrid(0:h:1, 0:ht:1);
streamline(X, Y, ht*v, ht*ones(t, n), 0:2*h:1, ht*zeros(ceil(n/2), 1));
title('Input image with streamlines superimposed.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'cmcr', sprintf('%s-input-000.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(2);
imagesc(0:h:1, 0:ht:1, v);
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
title('Velocity field for MC with source/sink term and convective regularisation.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'cmcr', sprintf('%s-velocity-000.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(3);
imagesc(0:h:1, 0:ht:1, k);
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
title('Source/sink term with convective regularisation.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'cmcr', sprintf('%s-source-000.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(4);
imagesc(0:h:1, 0:ht:1, cmsresidual(f, v, k, h, ht));
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
title('Residual.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'cmcr', sprintf('%s-residual-000.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(5);
imagesc(0:h:1, 0:ht:1, warpcms(f, v, k, h, ht));
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
colormap gray;
title('Warped image.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'cmcr', sprintf('%s-warp-000.png', name)), '-png', '-q300', '-a1', '-transparent');

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
export_fig(gcf, fullfile(outputPath, 'cmcr', sprintf('%s-transport-000.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(7);
imagesc(0:h:1, 0:ht:1, abs(f - fw));
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
title('Absolute difference between image and transported image.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'cmcr', sprintf('%s-diff-000.png', name)), '-png', '-q300', '-a1', '-transparent');

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
    
    figure(1);
    imagesc(0:h:1, 0:ht:1, f);
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    colormap gray;
    [X, Y] = meshgrid(0:h:1, 0:ht:1);
    streamline(X, Y, ht*v, ht*ones(t, n), 0:2*h:1, ht*zeros(ceil(n/2), 1));
    title('Input image with streamlines superimposed.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'cmcr', sprintf('%s-input-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    
    figure(2);
    imagesc(0:h:1, 0:ht:1, v);
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    title('Velocity field for MC with source/sink term and convective regularisation.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'cmcr', sprintf('%s-velocity-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    
    figure(3);
    imagesc(0:h:1, 0:ht:1, k);
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    title('Source/sink term with convective regularisation.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'cmcr', sprintf('%s-source-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    
    figure(4);
    imagesc(0:h:1, 0:ht:1, cmsresidual(f, v, k, h, ht));
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    title('Residual.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'cmcr', sprintf('%s-residual-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    
    figure(5);
    imagesc(0:h:1, 0:ht:1, warpcms(f, v, k, h, ht));
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    colormap gray;
    title('Warped image.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'cmcr', sprintf('%s-warp-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    
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
    export_fig(gcf, fullfile(outputPath, 'cmcr', sprintf('%s-transport-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');

    figure(7);
    imagesc(0:h:1, 0:ht:1, abs(f - fw));
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    title('Absolute difference between image and transported image.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'cmcr', sprintf('%s-diff-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    drawnow();
end

%% Mass conservation with source/sink, convective regularisation, and regularised TV.

% Spatial and temporal reguarisation of v.
alpha = 0.025;
beta = 0.01;
% Norm of k.
gamma = 0.001;
% Spatial and temporal regularisation of k.
delta = 0;
eta = 0;
% Convective regularisation of k.
theta = 0.001;

% Create output folder. 
mkdir(fullfile(outputPath, 'cmcrrtv'));

% Create linear system.
[A, B, C, D, E, F, b] = cms(f, h, ht);

% Solve system for mass conservation with source/sink term.
[x, ~, relres, iter] = gmres(A + alpha*B + beta*C + gamma*D + delta*E + eta*F, b, [], tolSolver, iterSolver);
fprintf('GMRES iter %i, relres %e\n', iter(1)*iter(2), relres);

% Recover flow.
v = reshape(x(1:t*n), n, t)';
k = reshape(x(t*n+1:end), n, t)';

figure(1);
imagesc(0:h:1, 0:ht:1, f);
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
colormap gray;
[X, Y] = meshgrid(0:h:1, 0:ht:1);
streamline(X, Y, ht*v, ht*ones(t, n), 0:2*h:1, ht*zeros(ceil(n/2), 1));
title('Input image with streamlines superimposed.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'cmcrrtv', sprintf('%s-input-000.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(2);
imagesc(0:h:1, 0:ht:1, v);
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
title('Velocity field for MC with source/sink term and convective regularisation.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'cmcrrtv', sprintf('%s-velocity-000.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(3);
imagesc(0:h:1, 0:ht:1, k);
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
title('Source/sink term with convective regularisation.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'cmcrrtv', sprintf('%s-source-000.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(4);
imagesc(0:h:1, 0:ht:1, cmsresidual(f, v, k, h, ht));
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
title('Residual.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'cmcrrtv', sprintf('%s-residual-000.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(5);
imagesc(0:h:1, 0:ht:1, warpcms(f, v, k, h, ht));
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
colormap gray;
title('Warped image.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'cmcrrtv', sprintf('%s-warp-000.png', name)), '-png', '-q300', '-a1', '-transparent');

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
export_fig(gcf, fullfile(outputPath, 'cmcrrtv', sprintf('%s-transport-000.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(7);
imagesc(0:h:1, 0:ht:1, abs(f - fw));
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
title('Absolute difference between image and transported image.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'cmcrrtv', sprintf('%s-diff-000.png', name)), '-png', '-q300', '-a1', '-transparent');

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
    
    figure(1);
    imagesc(0:h:1, 0:ht:1, f);
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    colormap gray;
    [X, Y] = meshgrid(0:h:1, 0:ht:1);
    streamline(X, Y, ht*v, ht*ones(t, n), 0:2*h:1, ht*zeros(ceil(n/2), 1));
    title('Input image with streamlines superimposed.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'cmcrrtv', sprintf('%s-input-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    
    figure(2);
    imagesc(0:h:1, 0:ht:1, v);
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    title('Velocity field for MC with source/sink term and convective regularisation.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'cmcrrtv', sprintf('%s-velocity-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    
    figure(3);
    imagesc(0:h:1, 0:ht:1, k);
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    title('Source/sink term with convective regularisation.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'cmcrrtv', sprintf('%s-source-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    
    figure(4);
    imagesc(0:h:1, 0:ht:1, cmsresidual(f, v, k, h, ht));
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    title('Residual.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'cmcrrtv', sprintf('%s-residual-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    
    figure(5);
    imagesc(0:h:1, 0:ht:1, warpcms(f, v, k, h, ht));
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    colormap gray;
    title('Warped image.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'cmcrrtv', sprintf('%s-warp-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    
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
    export_fig(gcf, fullfile(outputPath, 'cmcrrtv', sprintf('%s-transport-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');

    figure(7);
    imagesc(0:h:1, 0:ht:1, abs(f - fw));
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    title('Absolute difference between image and transported image.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'cmcrrtv', sprintf('%s-diff-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    drawnow();
end

%% Joint image, velocity, and source estimation with convective regularisation.

% Spatial and temporal reguarisation of v.
alpha = 0.025;
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

% Create output folder. 
mkdir(fullfile(outputPath, 'cmje'));

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
export_fig(gcf, fullfile(outputPath, 'cmje', sprintf('%s-input-000.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(2);
imagesc(0:h:1, 0:ht:1, v);
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
title('Velocity field for MC with source/sink term and convective regularisation.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'cmje', sprintf('%s-velocity-000.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(3);
imagesc(0:h:1, 0:ht:1, k);
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
title('Source/sink term with convective regularisation.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'cmje', sprintf('%s-source-000.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(4);
imagesc(0:h:1, 0:ht:1, cmsresidual(fdelta, v, k, h, ht));
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
title('Residual.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'cmje', sprintf('%s-residual-000.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(5);
imagesc(0:h:1, 0:ht:1, warpcms(fdelta, v, k, h, ht));
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
colormap gray;
title('Warped image.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'cmje', sprintf('%s-warp-000.png', name)), '-png', '-q300', '-a1', '-transparent');

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
export_fig(gcf, fullfile(outputPath, 'cmje', sprintf('%s-transport-000.png', name)), '-png', '-q300', '-a1', '-transparent');

figure(7);
imagesc(0:h:1, 0:ht:1, abs(f - fw));
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
title('Absolute difference between image and transported image.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
export_fig(gcf, fullfile(outputPath, 'cmje', sprintf('%s-diff-000.png', name)), '-png', '-q300', '-a1', '-transparent');

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
    export_fig(gcf, fullfile(outputPath, 'cmje', sprintf('%s-input-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    
    figure(2);
    imagesc(0:h:1, 0:ht:1, v);
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    title('Velocity field for MC with source/sink term and convective regularisation.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'cmje', sprintf('%s-velocity-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    
    figure(3);
    imagesc(0:h:1, 0:ht:1, k);
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    title('Source/sink term with convective regularisation.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'cmje', sprintf('%s-source-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    
    figure(4);
    imagesc(0:h:1, 0:ht:1, cmsresidual(f, v, k, h, ht));
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    title('Residual.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'cmje', sprintf('%s-residual-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    
    figure(5);
    imagesc(0:h:1, 0:ht:1, warpcms(f, v, k, h, ht));
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    colormap gray;
    title('Warped image.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'cmje', sprintf('%s-warp-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    
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
    export_fig(gcf, fullfile(outputPath, 'cmje', sprintf('%s-transport-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');

    figure(7);
    imagesc(0:h:1, 0:ht:1, abs(f - fw));
    set(gca, 'DataAspectRatio', [t, n, 1]);
    colorbar;
    title('Absolute difference between image and transported image.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'cmje', sprintf('%s-diff-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    
    figure(8);
    imagesc(f);
    axis image;
    colorbar;
    colormap gray;
    title('Denoised image.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    export_fig(gcf, fullfile(outputPath, 'cmje', sprintf('%s-image-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
    drawnow();
end