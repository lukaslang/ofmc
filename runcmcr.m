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

% Set parameters.
alpha = 0.01;
beta = 0.05;
theta = 0.01;

% Read data.
f = imread(fullfile(path, sprintf('%s.png', name)));

% Remove cut and everything before.
f = double(f(7:end, 1:140));
[t, n] = size(f);

% Set scaling parameters.
h = 1/n;
ht = 1/t;

% Scale image to [0, 1].
f = (f - min(f(:))) / max(f(:) - min(f(:)));

% Filter image.
f = imfilter(f, fspecial('gaussian', 5, 10), 'replicate');

% Create linear system.
[A, B, C, b] = cm(f, h, ht);

% Solve system.
[x, ~, relres, iter] = gmres(A + alpha*B + beta*C, b, [], 1e-3, 1000);
fprintf('GMRES iter %i, relres %e\n', iter(1)*iter(2), relres);

% Recover flow.
v = reshape(x, n, t)';

% Visualise flow.
figure(1);
imagesc(f);
axis image;
colorbar;
colormap gray;
title('Input image.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);

figure(2);
imagesc(v);
axis image;
colorbar;
title('Velocity field for mass conservation with source/sink term.', 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
drawnow();

niter = 10;
for j=1:niter

    % Create linear system.
    [A, B, b] = cmcrk(f, v, h, ht);

    % Solve system.
    [x, ~, relres, iter] = gmres(A + theta*B, b, [], 1e-3, 1000);
    fprintf('GMRES iter %i, relres %e\n', iter(1)*iter(2), relres);

    % Recover source.
    k = reshape(x, n, t)';

    figure(3);
    imagesc(k);
    axis image;
    colorbar;
    title('Source/sink term.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    drawnow();
    
    % Create linear system.
    [A, B, C, D, b, c] = cmcrv(f, k, h, ht);

    % Solve system.
    [x, ~, relres, iter] = gmres(A + alpha*B + beta*C + theta*D, b + theta*c, [], 1e-3, 1000);
    fprintf('GMRES iter %i, relres %e\n', iter(1)*iter(2), relres);

    % Recover flow.
    v = reshape(x, n, t)';

    figure(2);
    imagesc(v);
    axis image;
    colorbar;
    title('Velocity field for mass conservation with source/sink term.', 'FontName', 'Helvetica', 'FontSize', 14);
    xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
    ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);
    drawnow();
end