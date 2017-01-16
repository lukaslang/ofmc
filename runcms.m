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
gamma = 0.5;

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
[A, B, C, D, b] = cms(f, h, ht);

% Solve system.
[x, flag, relres, iter] = gmres(A + alpha*B + beta*C + gamma*D, b, [], 1e-3, 2000);
fprintf('GMRES iter %i, relres %e\n', iter(1)*iter(2), relres);

% Recover flow.
v = reshape(x(1:t*n), n, t)';
k = reshape(x(t*n+1:end), n, t)';

% Visualise flow.
figure(1);
imagesc(f);
colormap gray;
axis image;
colorbar;
figure(2);
imagesc(v);
axis image;
colorbar;
figure(3);
imagesc(k);
axis image;
colorbar;

for k=1:t
    figure(4);
    subplot(2, 1, 1);
    imagesc(f(k, :));
    colormap gray;
    
    subplot(2, 1, 2);
    plot(v(k, :));
    axis([1, n, min(v(:)), max(v(:))]);
    grid on;
    
    pause(0.2);
end