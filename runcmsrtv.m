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

% Define datasets.
path = 'data';
files = [dir(fullfile(path, '*.png')); dir(fullfile(path, '*.tif'))];

% Define time instant of laser cut.
cuts = repmat(6, 1, length(files));

% Create start date and time.
startdate = datestr(now, 'yyyy-mm-dd-HH-MM-SS');

% Define and create output folder.
outputPath = fullfile('results', startdate);
mkdir(outputPath);

% Number of inner iterations.
niterinner = 15;

% Set norm regularisation parameter.
epsilon = 1e-5;

% Spatial and temporal reguarisation of v.
alpha0 = 0.01;
alpha1 = 0.001;
beta0 = 0.01;
beta1 = 0.01;

% Save plots.
saveplots = false;

% Run through all files.
for q=1:length(files)

    % Read data.
    [~, name, ~] = fileparts(files(q).name);
    g = double(imread(fullfile(path, files(q).name)));

    % Scale data to interval [0, 1].
    g = (g - min(g(:))) / max(g(:) - min(g(:)));
    
    % Remove cut.
    fdelta = g([1:cuts(q)-1, cuts(q)+1:end], :);

    % Pad data.
    fdelta = padarray(fdelta, [0, 5]);

    % Get image size.
    [t, n] = size(fdelta);

    % Set scaling parameters.
    h = 1/(n-1);
    ht = 1/(t-1);

    % Filter image.
    f = imfilter(fdelta, fspecial('gaussian', 5, 5), 'symmetric');
    
    % Create output folder.
    alg = 'cmsrtv';
    mkdir(fullfile(outputPath, name, alg));

    % Create initial guess.
    v = zeros(t, n);

    for j=1:niterinner

        % Create linear system for mass conservation.
        [A, b] = cmsrtv(f, v, alpha0, alpha1, beta0, beta1, h, ht, epsilon);
        
        % Solve system.
        x = A \ b;

        % Recover flow.
        v = reshape(x(1:t*n), n, t)';
        k = reshape(x(t*n+1:end), n, t)';

        % Visualise flow.
        plotstreamlines(1, 'Input image with streamlines superimposed.', 'gray', f, v, h, ht, [t, n]);
        plotdata(2, 'Velocity.', 'default', v, h, ht);
        plotdata(3, 'Source.', 'default', k, h, ht);
        res = cmsresidual(f, v, k, h, ht);
        plotdata(4, 'Residual.', 'default', res, h, ht);
        fw = computecmstransport(f, v, k, h, ht);
        plotdata(5, 'Transport.', 'gray', fw, h, ht);
        diff = abs(f - fw);
        plotdata(6, 'Absolute difference between image and transported image.', 'default', diff, h, ht);
        [vx, ~] = gradient(v, h, ht);
        rnormvx = rnorm(vx, epsilon);
        plotdata(7, 'Regularised norm of \partial_x v.', 'default', rnormvx, h, ht);
        drawnow();

        if(saveplots)
            export_fig(1, fullfile(outputPath, name, alg, sprintf('%s-input-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
            export_fig(2, fullfile(outputPath, name, alg, sprintf('%s-velocity-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
            export_fig(3, fullfile(outputPath, name, alg, sprintf('%s-residual-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
            export_fig(4, fullfile(outputPath, name, alg, sprintf('%s-transport-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
            export_fig(5, fullfile(outputPath, name, alg, sprintf('%s-diff-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
            export_fig(6, fullfile(outputPath, name, alg, sprintf('%s-rnorm-vx-%.3i.png', name, j)), '-png', '-q300', '-a1', '-transparent');
        end
    end
end