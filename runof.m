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

% Spatial and temporal reguarisation of v.
alpha0 = 0.05;
alpha1 = 0.005;

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
    f = imfilter(fdelta, fspecial('gaussian', 3, 3), 'replicate');
    
    % Create output folder.
    alg = 'of';
    mkdir(fullfile(outputPath, name, alg));

    % Create linear system.
    [A, B, C, b] = of(f, h, ht);

    % Solve system and recover flow.
    x = (A + alpha0*B + alpha1*C) \ b;
    v = reshape(x, n, t)';

    % Visualise flow.
    plotstreamlines(1, 'Input image with streamlines superimposed.', 'gray', f, v, h, ht, [t, n]);
    plotdata(2, 'Velocity.', 'default', v, h, ht);
    res = ofresidual(f, v, h, ht);
    plotdata(3, 'Residual.', 'default', res, h, ht);
    fw = computeoftransport(f, v, h, ht);
    plotdata(4, 'Transport.', 'gray', fw, h, ht);
    diff = abs(f - fw);
    plotdata(5, 'Absolute difference between image and transported image.', 'default', diff, h, ht);
    drawnow();
    
    if(saveplots)
        export_fig(1, fullfile(outputPath, name, alg, sprintf('%s-input.png', name)), '-png', '-q300', '-a1', '-transparent');
        export_fig(2, fullfile(outputPath, name, alg, sprintf('%s-velocity.png', name)), '-png', '-q300', '-a1', '-transparent');
        export_fig(3, fullfile(outputPath, name, alg, sprintf('%s-residual.png', name)), '-png', '-q300', '-a1', '-transparent');
        export_fig(4, fullfile(outputPath, name, alg, sprintf('%s-transport.png', name)), '-png', '-q300', '-a1', '-transparent');
        export_fig(5, fullfile(outputPath, name, alg, sprintf('%s-diff.png', name)), '-png', '-q300', '-a1', '-transparent');
    end
end