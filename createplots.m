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

% Create start date and time.
startdate = '2017-10-10-09-21-15';
name = 'artificialcut_small';
name = 'E8-PSB1PMT';
alg = 'cms';

% Define and create output folder.
outputPath = fullfile('results', startdate, name, alg);

% Load results.
load(sprintf('%s/%s-results.mat', outputPath, name));

% Plot source vs. image.
plotdata(1, 'k/(f+1e-5)', 'default', k./(f+1e-5), h, ht);
export_fig(1, fullfile(outputPath, sprintf('%s-k-vs-f.png', name)), '-png', '-q600', '-a1', '-transparent');

[vx, vt] = gradient(v, h, ht);

n = 5;
lambda = linspace(1e-3, 0.1, n);
chi = linspace(-2e-4, 1, n);

for p=1:n
    for q=1:n
        dx = gradient(lambda(p)^2 * vx + chi(q)*f, h, ht) - v;
        plotdata(2, sprintf('lambda=%.4f chi=%.4f', lambda(p), chi(q)), 'default', dx, h, ht);
        export_fig(2, fullfile(outputPath, sprintf('%s-ode-lambda-%.4f-chi-%.4f.png', name, lambda(p), chi(q))), '-png', '-q600', '-a1', '-transparent');
    end
end