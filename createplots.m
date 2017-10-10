clear;
close all;
clc;

% Create start date and time.
startdate = '2017-10-10-09-21-15';
name = 'artificialcut_small';
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