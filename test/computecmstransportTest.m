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
function tests = computecmstransportTest
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    cd('../');
end

function teardownOnce(testCase)
    cd('test');
end

function resultTest(testCase)

n = 10;
t = 5;
h = 1;
ht = 1;

% Create empty image with t time steps and n pixels.
f = peaks(n);
f = repmat(f(1, :), t, 1);

% Create velocity field.
v = zeros(t, n);

% Create source.
k = zeros(t, n);

% Compute and chekc transported image.
fw = computecmstransport(f, v, k, h, ht);
verifyEqual(testCase, fw, repmat(f(1, :), t, 1), 'AbsTol', 1e-15);

end