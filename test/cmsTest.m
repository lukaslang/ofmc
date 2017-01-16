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
function tests = cmsTest
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

% Create empty image with three time steps and n pixels.
f = rand(3, n);

% Compute linear system.
[A, B, C, D, b] = cms(f, 1, 1);
verifyEqual(testCase, size(A), [2*3*n, 2*3*n]);
verifyEqual(testCase, size(B), [2*3*n, 2*3*n]);
verifyEqual(testCase, size(C), [2*3*n, 2*3*n]);
verifyEqual(testCase, size(D), [2*3*n, 2*3*n]);
verifyEqual(testCase, size(b), [2*3*n, 1]);

end