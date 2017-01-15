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
function tests = ofTest
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
f = zeros(3, n);

% Compute linear system.
[A, B, C, b] = cm(f, 1, 1);
verifyEqual(testCase, size(A), [3*n, 3*n]);
verifyEqual(testCase, size(B), [3*n, 3*n]);
verifyEqual(testCase, size(C), [3*n, 3*n]);
verifyEqual(testCase, size(b), [3*n, 1]);

end