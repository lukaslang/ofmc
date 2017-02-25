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
function tests = divgrad1dTest
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

% Set scaling parameters.
h = 1/(n-1);

% Set denominator to be constant one.
d = ones(t, n);

% Check if equal to Laplacian.
L = divgrad1d(d, h);
verifyEqual(testCase, L, laplacian1d(n, t, h));

% Set denominator to be constant two.
d = 2*ones(t, n);

% Check if equal to 0.5*Laplacian.
L = divgrad1d(d, h);
verifyEqual(testCase, L, 0.5*laplacian1d(n, t, h));

end