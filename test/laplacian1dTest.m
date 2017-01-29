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
function tests = laplacian1dTest
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    cd('../');
end

function teardownOnce(testCase)
    cd('test');
end

function resultTest(testCase)

n = 3;
t = 1;
h = 1;

L = [-2, 2, 0;
      1,-2, 1;
      0, 2,-2] ./ h^2;

verifyEqual(testCase, full(laplacian1d(n, t, h)), L);

n = 3;
t = 2;
h = 2;

L = [-2, 2, 0, 0, 0, 0;
      1,-2, 1, 0, 0, 0;
      0, 2,-2, 0, 0, 0;
      0, 0, 0,-2, 2, 0;
      0, 0, 0, 1,-2, 1;
      0, 0, 0, 0, 2,-2] ./ h^2;

verifyEqual(testCase, full(laplacian1d(n, t, h)), L);

n = 4;
t = 4;
h = 2;

L = [-2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
      1,-2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
      0, 1,-2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
      0, 0, 2,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
      0, 0, 0, 0,-2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
      0, 0, 0, 0, 1,-2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;
      0, 0, 0, 0, 0, 1,-2, 1, 0, 0, 0, 0, 0, 0, 0, 0;
      0, 0, 0, 0, 0, 0, 2,-2, 0, 0, 0, 0, 0, 0, 0, 0;
      0, 0, 0, 0, 0, 0, 0, 0,-2, 2, 0, 0, 0, 0, 0, 0;
      0, 0, 0, 0, 0, 0, 0, 0, 1,-2, 1, 0, 0, 0, 0, 0;
      0, 0, 0, 0, 0, 0, 0, 0, 0, 1,-2, 1, 0, 0, 0, 0;
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,-2, 0, 0, 0, 0;
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 2, 0, 0;
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,-2, 1, 0;
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,-2, 1;
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,-2] ./ (h^2);

Lres = full(laplacian1d(n, t, h));
verifyEqual(testCase, Lres, L);

end