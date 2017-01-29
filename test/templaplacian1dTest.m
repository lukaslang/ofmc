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
function tests = templaplacian1dTest
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    cd('../');
end

function teardownOnce(testCase)
    cd('test');
end

function resultTest(testCase)

n = 1;
t = 3;
ht = 2;

L = [-2, 2, 0;
      1,-2, 1;
      0, 2,-2] ./ (ht^2);

verifyEqual(testCase, full(templaplacian1d(n, t, ht)), L);


n = 4;
t = 4;
ht = 2;

L = [-2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
      0,-2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
      0, 0,-2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0;
      0, 0, 0,-2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0;
      1, 0, 0, 0,-2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0;
      0, 1, 0, 0, 0,-2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0;
      0, 0, 1, 0, 0, 0,-2, 0, 0, 0, 1, 0, 0, 0, 0, 0;
      0, 0, 0, 1, 0, 0, 0,-2, 0, 0, 0, 1, 0, 0, 0, 0;
      0, 0, 0, 0, 1, 0, 0, 0,-2, 0, 0, 0, 1, 0, 0, 0;
      0, 0, 0, 0, 0, 1, 0, 0, 0,-2, 0, 0, 0, 1, 0, 0;
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0,-2, 0, 0, 0, 1, 0;
      0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,-2, 0, 0, 0, 1;
      0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,-2, 0, 0, 0;
      0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,-2, 0, 0;
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,-2, 0;
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,-2] ./ (ht^2);

Lres = full(templaplacian1d(n, t, ht));
verifyEqual(testCase, Lres, L);

end