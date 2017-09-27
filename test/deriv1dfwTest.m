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
function tests = deriv1dfwTest
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

L = [-1, 1, 0;
      0,-1, 1;
      0, 0, 0] ./ (2*h);

verifyEqual(testCase, full(deriv1dfw(n, t, h)), L);


n = 3;
t = 2;
h = 2;

L = [-1, 1, 0, 0, 0, 0;
      0,-1, 1, 0, 0, 0;
      0, 0, 0, 0, 0, 0;
      0, 0, 0,-1, 1, 0;
      0, 0, 0, 0,-1, 1;
      0, 0, 0, 0, 0, 0] ./ (2*h);

verifyEqual(testCase, full(deriv1dfw(n, t, h)), L);

n = 4;
t = 4;
h = 2;

L = [-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
      0,-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
      0, 0,-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
      0, 0, 0, 0,-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
      0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;
      0, 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0, 0, 0, 0, 0;
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
      0, 0, 0, 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0, 0, 0;
      0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0, 0;
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0;
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 1, 0, 0;
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 1, 0;
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 1;
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ./ (2*h);

Lres = full(deriv1dfw(n, t, h));
verifyEqual(testCase, Lres, L);

end

function calcDerivativeOfZeroMatrixTest(testCase)

n = 4;
t = 4;
h = 2;

Dx = deriv1dfw(n, t, h);
f = ones(t, n);

% Compute derivatives.
fx = reshape(Dx*img2vec(f), n, t)';
verifyEqual(testCase, fx, zeros(t, n));

end

function adjointTest(testCase)

n = 4;
t = 4;
h = 2;

Dx = deriv1dfw(n, t, h);
f = rand(t, n);
v = rand(t, n);

% Compute derivatives.
fx = Dx*img2vec(f);

% Compute divergence.
vx = Dx'*img2vec(v);

% Verify adjoint property.
verifyEqual(testCase, fx' * img2vec(v), img2vec(f)' * vx, 'AbsTol', 1e-15);

end