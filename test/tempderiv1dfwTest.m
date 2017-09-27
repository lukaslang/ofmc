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
function tests = tempderiv1dTest
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
t = 3;
ht = 2;

L = [-1, 0, 0, 1, 0, 0, 0, 0, 0;
      0,-1, 0, 0, 1, 0, 0, 0, 0;
      0, 0,-1, 0, 0, 1, 0, 0, 0;
      0, 0, 0,-1, 0, 0, 1, 0, 0;
      0, 0, 0, 0,-1, 0, 0, 1, 0;
      0, 0, 0, 0, 0,-1, 0, 0, 1;
      0, 0, 0, 0, 0, 0, 0, 0, 0;
      0, 0, 0, 0, 0, 0, 0, 0, 0;
      0, 0, 0, 0, 0, 0, 0, 0, 0] ./ (2*ht);

verifyEqual(testCase, full(tempderiv1dfw(n, t, ht)), L);

end

function calcDerivativeOfZeroMatrixTest(testCase)

n = 4;
t = 4;
ht = 2;

Dx = tempderiv1dfw(n, t, ht);
f = ones(t, n);

% Compute derivatives.
fx = reshape(Dx*img2vec(f), n, t)';
verifyEqual(testCase, fx, zeros(t, n));

end

function adjointTest(testCase)

n = 4;
t = 4;
ht = 2;

Dt = tempderiv1dfw(n, t, ht);
f = rand(t, n);
v = rand(t, n);

% Compute derivatives.
fx = Dt*img2vec(f);

% Compute divergence.
vx = Dt'*img2vec(v);

% Verify adjoint property.
verifyEqual(testCase, fx' * img2vec(v), img2vec(f)' * vx, 'AbsTol', 1e-15);

end