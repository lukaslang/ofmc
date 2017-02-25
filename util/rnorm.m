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
function x = rnorm(x, epsilon)
%RNORM Computes the regularised norm.
%
%   x = RNORM(x) takes a matrix x and returns the regularised norm
%   
%   x = sqrt(abs(x).^2 + epsilon).
%
%   x is a matrix of size [m, n].

x = sqrt(abs(x).^2 + epsilon);

end