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
function v = img2vec(f)
%IMG2VEC Takes a 2D image and creates a vector.
%
%   v = IMG2VEC(f) takes an image f and creates a vector.
%
%   f is a matrix of size [m, n].
%   v is a vector of length m*n.
%
%   Note that elements in v are arranged row by row.

f = f';
v = f(:);

end