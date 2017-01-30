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
function fw = warpcms(f, v, k, h, ht)
%WARPCMS Warps an image by a given flow and source/sink.
%
%   fw = WARPCMS(f, v, k, h, ht) takes an image f, a displacement field v, 
%   a source/sink k, and parameters h and ht, and warps the image f along v.
%
%   f, v, and k are matrices of size [t, n].
%   h, ht are a scalars.
%   fw is a matrix of size [t, n].

% Create output image.
[t, n] = size(f);
fw = zeros(t, n);

% Copy first time instant.
fw(1, :) = f(1, :);

% Create grid.
X = 0:h:1;

% Interpolate each time instant.
for j=2:t
    fw(j, :) = interp1(X, fw(j-1, :), X-ht*v(j-1, :), 'spline', 0) + interp1(X, h*k(j-1, :), X-ht*v(j-1, :), 'spline', 0);
end

end