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
function fw = warpof(f, v, h, ht)
%WARPOF Warps an image by a given flow.
%
%   fw = WARPOF(f, v, h, ht) takes an image f, a displacement field v, and
%   parameters h and ht, and warps the image f along v.
%
%   f, v are matrices of size [t, n].
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
for k=2:t
    fw(k, :) = interp1(X, fw(k-1, :), X-ht*v(k-1, :), 'spline', 0);
end

end