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
function plotstreamlines(fh, ftitle, cmap, f, v, h, ht)
%PLOTSTREAMLINES Creates a plot of data with streamlines superimposed.
%
%   PLOTSTREAMLINES(fh, ftitle, f, v, h, ht) takes a figure handle fn, a
%   figure title ftitle, a colormap cmap, matrices f and v, and scaling 
%   parameters h and ht, and creates a plot.
%
%   fh is a figure handle.
%   ftitle is a string.
%   cmap is either a string (e.g. 'default' or 'gray') or a colormap.
%   f and v are matrices of size [m, n] where m is the number of time steps
%   and n the number of pixels.
%   fh is a figure handle.

% Get matrix size.
[t, n] = size(f);

% Create figure.
figure(fh);
imagesc(0:h:1, 0:ht:1, f);
set(gca, 'DataAspectRatio', [t, n, 1]);
colorbar;
colormap(cmap);
[X, Y] = meshgrid(0:h:1, 0:ht:1);
streamline(X, Y, ht*v, ht*ones(t, n), 0:2*h:1, ht*zeros(ceil(n/2), 1));
title(ftitle, 'FontName', 'Helvetica', 'FontSize', 14);
xlabel('Space', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Time', 'FontName', 'Helvetica', 'FontSize', 14);

end