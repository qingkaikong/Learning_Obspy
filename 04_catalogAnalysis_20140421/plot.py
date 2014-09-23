#!/usr/bin/env python
"""
Illustrates an example for plotting beachballs and data points on line with 
specific color based on values with a labelled colorbar.
"""
# free to use, modify and distribute under any licence you want


# imports
import sys
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import obspy.imaging.beachball as beach
import matplotlib.collections as collections
import matplotlib.transforms as transforms
import matplotlib.colors as colors
import matplotlib.colorbar as colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable

# data
data_x = [2, 4, 6, 8, 10, 12, 14, 15]
data_y = [0.8, 0.7, 0.76, 0.5, 0.9, 0.4, 0.74, 0.6]
data_dc = [60, 70, 45, 58, 78, 90, 81, 74]
data_beachballs = [
[134, 82, 145],
[144, 52, 134],
[64, 182, 185],
[-40, 41, 105],
[54, 86, 193],
[-9, 85, 62],
[20, 41, 12],
[24, 71, 61]]

# sets figure's dimensions
fig_x = 30
fig_y = 20

# creates figure
fig, ax = plt.subplots(figsize=(fig_x,fig_y))

# plots (data_x, data_y) points on line
ax.plot(data_x, data_y, '--', linewidth=2, color = 'k')

# creates a grid on the (main) plot
ax.grid()

# sets labels
plt.xlabel('Depth (km)')
plt.ylabel('Correlation')
# sets font size
plt.rcParams.update({'font.size': 28})

# sets fixed y-axis range
plt.ylim((0,1))

# sets margins on the plot
plt.margins(0.05)

# sets color canvas for coloring beachball
cm = plt.cm.jet

# creates a colorbar at the right of main plot:
divider = make_axes_locatable(ax)
# makes room for colorbar
cax = divider.append_axes("right", size="2%", pad=1)
# set values to colorbar
norm = colors.Normalize(vmin=0, vmax=100)
# creates colorbar on specific axes, norm, canvas color and orientation
cb1 = colorbar.ColorbarBase(cax, norm=norm, cmap=cm, orientation='vertical')
# sets colorbar label
cb1.set_label('DC%')

# plotting beachballs on specific x-axis and y-axis with a color based on the data_dc values (normalized to 0-1) 
for i in xrange(len(data_x)):
	# sets color value
    	color = cm(data_dc[i]/100.0)
	# draws beachball
	b = beach.Beach([data_beachballs[i][0], data_beachballs[i][1], data_beachballs[i][2]], xy=(data_x[i],data_y[i]), width=150, linewidth=1, facecolor=color)

        # holds the aspect but fixes positioning:
        b.set_transform(transforms.IdentityTransform())
        # brings the all patches to the origin (0, 0).
        for p in b._paths:
        	p.vertices -= [data_x[i], data_y[i]]
        # uses the offset property of the collection to position the patches
        b.set_offsets((data_x[i], data_y[i]))
        b._transOffset = ax.transData

	# adds beachball to plot
        ax.add_collection(b)

# saves plot to file
plt.savefig('plot.png')



