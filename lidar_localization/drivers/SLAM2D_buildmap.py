# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 19:15:23 2020

@author: Nagnanamus
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt


class LineBuilder:
    def __init__(self, line):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        print('click', event)
        if event.inaxes != self.line.axes:
            return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim([0, 100])
ax.set_ylim([0, 100])
ax.set_title('click to build line segments')
line, = ax.plot([0], [0])  # empty line
linebuilder = LineBuilder(line)

plt.show()
truemap = np.array([linebuilder.xs, linebuilder.ys]).T
truemap[-1, :] = truemap[0, :]
truemap

np.savez('map1.npz', truemap=truemap)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim([-10, 100])
ax.set_ylim([-10, 100])
ax.plot(truemap[:, 0], truemap[:, 1], 'bo-')
plt.show()
