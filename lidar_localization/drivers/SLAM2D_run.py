# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 20:48:05 2020

@author: Nagnanamus
"""

from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import Polygon, LineString
from shapely.geometry import MultiLineString
import matlab.engine
eng = matlab.engine.start_matlab()

np.random.seed(1)

# %%
plt.close('all')

data = np.load('map1.npz')
truemap = data['truemap']

# robd=np.array([[0,1],[1,-1],[-1,-1],[0,1]])
robd = np.array([[2, 0], [-0.5, 0.5], [-0.5, -0.5], [2, 0]])

robrng = 300

# state is x,y,th
x0 = np.array([10, 10, np.pi / 2])


def plotrobot(x, ax):
    th = x[2]
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    ss = np.zeros(robd.shape)
    for i in range(len(robd)):
        ss[i, :] = np.matmul(R, robd[i, :]) + x[0:2]
    ax.plot(ss[:, 0], ss[:, 1], 'k')


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim([-10, 100])
ax.set_ylim([-10, 100])

ax.plot(truemap[:, 0], truemap[:, 1], 'bo-')
plotrobot(x0, ax)
plt.show()


def recordmeas(truemap, x0, robrng):
    measbeam = np.array([x0[0:2], (500 * np.cos(x0[2]), 500 * np.sin(x0[2]))])
    poly1 = eng.polyshape(matlab.double(
        truemap[:, 0].tolist()), matlab.double(truemap[:, 1].tolist()))
    X = eng.intersect(poly1, matlab.double(measbeam.tolist()))
    #X = eng.polyxpoly(measbeam[:,0].tolist(),measbeam[:,1].tolist(),truemap[:,0].tolist(),truemap[:,1].tolist());
    #X = eng.intersect(matlab.double([[-1,1,1,-1,-1],[-1,-1,1,1,-1]]),matlab.double([[0,10],[0,10]]) )
    X = np.array(X)

    xx = X[1, :]
    if np.linalg.norm(xx - x0[:2]) <= robrng:
        return xx + 2 * np.random.randn(2)
    else:
        return None


X = []
for th in np.linspace(0, 2 * np.pi, 20):
    x = x0.copy()
    x[2] = th
    X.append(recordmeas(truemap, x, robrng))

X = np.array(X)
X = np.vstack((X, X[0, :], X[0, :], X[0, :], X[0, :], X[0, :], X[0, :]))
# ax.plot([x0[0],X[0]],[x0[1],X[1]],'k--')
ax.plot(X[:, 0], X[:, 1], 'k*')
plt.show()

d = np.sqrt(np.sum(pow(X[1:, :] - X[0:-1, :], 2), axis=1))
dd = np.hstack((0, d))
T = np.cumsum(dd)
# T=np.arange(len(X[:,0]))
t = np.linspace(0, T[-1], 100)

fig2 = plt.figure()
ax2x = fig2.add_subplot(211)
ax2y = fig2.add_subplot(212)
ax2x.plot(T, X[:, 0], 'bo-')
ax2y.plot(T, X[:, 1], 'bo-')
plt.show()


kernel = C(10, (1e-5, 1e5)) * RBF(50, (1e-5, 1e5))
gpx = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=1)
gpy = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=1)
gpf = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=1)
# Fit to data using Maximum Likelihood Estimation of the parameters
gpx.fit(T.reshape((-1, 1)), X[:, 0])
gpy.fit(T.reshape((-1, 1)), X[:, 1])

gpf.fit(T.reshape((-1, 1)), X)

# Make the prediction on the meshed x-axis (ask for MSE as well)
xt_pred, sigmax = gpx.predict(t.reshape((-1, 1)), return_std=True)
yt_pred, sigmay = gpy.predict(t.reshape((-1, 1)), return_std=True)
xyt_pred, sigmaxy = gpf.predict(t.reshape((-1, 1)), return_std=True)

ax2x.plot(
    t,
    xt_pred,
    'r',
    t,
    xt_pred +
    sigmax,
    'b--',
    t,
    xt_pred -
    sigmax,
    'b--')
ax2y.plot(
    t,
    yt_pred,
    'r',
    t,
    yt_pred +
    sigmay,
    'b--',
    t,
    yt_pred -
    sigmay,
    'b--')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim([-10, 100])
ax.set_ylim([-10, 100])

ax.plot(truemap[:, 0], truemap[:, 1], 'bo-')
plotrobot(x0, ax)
ax.plot(X[:, 0], X[:, 1], 'k*')
ax.plot(xt_pred, yt_pred, 'r')
ax.plot(xt_pred + sigmax, yt_pred + sigmay, 'r.')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim([-10, 100])
ax.set_ylim([-10, 100])

ax.plot(truemap[:, 0], truemap[:, 1], 'bo-')
plotrobot(x0, ax)
ax.plot(X[:, 0], X[:, 1], 'k*')
ax.plot(xyt_pred[:, 0], xyt_pred[:, 1], 'r')
plt.show()
