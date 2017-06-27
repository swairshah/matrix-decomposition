import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg') 
from matplotlib import pyplot as plt
import pylab
import argparse
import os
import subprocess
from matplotlib2tikz import save as tikz_save
from plotting import *

X = np.genfromtxt('data/foo.data')
B = X.T.dot(X)
_, _, V = np.linalg.svd(B)
plt.scatter(X[:,0],X[:,1], alpha = 0.5)
newline([0,0],V[:,0])
#plt.show()
tikz_save('original.tex')
plt.clf()

plt.scatter(X[:,0],X[:,1],alpha = 0.5)
plt.scatter(X[50:,0],X[50:,1],alpha = 0.6, marker='x', c='red')
tikz_save('outliers.tex')
plt.clf()

S = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
Y = X[:50,:]
mean = np.mean(Y, axis = 0)
Z = Y - mean
C = Z.T.dot(Z)
_, _, U = np.linalg.svd(C)
plt.scatter(Z[:,0],Z[:,1], alpha = 0.5)
newline([0,0],U[:,0])
tikz_save('final.tex')
