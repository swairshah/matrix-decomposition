#!/usr/bin/python
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

def generate():
    m = 50
    f = lambda x : 9*x + 3 + 2*np.random.normal(0,1,1)
    x = np.random.normal(0, 10, m)
    y = np.array([f(i) for i in x])
    x = x.reshape((m,))
    y = y.reshape((m,))

    # outliers:
    Outliers2 = np.random.normal(loc=(25,25), scale = (5,5), size = (10,2))
    x = np.hstack((x, Outliers2[:,0]))
    y = np.hstack((y, Outliers2[:,1]))

    #plt.scatter(x, y, alpha = 0.4)
    #pylab.show()
    #tikz_save('test.tex')
    return x, y

def save(data, name):
    cwd = os.getcwd()
    os.chdir('./data')
    #if os.path.isfile(name+'.data'):
    #    print("FILE ALREADY EXISTS")
    #    return False
    try:
        d = pd.DataFrame(data, columns = ['x','y'])
        d.to_csv(name+'.data', sep=' ', header=False, index=False)
        os.chdir(cwd)
        return True
    finally:
        os.chdir(cwd)
        return False

def dataset_import(name):
    cwd = os.getcwd()
    os.chdir('./data')
    if os.path.isfile(name+'.transpose'):
        return True
    else:
        os.chdir('./ImportDataFile')
        infile = "../"+name+'.data'
        outfile = "../"+name+'.transpose'
        command = ["java", "ImportDataset", 
                   "-input", infile,
                   "-output", outfile]
                   #"-transpose", "yes"]
        try:
            p = subprocess.Popen(' '.join(command), shell = True, stdout = subprocess.PIPE)
            print(' '.join(command))
            p.wait()
            output, err = p.communicate()
            os.chdir(cwd)
            return True
        finally:
            os.chdir(cwd)
            return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str, help="dataset name")
    args = parser.parse_args()
    name = args.name

    data = generate()
    #save(data, name)
