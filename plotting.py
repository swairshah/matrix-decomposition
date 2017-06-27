import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.lines as mlines
from matplotlib2tikz import save as tikz_save

def newline(p1, p2, c='orange'):
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if(p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

    l = mlines.Line2D([xmin,xmax], [ymin,ymax], c = c)
    ax.add_line(l)
    return l

