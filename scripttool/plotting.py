"""
provides convenience plotting tools for scripttools
"""
# Copyright (C) 2011 Steffen Waldherr waldherr@ist.uni-stuttgart.de
# Time-stamp: <Last change 2015-04-08 09:39:16 by Steffen Waldherr>

import matplotlib
import os
if not "DISPLAY" in os.environ:
    matplotlib.use("Agg")
from matplotlib import pyplot

def make_ax(xlabel="",ylabel="",title="", figtype=None, figargs={}, axargs={}, **kwargs):
    """
    make a simple figure with one axis for plotting

    returns figure, axis
    
    options for figtype:
    None - default figure appearance
    'beamer' - configuration for inclusion in latex beamer presentations
    'small' - configuration for small figures
    'one-column' - configuration for a one-column figure in a two-column paper (IFAC style)

    'figargs' are keyword arguments for figure()
    'axargs' are keyword arguments for add_subplot()
    'kwargs' are keyword arguments for set_{xlabel, ylabel, title}, e.g. font info
    """
    fig = pyplot.figure(**figargs)
    if figtype == 'beamer':
        fig.set_figwidth(6.0)
        fig.set_figheight(4.0)
        fig.set_dpi(200)
    if figtype == 'small':
        fig.set_figwidth(3.0)
        fig.set_figheight(2.0)
        fig.set_dpi(300)
    if figtype == 'one-column':
        fig.set_figwidth(5.4)
        fig.set_figheight(3.6)
        fig.set_dpi(200)
    if figtype == 'small-square':
        (w, h) = matplotlib.figure.figaspect(1)
        fig.set_figwidth(w/2 / 0.75)
        fig.set_figheight(h/2 / (0.75 if len(title) else 0.8))
        fig.set_dpi(200)
    ax = fig.add_subplot(111, **axargs)
    fontsize = 12
    if figtype=='beamer':
        fontsize = 12
        if len(title) is 0:
            ax.set_position([0.15,0.15,0.8,0.8])
        else:
            ax.set_position([0.15,0.15,0.8,0.75])
    if figtype=='small' or figtype=='small-square':
        fontsize = 12
        if len(title) is 0:
            ax.set_position([0.2,0.15,0.75,0.8])
        else:
            ax.set_position([0.2,0.15,0.75,0.75])
    if figtype=='one-column':
        fontsize = 14
        if len(title) is 0:
            ax.set_position([0.15,0.15,0.8,0.8])
        else:
            ax.set_position([0.15,0.15,0.8,0.75])
    if figtype is None:
        if len(title) > 0:
            ax.set_position([0.15,0.15,0.8,0.75])
    if "fontsize" in kwargs:
        ax.set_xlabel(xlabel, **kwargs)
        ax.set_ylabel(ylabel, **kwargs)
        ax.set_title(title, **kwargs)
    else:
        ax.set_xlabel(xlabel, fontsize=fontsize, **kwargs)
        ax.set_ylabel(ylabel, fontsize=fontsize, **kwargs)
        ax.set_title(title, fontsize=fontsize, **kwargs)
    return fig, ax

def show():
    if "DISPLAY" in os.environ:
        pyplot.show()
