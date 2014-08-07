"""
provides convenience plotting tools for scripttools
"""
# Copyright (C) 2011 Steffen Waldherr waldherr@ist.uni-stuttgart.de
# Time-stamp: <Last change 2012-10-10 10:07:00 by Steffen Waldherr>

import matplotlib
import os
if not "DISPLAY" in os.environ:
    matplotlib.use("Agg")
from matplotlib import pyplot

def make_ax(xlabel="",ylabel="",title="", figtype=None, figargs={}, axargs={}):
    """
    make a simple figure with one axis for plotting

    returns figure, axis
    
    options for figtype:
    None - default figure appearance
    'beamer' - configuration for inclusion in latex beamer presentations
    'small' - configuration for small figures

    'figargs' are keyword arguments for figure()
    'axargs' are keyword arguments for add_subplot()
    """
    fig = pyplot.figure(**figargs)
    ax = fig.add_subplot(111, **axargs)
    if figtype=='beamer':
        if len(title) is 0:
            ax.set_position([0.15,0.15,0.8,0.8])
        else:
            ax.set_position([0.15,0.15,0.8,0.75])
    if figtype=='small':
        if len(title) is 0:
            ax.set_position([0.2,0.2,0.75,0.75])
        else:
            ax.set_position([0.2,0.2,0.75,0.7])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if figtype == 'beamer':
        fig.set_figwidth(6.0)
        fig.set_figheight(4.0)
        fig.set_dpi(200)
    if figtype == 'small':
        fig.set_figwidth(3.0)
        fig.set_figheight(2.0)
        fig.set_dpi(300)
    return fig, ax

def show():
    if "DISPLAY" in os.environ:
        pyplot.show()
