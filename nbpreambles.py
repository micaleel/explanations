import os
import pandas as pd
from IPython.display import display, set_matplotlib_formats
import logging
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from pprint import pprint
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches

LOG_FORMAT = '[%(asctime)s] %(levelname)s: %(message)s\n  %(module)s:%(funcName)s:%(lineno)s'
logging.basicConfig(format=LOG_FORMAT, datefmt='%Y.%m.%d %I:%M:%S')
logging.getLogger().setLevel(logging.INFO)
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
logging.getLogger('elasticsearch').setLevel(logging.WARNING)
np.set_printoptions(precision=3)
pd.set_option('precision', 3)

set_matplotlib_formats('pdf', 'png')

plt.rcParams['savefig.dpi'] = 300
plt.rcParams['image.interpolation'] = "none"
plt.rcParams['savefig.bbox'] = "tight"

plt.rcParams["font.family"] = 'Merriweather Sans, Ubuntu'

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


def to_percent(y, position=None):
    """Formats tick labels (of a given axis) to percentages

    - The argument should be a normalized value such that -1 <= y <= 1
    - Ignore the passed in position; it scales the default tick locations.
    """
    s = str(100 * y)

    # The percent symbol needs escaping in latex
    if mpl.rcParams['text.usetex']:
        # return '${}$'.format(int(float(s)))
        return '${}\%$'.format(int(float(s)))
    else:
        return '{}%'.format(int(float(s)))


def inches_to_points(inches):
    inches_per_pt = 1.0 / 72.27  # to convert pt to inches
    return inches / inches_per_pt


def points_to_inches(points):
    inches_per_pt = 1.0 / 72.27  # to convert pt to inches
    return points * inches_per_pt


def latexify(column_width_pt=243.91125, text_width_pt=505.89, scale=2, fontsize_pt=11, usetex=True):
    import matplotlib.pyplot as plt

    from IPython.display import set_matplotlib_formats
    set_matplotlib_formats('pdf', 'png')

    # sorted([f.name for f in mpl.matplotlib.font_manager.fontManager.ttflist])

    fig_width_pt = column_width_pt
    inches_per_pt = 1.0 / 72.27  # to convert pt to inches
    golden_mean = 0.61803398875  # (math.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
    fig_proportion = golden_mean
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = fig_width * fig_proportion  # height in inches
    fig_size = [scale * fig_width, scale * fig_height]

    # Legend
    plt.rcParams['legend.fontsize'] = 14  # in pts (e.g. "x-small")

    # Lines
    plt.rcParams['lines.markersize'] = 8
    plt.rcParams['lines.linewidth'] = 2.0

    # Ticks
    # plt.rcParams['xtick.labelsize'] = 'x-small'
    # plt.rcParams['ytick.labelsize'] = 'x-small'
    # plt.rcParams['xtick.major.pad'] = 1
    # plt.rcParams['ytick.major.pad'] = 1

    # Axes
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 18
    # plt.rcParams['axes.labelpad'] = 0

    # LaTeX
    plt.rcParams['text.usetex'] = usetex
    plt.rcParams['text.latex.unicode'] = True
    plt.rcParams['text.latex.preview'] = False

    # use utf8 fonts becasue your computer can handle it :)
    # plots will be generated using this preamble
    plt.rcParams['text.latex.preamble'] = [
        r'\usepackage[utf8x]{inputenc}',
        r'\usepackage[T1]{fontenc}',
        r'\usepackage{amssymb}',
        r'\usepackage{amsmath}',
        r'\usepackage{wasysym}',
        r'\usepackage{stmaryrd}',
        r'\usepackage{subdepth}',
        r'\usepackage{type1cm}'
    ]

    # Fonts
    plt.rcParams['font.size'] = fontsize_pt  # font size in pts (good size 16)
    plt.rcParams['font.family'] = 'sans-serif'  # , 'Merriweather Sans'  # ['DejaVu Sans Display', "serif"]
    plt.rcParams['font.serif'] = ['Merriweather',
                                  'cm']  # blank entries should cause plots to inherit fonts from the document
    plt.rcParams['font.sans-serif'] = 'Merriweather Sans'
    plt.rcParams['font.monospace'] = 'Operator Mono'

    # Figure
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 75
    plt.rcParams['savefig.pad_inches'] = 0.01
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['figure.autolayout'] = False
    plt.rcParams['figure.figsize'] = fig_size
    plt.rcParams['image.interpolation'] = 'none'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
