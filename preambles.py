import os
import logging
from glob import glob
from pprint import pprint
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib as mpl
from IPython.display import display, set_matplotlib_formats
import matplotlib.pyplot as plt

LOG_FORMAT = '[%(asctime)s] %(levelname)s: %(message)s \n\t%(module)s:%(funcName)s:%(lineno)s'
logging.basicConfig(format=LOG_FORMAT)
logging.getLogger().setLevel(logging.INFO)
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
logging.getLogger('elasticsearch').setLevel(logging.WARNING)
# logging.getLogger('elasticsearch').setLevel(logging.DEBUG)


set_matplotlib_formats('pdf', 'png')
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = "tight"
plt.rcParams['image.cmap'] = 'Set3'
plt.rcParams['image.interpolation'] = "none"
np.set_printoptions(precision=3)
pd.set_option('precision', 3)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
#plt.rcParams["font.family"] = 'Merriweather Sans, Ubuntu'
mpl.rcParams['xtick.labelsize'] = 'small' 
mpl.rcParams['ytick.labelsize'] = 'small' 

log = logging.getLogger()


