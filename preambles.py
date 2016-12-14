import os
import logging
from glob import glob
from pprint import pprint
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib as mpl
from IPython.display import display
import matplotlib.pyplot as plt

LOG_FORMAT = '[%(asctime)s] %(levelname)s: %(message)s \n\t%(module)s:%(funcName)s:%(lineno)s'
logging.basicConfig(format=LOG_FORMAT)
logging.getLogger().setLevel(logging.INFO)
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
logging.getLogger('elasticsearch').setLevel(logging.WARNING)
# logging.getLogger('elasticsearch').setLevel(logging.DEBUG)

log = logging.getLogger()

def get_config_paths(directory):
    """Gets a list of all configuration file paths in a given directory"""
    config_paths = []
    for config_path in glob('{}/*.yml'.format(directory)):
        config_paths.append(os.path.realpath(config_path))
    log.info('Loaded %d configuration paths' % len(config_paths))
    return config_paths
