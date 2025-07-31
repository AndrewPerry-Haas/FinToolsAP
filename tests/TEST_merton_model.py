import os
import sys
import time
import numpy
import pandas
import pathlib
import datetime
import functools

# add source directory to path
sys.path.insert(0, '../src/FinToolsAP/')

import MertonModel
import LaTeXBuilder
import LocalDatabase
import PortfolioSorts
import UtilityFunctions

# set printing options
import shutil
pandas.set_option('display.max_rows', None)
pandas.set_option('display.max_columns', None)
pandas.set_option('display.width', shutil.get_terminal_size()[0])
pandas.set_option('display.float_format', lambda x: '%.3f' % x)

# standard plotting options
import matplotlib
import matplotlib.pyplot as plt

# Define a new list of colors
COLORS = ['#002676', '#FDB515', '#C0362C', '#FFFFFF', '#010133',
          '#FC9313', '#00553A', '#770747', '#431170', '#004AAE',
          '#FFC31B', '#018943', '#E7115E', '#8236C7', '#9FD1FF',
          '#FFE88D', '#B3E59A', '#FFCFE5', '#D9CEFF', '#000000',
          '#808080', '#F2F2F2', '#C09748']

plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['legend.frameon'] = 'False'
matplotlib.rcParams['text.usetex'] = True
plt.rcParams['axes.facecolor'] = 'lightgrey'
plt.rcParams['patch.force_edgecolor'] = True
plt.rcParams['axes.prop_cycle'] = plt.cycler(color = COLORS)

# directory for loacl wrds database
LOCAL_DB = pathlib.Path('/home/andrewperry/Documents')

def main():
    
    mm = MertonModel.MertonModel(1, 0.2, 0.01, 1, 0.5)
    
    res = mm.simulate(0.02, n_simulations = 2)
    
    f, a = plt.subplots()
    a.plot(res[:, 0])
    a.plot(res[:, 1])
    
    f.savefig('/home/andrewperry/Nextcloud/FinToolsAP/tests/figures/merton_sim.png')
    
    
    
    
    
    #print(res)
    
    
    


if __name__ == '__main__':
    main()