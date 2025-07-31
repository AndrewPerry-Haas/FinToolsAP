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

import WebData
import LaTeXBuilder
import FactorModels
import LocalDatabase
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

plt.rcParams['axes.grid'] = True
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['legend.frameon'] = 'True'
matplotlib.rcParams['text.usetex'] = True
plt.rcParams['axes.facecolor'] = '#f0f0f0'
plt.rcParams['patch.force_edgecolor'] = True
plt.rcParams['axes.prop_cycle'] = plt.cycler(color = COLORS)

class Colors:
     BLUE = '#002676'
     YELLOW = '#FDB515'
     RED = '#C0362C'
     WHITE = '#FFFFFF'
     BLACK = '#000000'


def main():
    
    T = 600
    N = 50
    K = 10
    
    numpy.random.seed(0)
    
    factors = pandas.DataFrame(
        numpy.random.randn(T, K),
        columns=[f"Factor{i+1}" for i in range(K)]
    )
    test_assets = pandas.DataFrame(
        numpy.random.randn(T, N),
        columns=[f"Asset{i+1}" for i in range(N)]
    )
    riskfree = numpy.random.randn(T, 1)
   

    res = FactorModels.FamaMacBeth(test_assets, factors, riskfree, bandwidth = 1)
    print(res.riskPremia())
    



if __name__ == '__main__':
    main()