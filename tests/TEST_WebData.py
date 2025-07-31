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
    
    WD = WebData.WebData('andrewperry')
    
    
    df = WD.getData(tickers = ['GE'], 
                    fields = ['dp'], 
                    start_date = '2000-01-01', 
                    end_date = '2010-01-01', 
                    freq = 'D')
    df = df.set_index('date')
    ax = df.dp.plot()
    ax.set_xlabel('Date')
    ax.set_ylabel('Dividend Yield')
    ax.set_title('General Electric')
    plt.show()
    raise
    
    df = WD.getData(tickers = ['AAPL', 'MSFT', 'F', 'GE'], 
                    fields = ['ep', 'eps', 'ret', 'vwretd', 'sprtrn', 'shrout'], 
                    freq = 'D')
    
    df = df.set_index('date')
    
    df[df.ticker == 'GE'].shrout.plot()
    
    f, a = plt.subplots(2, 2, figsize=(20, 8), tight_layout=True)
    r = 0
    c = 0
    for i, tic in enumerate(['AAPL', 'MSFT', 'F', 'GE']):
        c = i % 2
        if i % 2 == 0 and i != 0:
            r += 1
            
        axL = a[r, c]
        axR = a[r, c].twinx()
        
        df[df.ticker == tic].ep.plot(ax=axL)
        df[df.ticker == tic].eps.plot(ax=axR, color = Colors.YELLOW)
            
        axL.set_title(tic)
        axL.set_ylabel('Earnings to Price')
        axR.set_ylabel('Earnings to Shares')
        axL.set_xlabel('Date')
        axL.legend(['Earnings to Price'])
        axR.legend(['Earnings to Shares'])
        
    plt.show()
        
    
    



if __name__ == '__main__':
    main()