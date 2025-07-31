import os
import sys
import time
import shutil
import pathlib
import datetime
import functools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas.tseries.offsets

# add source directory to path
sys.path.insert(0, '../src/FinToolsAP/')

import LocalDatabase
import PortfolioSorts as PS
import LaTeXBuilder


# set printing options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', shutil.get_terminal_size()[0])
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# directory for loacl wrds database
LOCAL_WRDS_DB = pathlib.Path('/home/andrewperry/Documents')

desktop = pathlib.Path('/home/andrewperry/Desktop')

tercile_sorts = pathlib.Path('/home/andrewperry/Dropbox/Characteristics_construction/Characteristic_Con/tercile_sorts')
quartile_sorts = pathlib.Path('/home/andrewperry/Dropbox/Characteristics_construction/Characteristic_Con/quartile_sorts')
qunitle_sorts = pathlib.Path('/home/andrewperry/Dropbox/Characteristics_construction/Characteristic_Con/quintile_sorts')
decile_sorts = pathlib.Path('/home/andrewperry/Dropbox/Characteristics_construction/Characteristic_Con/decile_sorts')

def main():

    DB = LocalDatabase.LocalDatabase(LOCAL_WRDS_DB, database_name = 'WRDS')
    df = DB.queryDB(DB.DBP.CHAR, all_vars = True)

    save_dir = desktop / 'char_hist'
    os.makedirs(save_dir, exist_ok = True)
    for var in DB.DBP.CHAR.CHARACTERISTICS:
        fig, axs = plt.subplots(nrows = 1, ncols = 1)
        fig.suptitle(f'{var}')
        df[var].plot.kde(ax = axs)
        fig.savefig(save_dir / f'{var}.pdf')
        plt.close(fig)

    LaTeXBuilder.graph_document(save_dir)
    
    # tercile sorts
    for var in DB.DBP.CHAR.CHARACTERISTICS:
        sorts_df = PS.sort_portfolios(dfin = df, 
                                      sorting_funcs = {'me': PS.sort_tercile, var: PS.sort_tercile},
                                      char_bkpts = {'me': [0.33, 0.66], var: [0.33, 0.66]},
                                      drop_na = False
                                    )
        filename = f'{var}.csv'
        csv_dir = tercile_sorts / 'csv/'
        os.makedirs(csv_dir, exist_ok = True)
        sorts_df.to_csv(csv_dir / filename, index = False)
        
        sorts_df = sorts_df.set_index('date')
        plot_dir = tercile_sorts / f'plots_{var}/'
        os.makedirs(plot_dir, exist_ok = True)
        for col in sorts_df.columns:
            fig, axs = plt.subplots(nrows = 1, ncols = 1)
            fig.suptitle(f'{col}')
            sorts_df[col].plot(ax = axs)
            fig.savefig(plot_dir / f'{col}.pdf')
            plt.close(fig)
        
        LaTeXBuilder.graph_document(plot_dir)
        
    # quartile sorts
    for var in DB.DBP.CHAR.CHARACTERISTICS:
        sorts_df = PS.sort_portfolios(dfin = df, 
                                      sorting_funcs = {'me': PS.sort_quartile, var: PS.sort_quartile},
                                      char_bkpts = {'me': [0.25, 0.5, 0.75], var: [0.25, 0.5, 0.75]},
                                      drop_na = False
                                    )
        filename = f'{var}.csv'
        csv_dir = quartile_sorts / 'csv/'
        os.makedirs(csv_dir, exist_ok = True)
        sorts_df.to_csv(csv_dir / filename, index = False)
        
        sorts_df = sorts_df.set_index('date')
        plot_dir = quartile_sorts / f'plots_{var}/'
        os.makedirs(plot_dir, exist_ok = True)
        for col in sorts_df.columns:
            fig, axs = plt.subplots(nrows = 1, ncols = 1)
            fig.suptitle(f'{col}')
            sorts_df[col].plot(ax = axs)
            fig.savefig(plot_dir / f'{col}.pdf')
            plt.close(fig)
        
        LaTeXBuilder.graph_document(plot_dir)
        
    # quintile sorts
    for var in DB.DBP.CHAR.CHARACTERISTICS:
        sorts_df = PS.sort_portfolios(dfin = df, 
                                      sorting_funcs = {'me': PS.sort_quintile, var: PS.sort_quintile},
                                      char_bkpts = {'me': [0.2, 0.4, 0.6, 0.8], var: [0.2, 0.4, 0.6, 0.8]},
                                      drop_na = False
                                    )
        filename = f'{var}.csv'
        csv_dir = qunitle_sorts / 'csv/'
        os.makedirs(csv_dir, exist_ok = True)
        sorts_df.to_csv(csv_dir / filename, index = False)
        
        sorts_df = sorts_df.set_index('date')
        plot_dir = qunitle_sorts / f'plots_{var}/'
        os.makedirs(plot_dir, exist_ok = True)
        for col in sorts_df.columns:
            fig, axs = plt.subplots(nrows = 1, ncols = 1)
            fig.suptitle(f'{col}')
            sorts_df[col].plot(ax = axs)
            fig.savefig(plot_dir / f'{col}.pdf')
            plt.close(fig)
        
        LaTeXBuilder.graph_document(plot_dir)
        
    # decile sorts
    for var in DB.DBP.CHAR.CHARACTERISTICS:
        sorts_df = PS.sort_portfolios(dfin = df, 
                                      sorting_funcs = {var: PS.sort_decile},
                                      char_bkpts = {var: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
                                      drop_na = False
                                    )
        filename = f'{var}.csv'
        csv_dir = decile_sorts / 'csv/'
        os.makedirs(csv_dir, exist_ok = True)
        sorts_df.to_csv(csv_dir / filename, index = False)
        
        sorts_df = sorts_df.set_index('date')
        plot_dir = decile_sorts / f'plots_{var}/'
        os.makedirs(plot_dir, exist_ok = True)
        for col in sorts_df.columns:
            fig, axs = plt.subplots(nrows = 1, ncols = 1)
            fig.suptitle(f'{col}')
            sorts_df[col].plot(ax = axs)
            fig.savefig(plot_dir / f'{col}.pdf')
            plt.close(fig)
        
        LaTeXBuilder.graph_document(plot_dir)
        

if __name__ == '__main__':
    main()
