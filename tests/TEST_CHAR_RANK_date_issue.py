import os
import sys
import time
import numpy
import pandas
import pathlib
import datetime
import functools
import matplotlib.pyplot as plt

# add source directory to path
sys.path.insert(0, '../src/FinToolsAP/')

import LaTeXBuilder
import LocalDatabase
import UtilityFunctions

# set printing options
import shutil
pandas.set_option('display.max_rows', None)
pandas.set_option('display.max_columns', None)
pandas.set_option('display.width', shutil.get_terminal_size()[0])
pandas.set_option('display.float_format', lambda x: '%.3f' % x)

# directory for loacl wrds database
LOCAL_DB = pathlib.Path('/home/andrewperry/Documents')

def main():
    
    start_date = '1980-01-01'
    end_date = '2023-12-31'
    
    # get stock level characteristics
    stock_db_path = pathlib.Path('/home/andrewperry/Dropbox/Characteristics_construction/Data/StockDB')
    STOCK_DB = LocalDatabase.LocalDatabase(stock_db_path)

    char = STOCK_DB.queryDB(
        STOCK_DB.DBP.CHAR_RANK2,
        start_date = start_date,
        end_date = end_date,
        all_vars = True, 
        cusip = ['25862055', '25862056', '25862054', '31614623'],
        return_type = 'polars'
    )
    


if __name__ == '__main__':
    main()