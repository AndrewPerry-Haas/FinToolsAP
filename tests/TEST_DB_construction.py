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
    
    DB = LocalDatabase.LocalDatabase('/home/andrewperry/Documents/TESTDB/')
    
    
    df = DB.queryDB(DB.DBP.COMPA_FUNDQ, vars = ['gvkey', 'datadate', 'conm', 'ltq', 'atq'])
    
    print(df.shape)
    print(df.head(20))
    print(df.dtypes)
    
    
    
    
    print(DB.getFFIndustryClassification(2040, level = 49, info = 'desc'))


if __name__ == '__main__':
    main()