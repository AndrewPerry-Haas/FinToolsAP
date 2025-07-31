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
    
    DB = LocalDatabase.LocalDatabase(
        '/home/andrewperry/Dropbox/Characteristics_construction/Data/MutualFundDB'
    )


    df = DB.queryDB(DB.DBP.MF_CHAR, vars = ['date', 'me_ew'], wficn = [100001, 100784], end_date = '2000-01-01')

    #print(df.shape)
    #print(df)
    #print(df.dtypes)

    df = UtilityFunctions.group_quantile(
        df = df,
        qtiles = 4,
        gr = 'date',
        vr = ['me_ew'],
        set_index = 'wficn'
    )

    print(df)


if __name__ == '__main__':
    main()