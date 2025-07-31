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
import PortfolioSorts
import UtilityFunctions

# set printing options
import shutil
pandas.set_option('display.max_rows', None)
pandas.set_option('display.max_columns', None)
pandas.set_option('display.width', shutil.get_terminal_size()[0])
pandas.set_option('display.float_format', lambda x: '%.3f' % x)

# directory for loacl wrds database
LOCAL_DB = pathlib.Path('/home/andrewperry/Documents')

import tqdm
import seaborn
import connectorx
import sqlalchemy

def main():

    query = """SELECT * FROM COMP"""
    con = LOCAL_DB / 'LCLDB/LCLDB.db'

    SQL_ENGINE = sqlalchemy.create_engine(f'sqlite:///{con}')

    times = []
    for _ in tqdm.tqdm(range(20)):
        s = time.time()
        df = pandas.read_sql(sql = query, con = SQL_ENGINE)

        times.append(time.time() - s)

    seaborn.kdeplot(times)
    plt.show()

    #df = LocalDatabase.raw_sql(query = query, conn = con)
    #df = connectorx.read_sql(query = query, conn = f'sqlite:///{con}')

    #df = pandas.read_sql(sql = query, con = SQL_ENGINE)


if __name__ == '__main__':
    main()