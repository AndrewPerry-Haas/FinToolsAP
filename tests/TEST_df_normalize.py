import os
import sys
import time
import tqdm
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

import polars

# for FutureWarning: downcast object fillna blah blah blah infer_objects(copy = False)
# and the connectorx future warning
import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

def main():

    data_path = pathlib.Path('/home/andrewperry/Nextcloud/Research/Bank Elasticity')
    tab_path = pathlib.Path('/home/andrewperry/Nextcloud/Research/Bank Elasticity/writeup/tables')
    fig_path = pathlib.Path('/home/andrewperry/Nextcloud/Research/Bank Elasticity/writeup/figures')

    DB = LocalDatabase.LocalDatabase(
        save_directory = data_path, 
        database_name = 'BankElasticityDB'
    ) 

    # cds_df.shape = (1083, 7)
    cds_df = DB.queryDB(DB.DBP.Bloomberg.CDS)
    cds_df = cds_df.dropna(axis = 0)
    
    # NOTE: every bank has the same columns
    # get the columns for each tableW
    table_column_map = {}
    
    for _table in DB.DBP.UBPR.TABLES:
        
        df = DB.queryDB(f'UBPR.{_table}',
                        idrssd = int(cds_df.idrssd.unique()[0]),
                        all_vars = True,
                        row_limit = 1
                    )
        df = df.drop(columns = ['?', 'date', 'idrssd'])
        
        table_column_map[_table] = set(df.columns)
                
    # columns of all tables
    unified_columns = set.union(*list(table_column_map.values()))
    
    for _table, _cols in table_column_map.items():
        
        # get intersection of columns in table and in unified columns        
        table_column_map[_table] = list(set.intersection(unified_columns, set(_cols)))
        
        # remove columns from unified columns to duplicates arent quiried
        unified_columns -= set(_cols)

    # query the data for all banks
    rssd_df_map = {}
    rssd_columns_after_filters_map = {}
    for _rssd in tqdm.tqdm(cds_df.idrssd.unique(), desc = 'Loading Data'):
        
        _dfs_to_merge = []
        
        for _table, _cols in table_column_map.items():
            
            df = DB.queryDB(f'UBPR.{_table}',
                            idrssd = int(_rssd),
                            vars = _cols
                        )
            _dfs_to_merge.append(df)
            
        _merged_df = functools.reduce(lambda x, y: pandas.merge(x, y, 
                                                                how = 'inner', 
                                                                on = ['idrssd', 'date']),
                                      _dfs_to_merge
                                    )
        _merged_df = _merged_df.sort_values(by = 'date')
        _merged_df = _merged_df.reset_index(drop = True)
        
        # drop columns that have nans
        # _merged_df.shape = (85, 2943)
        # _merged_df.shape = (85, 1754)        
        _merged_df = _merged_df.dropna(axis = 1)

        # remove colunmns that were confidential at any point in time
        # _merged_df.shape = (85, 1754)
        # _merged_df.shape = (85, 1741)
        _merged_df = _merged_df[[c for c in _merged_df if 'CONF' not in _merged_df[c].unique()]]

        # drop columns that have a constant value
        # _merged_df.shape = (85, 1741)
        # _merged_df.shape = (85, 1595)
        _numeric_cols = _merged_df.select_dtypes(include = 'number')
        _nonnumeric_cols = _merged_df.select_dtypes(exclude = 'number')
        ubpr_cols = [x for x in _numeric_cols.columns if 'ubpr' in x]
        _merged_df = _merged_df[[c for c in ubpr_cols if numpy.std(_merged_df[c]) != 0]]
        _merged_df = _merged_df.merge(_nonnumeric_cols, 
                                      right_index = True,
                                      left_index = True)
        
        # add back rssd that got removed in last step 
        _merged_df['idrssd'] = _rssd
        
        rssd_columns_after_filters_map[_rssd] = set(_merged_df.columns)
        rssd_df_map[_rssd] = _merged_df
    
    columns_that_survivied_filters = set.intersection(*list(rssd_columns_after_filters_map.values()))

    for _rssd, _df in rssd_df_map.items():
        rssd_df_map[_rssd] = _df[list(columns_that_survivied_filters)]
        
    # data.shape = (1438, 612)
    data = pandas.concat(list(rssd_df_map.values()))
    fc = ['date', 'idrssd']
    lc = [c for c in data.columns if c not in fc]
    data = data[fc + lc]
    
    # align cds dates to end of quarter 
    # (some dont fall exactly on the quarter but are within a few days)
    cds_df.date += pandas.tseries.offsets.QuarterEnd(0)

    # data.shape = (1438, 612)
    # data.shape = (1026, 617)
    data = cds_df.merge(data, how = 'inner', on = ['idrssd', 'date'])
    
    # column names
    ubpr_cols = [x for x in data.columns if 'ubpr' in x]
    ubpr_lags = [f'{x}_L1' for i, x in enumerate(data.columns) if 'ubpr' in x]
    
    # windsorize at 1% and 99%
    print(data['ubpre556'].describe())
    
    data = UtilityFunctions.df_normalize(
        df = data,
        gr = 'idrssd', 
        vr = ubpr_cols,
        method = 'clip',
        quantiles = (0.01, 0.99)
    )
    
    print(data['ubpre556'].describe())
    
    raise ValueError

if __name__ == '__main__':
    main()