import os
import sys
import time
import numpy
import pandas
import pathlib
import datetime
import functools
import collections
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

def main():
    

    data_path = pathlib.Path('/home/andrewperry/Nextcloud/Research/Bank Elasticity')
    tab_path = pathlib.Path('/home/andrewperry/Nextcloud/Research/Bank Elasticity/writeup/tables')
    fig_path = pathlib.Path('/home/andrewperry/Nextcloud/Research/Bank Elasticity/writeup/figures')

    DB = LocalDatabase.LocalDatabase(save_directory = data_path, database_name = 'BankElasticityDB') 

    cds_df = DB.queryDB(DB.DBP.Bloomberg.CDS)

    # remove citibank (keep citigroup)
    # remove first-citizens 
    # both due to data avaliability
    # Pre shape = (910, 5)
    # Post shape = (728, 5)
    cds_df = cds_df[~cds_df.id.isin(['citibank', 'firstcitizens'])]

    # remove data pre 2010 for U.S. Bank
    # Pre shape = (728, 5)
    # Post shape = (694, 5)
    ubs_df = cds_df[cds_df.id == 'usb']
    cds_df = cds_df[~cds_df.id.isin(['usb'])]
    ubs_df = ubs_df[ubs_df.date >= datetime.datetime(2010, 1, 1)]
    cds_df = pandas.concat([cds_df, ubs_df])

    # standardize cds prices
    cds_df['close_normalized'] = cds_df.groupby(
        by = 'idrssd'
    ).close.transform(lambda x: (x - x.mean()) / x.std())

    # pull in UBPR data and make a full sample

    # find columns that different banks have in common for each table
    columns_used = collections.defaultdict(dict)
    for rssd in list(cds_df.idrssd.unique()):
        _cds_df = cds_df[cds_df.idrssd == rssd]
        _fullname = _cds_df.fullname.unique()[0]
        for i, table in enumerate(DB.DBP.UBPR.TABLES):

            # load data
            table_info = f'UBPR.{table}'
            df = DB.queryDB(table_info, 
                            idrssd = int(rssd), 
                            all_vars = True
                        )

            # subset on dates and remove dumb column
            min_date = _cds_df.date.min()
            max_date = _cds_df.date.max()
            df = df[(df.date >= min_date) & (df.date <= max_date)]

            # front fillna and drop columns that have NaN
            df = df.ffill()
            df = df.dropna(axis = 1, how = 'any')

            # remove columns that are constant
            n_unique = df.nunique()
            cols_to_drop = n_unique[n_unique == 1].index
            df = df.drop(cols_to_drop, axis = 1)

            # remove non numeric columns
            non_numeric_cols = df.select_dtypes(
                exclude = numpy.number
            ).columns
            date_col = df.date
            df = df.drop(columns = non_numeric_cols)

            # add back id column and dates
            df['date'] = date_col
            df['idrssd'] = rssd
            df['fullname'] = _fullname

            # get columns to preform pca on
            pca_col = list(df.columns)
            columns_used[rssd][table] = (pca_col, df)

    # get the columns for each table that every bank has
    cols_by_table = {}
    for table in DB.DBP.UBPR.TABLES:
        list_of_list = [None] * len(list(cds_df.idrssd.unique()))
        for i, rssd in enumerate(list(cds_df.idrssd.unique())):
            list_of_list[i] = columns_used[rssd][table][0]
        
        # make sure every bank has the same columns per table
        cols_by_table[table] = (list(set.intersection(*map(set, list_of_list))))
        if(len(cols_by_table[table]) == 0):
            del cols_by_table[table]

    # combine all tables and banks into one data frame
    # data.shape = (394, 1627)
    dfs_to_concat = []
    for rssd in cds_df.idrssd.unique():
        dfs_to_merge = []
        for table in DB.DBP.UBPR.TABLES:
            _df = columns_used[rssd][table][1]
            _cols = cols_by_table[table]
            _df = _df[_cols]
            dfs_to_merge.append(_df)
        df = functools.reduce(lambda x, y: pandas.merge(x, y, 
                                                        how = 'inner', 
                                                        on = ['date', 'idrssd'], 
                                                        suffixes = (None, '_x')
                                                    ), 
                                                    dfs_to_merge
                                                )
        cols_to_keep = list(df.columns)
        cols_to_keep = [col for col in cols_to_keep if '_x' not in col]
        df = df[cols_to_keep]
        df = df.merge(cds_df[['close', 'close_normalized', 'date', 'idrssd']], 
                      how = 'inner', 
                      on = ['date', 'idrssd']
                    )
        dfs_to_concat.append(df)
    data = pandas.concat(dfs_to_concat)

    # close has 79 NaN values
    data = data.dropna(axis = 0)

    # there should be no NaNs for PCA
    assert(data.isnull().sum().sum() == 0)

    # compute pca for all banks
    for rssd in data.idrssd.unique():
        tmp = data[data.idrssd == rssd]
        
        # get columns
        pca_cols = [x for x in tmp.columns if 'ubpr' in x]
                    
        # preform PCA
        pca_res = UtilityFunctions.pca(X = tmp, vr = pca_cols, disp_corr = True)
        
        print(pca_res)


        raise ValueError



if __name__ == '__main__':
    main()