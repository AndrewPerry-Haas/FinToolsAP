import os
import sys
import pathlib
import shutil
import time
import datetime
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, '../src/FinToolsAP/')

import LocalDatabase
import PortfolioSorts
import Decorators
import LaTeXBuilder

# set printing options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', shutil.get_terminal_size()[0])
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# directory for loacl wrds database 

# linux
LOCAL_WRDS_DB = pathlib.Path('/home/andrewperry/Documents')


@Decorators.Performance
def query_CRSP_performance(DB):
    return(DB.queryDB(DB.DBP.CHAR, ticker = 'AAPL', all_vars = True))

def main():

    PATH_TO_DESCRIPTIVE = pathlib.Path('/home/andrewperry/Desktop/score_descriptive/bin2')

    LaTeXBuilder.table_document(input_path = PATH_TO_DESCRIPTIVE)


    raise ValueError

    DB = LocalDatabase.LocalDatabase(save_directory = LOCAL_WRDS_DB, 
                                     database_name = 'LCLDB'
                                    )

    df = query_CRSP_performance(DB)
    print(df.head())
    print(df.info(verbose = True))
    raise ValueError


    df = DB.queryDB(DB.DBP.CHAR, all_vars = True)
    print(df.shape)
    raise ValueError
    
    df = PortfolioSorts.score_characteristics(dfin = df, 
                                              vars = DB.DBP.MutualFundCharacteristics.CHARACTERISTICS,
                                              bins = 5
                                            )
    
    print(df.head(30))
    

if __name__ == "__main__":
    main()
