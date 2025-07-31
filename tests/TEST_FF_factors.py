# Standard Imports
import sys
import pathlib
import pandas as pd
import shutil
from pandas.tseries.offsets import *
import datetime
import matplotlib.pyplot as plt

# Include path to custom files
sys.path.insert(0, '../src/FinToolsAP/')
import QueryWRDS
import FamaFrench

# set printing options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', shutil.get_terminal_size()[0])
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# directory for loacl wrds database 
LOCAL_WRDS_DB = pathlib.Path('/home/andrewperry/Documents/wrds_database/WRDS.db')

# directory to test data
TEST_DATA = pathlib.Path('test_data/FFSorts')

# directory to output data
OUTPUT_DATA = pathlib.Path('test_output/TEST_FF_Factors')

def main():
    ff = pd.read_csv(f'{TEST_DATA}/Sorts6_ME_BM.CSV')
    ff.date = pd.to_datetime(ff.date, format = '%Y%m')
    ff.date += MonthEnd(0)
    ff = ff.set_index('date').sort_index()
    start_date = datetime.datetime(1900, 6, 30)
    end_date = datetime.datetime(2100, 6, 30)
    DB = QueryWRDS.QueryWRDS('andrewperry', LOCAL_WRDS_DB)
    FF = FamaFrench.FamaFrench('andrewperry', LOCAL_WRDS_DB)
    ccm_df = DB.query_CCM(start_date, end_date)
    ccm_HMBSML_df = ccm_df[(ccm_df.years_in >= 2) & (ccm_df.ffbm > 0)]


    
    df = FF.sort_portfolios(ccm_HMBSML_df, 
                            char_bkpts = {'me': [0.5], 'ffbm': [0.3, 0.7]}, 
                            sorting_funcs = {'me': FF.sort_50, 'ffbm': FF.sort_3070}, 
                            rebalance_freq = 'A')
    



    df = df.set_index('date').sort_index()
    df = df[['me1_ffbm1', 'me1_ffbm2', 'me1_ffbm3', 'me2_ffbm1', 'me2_ffbm2', 'me2_ffbm3']]
    df = df[df.index >= datetime.datetime(1970, 1, 1)]
    ff = ff.loc[ff.index >= df.index.min()]
    ff = ff.loc[ff.index <= df.index.max()]
    ff['SMB'] = ((1/3) * (ff['SMALL LoBM'] + ff['ME1 BM2'] + ff['SMALL HiBM']) - (1/3) * (ff['BIG LoBM'] + ff['ME2 BM2'] + ff['BIG HiBM'])) / 100
    df['SMB'] = (1/3) * (df.me1_ffbm1 + df.me1_ffbm2 + df.me1_ffbm3) - (1/3) * (df.me2_ffbm1 + df.me2_ffbm2 + df.me2_ffbm3)
    ff['HML'] = ((1/2) * (ff['SMALL HiBM'] + ff['BIG HiBM']) - (1/2) * (ff['SMALL LoBM'] + ff['BIG LoBM'])) / 100
    df['HML'] = (1/2) * (df.me1_ffbm3 + df.me2_ffbm3) - (1/2) * (df.me1_ffbm1 + df.me2_ffbm1)
    ff.describe().to_csv(f'{OUTPUT_DATA}/ff_describe.csv')
    df.describe().to_csv(f'{OUTPUT_DATA}/mine_describe.csv')
    df['roll_SMB'] = df.SMB.rolling(window = 12, min_periods = 12).mean()
    ff['roll_SMB'] = ff.SMB.rolling(window = 12, min_periods = 12).mean()
    df['roll_SMB_std'] = df.SMB.rolling(window = 12, min_periods = 12).std()
    ff['roll_SMB_std'] = ff.SMB.rolling(window = 12, min_periods = 12).std()
    df['roll_HML'] = df.HML.rolling(window = 12, min_periods = 12).mean()
    ff['roll_HML'] = ff.HML.rolling(window = 12, min_periods = 12).mean()
    df['roll_HML_std'] = df.HML.rolling(window = 12, min_periods = 12).std()
    ff['roll_HML_std'] = ff.HML.rolling(window = 12, min_periods = 12).std()
    corr_SMB = ff.SMB.corr(df.SMB)
    corr_HML = ff.HML.corr(df.HML)
    corr_roll_SMB = ff.roll_SMB.corr(df.roll_SMB)
    corr_roll_HML = ff.roll_HML.corr(df.roll_HML)
    corr_roll_std_SMB = ff.roll_SMB_std.corr(df.roll_SMB_std)
    corr_roll_std_HML = ff.roll_HML_std.corr(df.roll_HML_std)
    fig, ax = plt.subplots(2, 1, figsize = (32, 18))
    ax[0].plot(ff.SMB, label = 'ff')
    ax[0].plot(df.SMB, label = 'mine')
    ax[0].legend()
    ax[0].set_title(f'SMB: Corr = {corr_SMB}')
    ax[0].set_ylabel('Return')
    ax[1].plot(ff.HML, label = 'ff')
    ax[1].plot(df.HML, label = 'mine')
    ax[1].legend()
    ax[1].set_title(f'HML: Corr = {corr_HML}')
    ax[1].set_ylabel('Return')
    fig.savefig(f'{OUTPUT_DATA}/SMB_HML.png')
    fig, ax = plt.subplots(2, 1, figsize = (32, 18))
    ax[0].plot(ff.roll_SMB, label = 'ff')
    ax[0].plot(df.roll_SMB, label = 'mine')
    ax[0].legend()
    ax[0].set_title(f'Rolling Average SMB: Corr = {corr_roll_SMB}')
    ax[0].set_ylabel('Return')
    ax[1].plot(ff.roll_SMB_std, label = 'ff')
    ax[1].plot(df.roll_SMB_std, label = 'mine')
    ax[1].legend()
    ax[1].set_title(f'Rolling Std Dev SMB: Corr = {corr_roll_std_SMB}')
    ax[1].set_ylabel('Return')
    fig.savefig(f'{OUTPUT_DATA}/SMB_roll.png')
    fig, ax = plt.subplots(2, 1, figsize = (32, 18))
    ax[0].plot(ff.roll_HML, label = 'ff')
    ax[0].plot(df.roll_HML, label = 'mine')
    ax[0].legend()
    ax[0].set_title(f'Rolling Average HML: Corr = {corr_roll_HML}')
    ax[0].set_ylabel('Return')
    ax[1].plot(ff.roll_HML_std, label = 'ff')
    ax[1].plot(df.roll_HML_std, label = 'mine')
    ax[1].legend()
    ax[1].set_title(f'Rolling Std Dev HML: Corr = {corr_roll_std_HML}')
    ax[1].set_ylabel('Return')
    fig.savefig(f'{OUTPUT_DATA}/HML_roll.png')
    plt.show()


if __name__ == "__main__":
    main()