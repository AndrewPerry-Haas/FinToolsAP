import sys
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tseries.offsets import *
import shutil
import datetime
import seaborn as sns

sys.path.insert(0, '../src/FinToolsAP/')

import LocalDatabase
import UtilityFunctions


# set printing options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', shutil.get_terminal_size()[0])
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# directory for loacl wrds database 
LOCAL_WRDS_DB = pathlib.Path('/home/andrewperry/Documents')


def main():
    # me and bm and pr2-12 breakpoints
    ff_me = pd.read_csv('FFBreakpoints/ME_Breakpoints.CSV')
    ff_bm = pd.read_csv('FFBreakpoints/BM_Breakpoints.CSV')
    ff_pr = pd.read_csv('FFBreakpoints/PR2-12_Breakpoints.CSV')
    
    
    # convert dates to datetimes
    ff_me.date = pd.to_datetime(ff_me.date, format = '%Y%m')
    ff_bm.date = pd.to_datetime(ff_bm.date, format = '%Y')
    ff_pr.date = pd.to_datetime(ff_pr.date, format = '%Y%m')
    ff_me.date += MonthEnd(0)
    ff_bm.date += YearEnd(0)
    ff_pr.date += MonthEnd(0)
    ff_me = ff_me.set_index('date').sort_index()
    ff_bm = ff_bm.set_index('date').sort_index()
    ff_pr = ff_pr.set_index('date').sort_index()
    ff_pr /= 100
    
    # compute my breakpoints
    DB = LocalDatabase.LocalDatabase(LOCAL_WRDS_DB, 'LCLDB')
    start_date = datetime.date(1900, 6, 30)
    end_date = datetime.date(2100, 6, 30)
    
    raise
    
    
    
    
    ccm_df = ccm_df[ccm_df.years_in >= 2]
    nyse_df = ccm_df[(ccm_df.exchcd == '1')]
    nyse_bm_df = nyse_df[(nyse_df.bm > 0) & (nyse_df.me > 0)]
    ap_me = FF.breakpoint_ts_new(nyse_df, vars = ['me'])
    ap_bm = FF.breakpoint_ts_new(nyse_bm_df, vars = ['bm'])
    ap_pr = FF.breakpoint_ts_new(nyse_df, vars = ['pr2_12'])
    ap_me = ap_me.set_index('date').sort_index()
    ap_bm = ap_bm.set_index('date').sort_index()
    ap_pr = ap_pr.set_index('date').sort_index()
    ff_me = ff_me.loc[ff_me.index >= ap_me.index.min()]
    ff_me = ff_me.loc[ff_me.index <= ap_me.index.max()]
    ff_bm = ff_bm.loc[ff_bm.index >= ap_bm.index.min()]
    ff_bm = ff_bm.loc[ff_bm.index <= ap_bm.index.max()]
    ff_pr = ff_pr.loc[ff_pr.index >= ap_pr.index.min()]
    ff_pr = ff_pr.loc[ff_pr.index <= ap_pr.index.max()]
    fig, ax = plt.subplots(3, 1, figsize = (32, 18))
    fig.suptitle('TS of Market Equity Breakpoints')
    ax[0].plot(ff_me['30.00%'], label = 'ff')
    ax[0].plot(ap_me['me_30%'], label = 'mine')
    ax[0].set_ylabel('30th Percentile')
    ax[1].plot(ff_me['50.00%'], label = 'ff')
    ax[1].plot(ap_me['me_50%'], label = 'mine')
    ax[1].set_ylabel('50th Percentile')
    ax[2].plot(ff_me['70.00%'], label = 'ff')
    ax[2].plot(ap_me['me_70%'], label = 'mine')
    ax[2].set_ylabel('70th Percentile')
    plt.legend()
    fig, ax = plt.subplots(3, 1, figsize = (32, 18))
    fig.suptitle('TS of Book-to-Market Breakpoints')
    ax[0].plot(ff_bm['30.00%'], label = 'ff')
    ax[0].plot(ap_bm['bm_30%'], label = 'mine')
    ax[0].set_ylabel('30th Percentile')
    ax[1].plot(ff_bm['50.00%'], label = 'ff')
    ax[1].plot(ap_bm['bm_50%'], label = 'mine')
    ax[1].set_ylabel('50th Percentile')
    ax[2].plot(ff_bm['70.00%'], label = 'ff')
    ax[2].plot(ap_bm['bm_70%'], label = 'mine')
    ax[2].set_ylabel('70th Percentile')
    plt.legend()
    fig, ax = plt.subplots(3, 1, figsize = (32, 18))
    fig.suptitle('TS of Prior 2-12 Return Breakpoints')
    ax[0].plot(ff_pr['30.00%'], label = 'ff')
    ax[0].plot(ap_pr['pr2_12_30%'], label = 'mine')
    ax[0].set_ylabel('30th Percentile')
    ax[1].plot(ff_pr['50.00%'], label = 'ff')
    ax[1].plot(ap_pr['pr2_12_50%'], label = 'mine')
    ax[1].set_ylabel('50th Percentile')
    ax[2].plot(ff_pr['70.00%'], label = 'ff')
    ax[2].plot(ap_pr['pr2_12_70%'], label = 'mine')
    ax[2].set_ylabel('70th Percentile')
    plt.legend()
    ########################################################################################
    # distribution of bm
    sns.displot(ccm_df.bm, kind = 'kde')
    ########################################################################################
    # 6 sorts
    ccm_bm_sorts_df = ccm_df[(ccm_df.bm > 0) & (ccm_df.me > 0)]
    sortsBMME_df = FF.sort_portfolios_new(ccm_bm_sorts_df, char_bkpts = {'me': [0.5], 'bm': [0.3, 0.7]}, tqdm_desc = 'Sorting on ME & BM', drop_na = False, rebalance_freq = 'A')
    sortsBMME_df = sortsBMME_df.set_index('date').sort_index()
    sortsBMME_df = sortsBMME_df[sortsBMME_df.index > datetime.datetime(1970, 1, 1)]
    
    ff_BMME = pd.read_csv('FFSorts/Sorts6_ME_BM.CSV')
    ff_BMME.date = pd.to_datetime(ff_BMME.date, format = '%Y%m')
    ff_BMME.date += MonthEnd(0)
    ff_BMME = ff_BMME.set_index('date').sort_index()
    ff_BMME_nf = pd.read_csv('FFSorts/Sorts6_ME_BM_num_firms.CSV')
    ff_BMME_nf.date = pd.to_datetime(ff_BMME_nf.date, format = '%Y%m')
    ff_BMME_nf.date += MonthEnd(0)
    ff_BMME_nf = ff_BMME_nf.set_index('date').sort_index()
    ff_BMME = ff_BMME.loc[ff_BMME.index >= sortsBMME_df.index.min()]
    ff_BMME = ff_BMME.loc[ff_BMME.index <= sortsBMME_df.index.max()]
    ff_BMME_nf = ff_BMME_nf.loc[ff_BMME_nf.index >= sortsBMME_df.index.min()]
    ff_BMME_nf = ff_BMME_nf.loc[ff_BMME_nf.index <= sortsBMME_df.index.max()]
    print(sortsBMME_df.describe() * 100)
    print(ff_BMME.describe())

    corr_1 = ff_BMME['SMALL LoBM'].corr(sortsBMME_df.me1_bm1)
    corr_2 = ff_BMME['ME1 BM2'].corr(sortsBMME_df.me1_bm2)
    corr_3 = ff_BMME['SMALL HiBM'].corr(sortsBMME_df.me1_bm3)
    corr_4 = ff_BMME['BIG LoBM'].corr(sortsBMME_df.me2_bm1)
    corr_5 = ff_BMME['ME2 BM2'].corr(sortsBMME_df.me2_bm2)
    corr_6 = ff_BMME['BIG HiBM'].corr(sortsBMME_df.me2_bm3)

    fig, ax = plt.subplots(3, 2, figsize = (32, 18))
    ax[0, 0].plot(ff_BMME['SMALL LoBM'] / 100, label = 'ff')
    ax[0, 0].plot(sortsBMME_df.me1_bm1, label = 'mine')
    ax[0, 0].legend()
    ax[0, 0].set_title(f'Small Lo BM: Corr = {corr_1}')
    ax[0, 0].set_ylabel('Return')

    ax[1, 0].plot(ff_BMME['ME1 BM2'] / 100, label = 'ff')
    ax[1, 0].plot(sortsBMME_df.me1_bm2, label = 'mine')
    ax[1, 0].legend()
    ax[1, 0].set_title(f'Small Med BM: Corr = {corr_2}')
    ax[1, 0].set_ylabel('Return')

    ax[2, 0].plot(ff_BMME['SMALL HiBM'] / 100, label = 'ff')
    ax[2, 0].plot(sortsBMME_df.me1_bm3, label = 'mine')
    ax[2, 0].legend()
    ax[2, 0].set_title(f'Small Hi BM: Corr = {corr_3}')
    ax[2, 0].set_ylabel('Return')

    ax[0, 1].plot(ff_BMME['BIG LoBM'] / 100, label = 'ff')
    ax[0, 1].plot(sortsBMME_df.me2_bm1, label = 'mine')
    ax[0, 1].legend()
    ax[0, 1].set_title(f'BIG Lo BM: Corr = {corr_4}')
    ax[0, 1].set_ylabel('Return')

    ax[1, 1].plot(ff_BMME['ME2 BM2'] / 100, label = 'ff')
    ax[1, 1].plot(sortsBMME_df.me2_bm2, label = 'mine')
    ax[1, 1].legend()
    ax[1, 1].set_title(f'BIG Med BM: Corr = {corr_5}')
    ax[1, 1].set_ylabel('Return')
    ax[2, 1].plot(ff_BMME['BIG HiBM'] / 100, label = 'ff')
    ax[2, 1].plot(sortsBMME_df.me2_bm3, label = 'mine')
    ax[2, 1].legend()
    ax[2, 1].set_title(f'BIG Hi BM: Corr = {corr_6}')
    ax[2, 1].set_ylabel('Return')
    sortsBMME_df['roll_me1_bm1'] = sortsBMME_df.me1_bm1.rolling(window = 12, min_periods = 12).mean()
    sortsBMME_df['roll_me1_bm1_std'] = sortsBMME_df.me1_bm1.rolling(window = 12, min_periods = 12).std()
    sortsBMME_df['roll_me1_bm2'] = sortsBMME_df.me1_bm2.rolling(window = 12, min_periods = 12).mean()
    sortsBMME_df['roll_me1_bm2_std'] = sortsBMME_df.me1_bm2.rolling(window = 12, min_periods = 12).std()
    sortsBMME_df['roll_me1_bm3'] = sortsBMME_df.me1_bm3.rolling(window = 12, min_periods = 12).mean()
    sortsBMME_df['roll_me1_bm3_std'] = sortsBMME_df.me1_bm3.rolling(window = 12, min_periods = 12).std()
    sortsBMME_df['roll_me2_bm1'] = sortsBMME_df.me2_bm1.rolling(window = 12, min_periods = 12).mean()
    sortsBMME_df['roll_me2_bm1_std'] = sortsBMME_df.me2_bm1.rolling(window = 12, min_periods = 12).std()
    sortsBMME_df['roll_me2_bm2'] = sortsBMME_df.me2_bm2.rolling(window = 12, min_periods = 12).mean()
    sortsBMME_df['roll_me2_bm2_std'] = sortsBMME_df.me2_bm2.rolling(window = 12, min_periods = 12).std()
    sortsBMME_df['roll_me2_bm3'] = sortsBMME_df.me2_bm3.rolling(window = 12, min_periods = 12).mean()
    sortsBMME_df['roll_me2_bm3_std'] = sortsBMME_df.me2_bm3.rolling(window = 12, min_periods = 12).std()
    ff_BMME['roll_SMALL_LoBM'] = ff_BMME['SMALL LoBM'].rolling(window = 12, min_periods = 12).mean()
    ff_BMME['roll_ME1_BM2'] = ff_BMME['ME1 BM2'].rolling(window = 12, min_periods = 12).mean()
    ff_BMME['roll_SMALL_HiBM'] = ff_BMME['SMALL HiBM'].rolling(window = 12, min_periods = 12).mean()
    ff_BMME['roll_BIG_LoBM'] = ff_BMME['BIG LoBM'].rolling(window = 12, min_periods = 12).mean()
    ff_BMME['roll_ME2_BM2'] = ff_BMME['ME2 BM2'].rolling(window = 12, min_periods = 12).mean()
    ff_BMME['roll_BIG_HiBM'] = ff_BMME['BIG HiBM'].rolling(window = 12, min_periods = 12).mean()
    fig2, ax2 = plt.subplots(3, 2, figsize = (32, 18))
    ax2[0, 0].plot(ff_BMME['roll_SMALL_LoBM'] / 100, label = 'ff')
    ax2[0, 0].plot(sortsBMME_df.roll_me1_bm1, label = 'mine')
    ax2[0, 0].legend()
    ax2[0, 0].set_title(f'Small Lo BM: Corr = {corr_1}')
    ax2[0, 0].set_ylabel('Return')

    ax2[1, 0].plot(ff_BMME['roll_ME1_BM2'] / 100, label = 'ff')
    ax2[1, 0].plot(sortsBMME_df.roll_me1_bm2, label = 'mine')
    ax2[1, 0].legend()
    ax2[1, 0].set_title(f'Small Med BM: Corr = {corr_2}')
    ax2[1, 0].set_ylabel('Return')

    ax2[2, 0].plot(ff_BMME['roll_SMALL_HiBM'] / 100, label = 'ff')
    ax2[2, 0].plot(sortsBMME_df.roll_me1_bm3, label = 'mine')
    ax2[2, 0].legend()
    ax2[2, 0].set_title(f'Small Hi BM: Corr = {corr_3}')
    ax2[2, 0].set_ylabel('Return')

    ax2[0, 1].plot(ff_BMME['roll_BIG_LoBM'] / 100, label = 'ff')
    ax2[0, 1].plot(sortsBMME_df.roll_me2_bm1, label = 'mine')
    ax2[0, 1].legend()
    ax2[0, 1].set_title(f'BIG Lo BM: Corr = {corr_4}')
    ax2[0, 1].set_ylabel('Return')

    ax2[1, 1].plot(ff_BMME['roll_ME2_BM2'] / 100, label = 'ff')
    ax2[1, 1].plot(sortsBMME_df.roll_me2_bm2, label = 'mine')
    ax2[1, 1].legend()
    ax2[1, 1].set_title(f'BIG Med BM: Corr = {corr_5}')
    ax2[1, 1].set_ylabel('Return')
    ax2[2, 1].plot(ff_BMME['roll_BIG_HiBM'] / 100, label = 'ff')
    ax2[2, 1].plot(sortsBMME_df.roll_me2_bm3, label = 'mine')
    ax2[2, 1].legend()
    ax2[2, 1].set_title(f'BIG Hi BM: Corr = {corr_6}')
    ax2[2, 1].set_ylabel('Return')
     
     
    # Number of firms
    fig, ax = plt.subplots(3, 2, figsize = (32, 18))
    ax[0, 0].plot(ff_BMME_nf['SMALL LoBM'], label = 'ff')
    ax[0, 0].plot(sortsBMME_df.me1_bm1_num_firms, label = 'mine')
    ax[0, 0].legend()
    ax[0, 0].set_title(f'Small Lo BM')
    ax[0, 0].set_ylabel('Number of Firms')

    ax[1, 0].plot(ff_BMME_nf['ME1 BM2'], label = 'ff')
    ax[1, 0].plot(sortsBMME_df.me1_bm2_num_firms, label = 'mine')
    ax[1, 0].legend()
    ax[1, 0].set_title(f'Small Med BM')
    ax[1, 0].set_ylabel('Number of Firms')

    ax[2, 0].plot(ff_BMME_nf['SMALL HiBM'], label = 'ff')
    ax[2, 0].plot(sortsBMME_df.me1_bm3_num_firms, label = 'mine')
    ax[2, 0].legend()
    ax[2, 0].set_title(f'Small Hi BM')
    ax[2, 0].set_ylabel('Number of Firms')

    ax[0, 1].plot(ff_BMME_nf['BIG LoBM'], label = 'ff')
    ax[0, 1].plot(sortsBMME_df.me2_bm1_num_firms, label = 'mine')
    ax[0, 1].legend()
    ax[0, 1].set_title(f'BIG Lo BM')
    ax[0, 1].set_ylabel('Number of Firms')

    ax[1, 1].plot(ff_BMME_nf['ME2 BM2'], label = 'ff')
    ax[1, 1].plot(sortsBMME_df.me2_bm2_num_firms, label = 'mine')
    ax[1, 1].legend()
    ax[1, 1].set_title(f'BIG Med BM')
    ax[1, 1].set_ylabel('Number of Firms')
    ax[2, 1].plot(ff_BMME_nf['BIG HiBM'], label = 'ff')
    ax[2, 1].plot(sortsBMME_df.me2_bm3_num_firms, label = 'mine')
    ax[2, 1].legend()
    ax[2, 1].set_title(f'BIG Hi BM')
    ax[2, 1].set_ylabel('Number of Firms')
    # HML, SMB, MOM
    ff_BMME['SMB'] = ((1/3) * (ff_BMME['SMALL LoBM'] + ff_BMME['ME1 BM2'] + ff_BMME['SMALL HiBM']) - (1/3) * (ff_BMME['BIG LoBM'] + ff_BMME['ME2 BM2'] + ff_BMME['BIG HiBM'])) / 100
    sortsBMME_df['SMB'] = (1/3) * (sortsBMME_df.me1_bm1 + sortsBMME_df.me1_bm2 + sortsBMME_df.me1_bm3) - (1/3) * (sortsBMME_df.me2_bm1 + sortsBMME_df.me2_bm2 + sortsBMME_df.me2_bm3)
    ff_BMME['HML'] = ((1/2) * (ff_BMME['SMALL HiBM'] + ff_BMME['BIG HiBM']) - (1/2) * (ff_BMME['SMALL LoBM'] + ff_BMME['BIG LoBM'])) / 100
    sortsBMME_df['HML'] = (1/2) * (sortsBMME_df.me1_bm3 + sortsBMME_df.me2_bm3) - (1/2) * (sortsBMME_df.me1_bm1 + sortsBMME_df.me2_bm1)
    sortsBMME_df['roll_SMB'] = sortsBMME_df.SMB.rolling(window = 12, min_periods = 12).mean()
    ff_BMME['roll_SMB'] = ff_BMME.SMB.rolling(window = 12, min_periods = 12).mean()
    sortsBMME_df['roll_SMB_std'] = sortsBMME_df.SMB.rolling(window = 12, min_periods = 12).std()
    ff_BMME['roll_SMB_std'] = ff_BMME.SMB.rolling(window = 12, min_periods = 12).std()
    sortsBMME_df['roll_HML'] = sortsBMME_df.HML.rolling(window = 12, min_periods = 12).mean()
    ff_BMME['roll_HML'] = ff_BMME.HML.rolling(window = 12, min_periods = 12).mean()
    sortsBMME_df['roll_HML_std'] = sortsBMME_df.HML.rolling(window = 12, min_periods = 12).std()
    ff_BMME['roll_HML_std'] = ff_BMME.HML.rolling(window = 12, min_periods = 12).std()
    corr_SMB = ff_BMME.SMB.corr(sortsBMME_df.SMB)
    corr_HML = ff_BMME.HML.corr(sortsBMME_df.HML)
    corr_roll_SMB = ff_BMME.roll_SMB.corr(sortsBMME_df.roll_SMB)
    corr_roll_HML = ff_BMME.roll_HML.corr(sortsBMME_df.roll_HML)
    corr_roll_std_SMB = ff_BMME.roll_SMB_std.corr(sortsBMME_df.roll_SMB_std)
    corr_roll_std_HML = ff_BMME.roll_HML_std.corr(sortsBMME_df.roll_HML_std)
    fig, ax = plt.subplots(2, 1, figsize = (32, 18))
    ax[0].plot(ff_BMME.SMB, label = 'ff')
    ax[0].plot(sortsBMME_df.SMB, label = 'mine')
    ax[0].legend()
    ax[0].set_title(f'SMB: Corr = {corr_SMB}')
    ax[0].set_ylabel('Return')
    ax[1].plot(ff_BMME.HML, label = 'ff')
    ax[1].plot(sortsBMME_df.HML, label = 'mine')
    ax[1].legend()
    ax[1].set_title(f'HML: Corr = {corr_HML}')
    ax[1].set_ylabel('Return')
    fig, ax = plt.subplots(2, 1, figsize = (32, 18))
    ax[0].plot(ff_BMME.roll_SMB, label = 'ff')
    ax[0].plot(sortsBMME_df.roll_SMB, label = 'mine')
    ax[0].legend()
    ax[0].set_title(f'Rolling Average SMB: Corr = {corr_roll_SMB}')
    ax[0].set_ylabel('Return')
    ax[1].plot(ff_BMME.roll_SMB_std, label = 'ff')
    ax[1].plot(sortsBMME_df.roll_SMB_std, label = 'mine')
    ax[1].legend()
    ax[1].set_title(f'Rolling Std Dev SMB: Corr = {corr_roll_std_SMB}')
    ax[1].set_ylabel('Return')
    fig, ax = plt.subplots(2, 1, figsize = (32, 18))
    ax[0].plot(ff_BMME.roll_HML, label = 'ff')
    ax[0].plot(sortsBMME_df.roll_HML, label = 'mine')
    ax[0].legend()
    ax[0].set_title(f'Rolling Average HML: Corr = {corr_roll_HML}')
    ax[0].set_ylabel('Return')
    ax[1].plot(ff_BMME.roll_HML_std, label = 'ff')
    ax[1].plot(sortsBMME_df.roll_HML_std, label = 'mine')
    ax[1].legend()
    ax[1].set_title(f'Rolling Std Dev HML: Corr = {corr_roll_std_HML}')
    ax[1].set_ylabel('Return')
    def me_sort(row):
        if(row['me'] < row['me_50%']):
            res = 'me1'
        elif(row['me'] >= row['me_50%']):
            res = 'me2'
        else:
            res = '--fail'
        return(res)
    
    def pr_sort(row):
        if(row['pr2_12'] < row['pr2_12_30%']):
            res = 'pr2_121'
        elif(row['pr2_12'] >= row['pr2_12_30%'] and row['pr2_12'] < row['pr2_12_70%']):
            res = 'pr2_122'
        elif(row['pr2_12'] >= row['pr2_12_70%']):
            res = 'pr2_123'
        else:
            res = '--fail'
        return(res)
    df_MOM = FF.sort_portfolios_v3(ccm_df, char_bkpts = {'me': [0.5], 'pr2_12': [0.3, 0.7]}, sorting_funcs = {'me': me_sort, 'pr2_12': pr_sort}, drop_na = False, rebalance_freq = 'M')
    df_MOM = df_MOM.set_index('date').sort_index()
    df_MOM = df_MOM[df_MOM.index >= datetime.datetime(1970, 1, 1)]
    ff_MOM = pd.read_csv('FFSorts/Sorts6_ME_PR2_12.CSV')
    ff_MOM.date = pd.to_datetime(ff_MOM.date, format = '%Y%m')
    ff_MOM.date += MonthEnd(0)
    ff_MOM = ff_MOM.set_index('date').sort_index()
    ff_MOM_nf = pd.read_csv('FFSorts/Sorts6_ME_PR2_12_num_firms.CSV')
    ff_MOM_nf.date = pd.to_datetime(ff_MOM_nf.date, format = '%Y%m')
    ff_MOM_nf.date += MonthEnd(0)
    ff_MOM_nf = ff_MOM_nf.set_index('date').sort_index()
    ff_MOM = ff_MOM.loc[ff_MOM.index >= df_MOM.index.min()]
    ff_MOM = ff_MOM.loc[ff_MOM.index <= df_MOM.index.max()]
    ff_MOM_nf = ff_MOM_nf.loc[ff_MOM_nf.index >= df_MOM.index.min()]
    ff_MOM_nf = ff_MOM_nf.loc[ff_MOM_nf.index <= df_MOM.index.max()]
    print(df_MOM.describe() * 100)
    print(ff_MOM.describe())

    corr_1 = ff_MOM['SMALL LoPRIOR'].corr(df_MOM.me1_pr2_121)
    corr_2 = ff_MOM['ME1 PRIOR2'].corr(df_MOM.me1_pr2_123)
    corr_3 = ff_MOM['SMALL HiPRIOR'].corr(df_MOM.me1_pr2_123)
    corr_4 = ff_MOM['BIG LoPRIOR'].corr(df_MOM.me2_pr2_121)
    corr_5 = ff_MOM['ME2 PRIOR2'].corr(df_MOM.me2_pr2_122)
    corr_6 = ff_MOM['BIG HiPRIOR'].corr(df_MOM.me2_pr2_123)

    fig, ax = plt.subplots(3, 2, figsize = (32, 18))
    ax[0, 0].plot(ff_MOM['SMALL LoPRIOR'] / 100, label = 'ff')
    ax[0, 0].plot(df_MOM.me1_pr2_121, label = 'mine')
    ax[0, 0].legend()
    ax[0, 0].set_title(f'Small Lo Prior: Corr = {corr_1}')
    ax[0, 0].set_ylabel('Return')

    ax[1, 0].plot(ff_MOM['ME1 PRIOR2'] / 100, label = 'ff')
    ax[1, 0].plot(df_MOM.me1_pr2_122, label = 'mine')
    ax[1, 0].legend()
    ax[1, 0].set_title(f'Small Med Prior: Corr = {corr_2}')
    ax[1, 0].set_ylabel('Return')

    ax[2, 0].plot(ff_MOM['SMALL HiPRIOR'] / 100, label = 'ff')
    ax[2, 0].plot(df_MOM.me1_pr2_123, label = 'mine')
    ax[2, 0].legend()
    ax[2, 0].set_title(f'Small Hi Prior: Corr = {corr_3}')
    ax[2, 0].set_ylabel('Return')

    ax[0, 1].plot(ff_MOM['BIG LoPRIOR'] / 100, label = 'ff')
    ax[0, 1].plot(df_MOM.me2_pr2_121, label = 'mine')
    ax[0, 1].legend()
    ax[0, 1].set_title(f'BIG Lo Prior: Corr = {corr_4}')
    ax[0, 1].set_ylabel('Return')

    ax[1, 1].plot(ff_MOM['ME2 PRIOR2'] / 100, label = 'ff')
    ax[1, 1].plot(df_MOM.me2_pr2_122, label = 'mine')
    ax[1, 1].legend()
    ax[1, 1].set_title(f'BIG Med Prior: Corr = {corr_5}')
    ax[1, 1].set_ylabel('Return')
    ax[2, 1].plot(ff_MOM['BIG HiPRIOR'] / 100, label = 'ff')
    ax[2, 1].plot(df_MOM.me2_pr2_123, label = 'mine')
    ax[2, 1].legend()
    ax[2, 1].set_title(f'BIG Hi Prior: Corr = {corr_6}')
    ax[2, 1].set_ylabel('Return')
    df_MOM['roll_me1_pr2_121'] = df_MOM.me1_pr2_121.rolling(window = 12, min_periods = 12).mean()
    df_MOM['roll_me1_pr2_121_std'] = df_MOM.me1_pr2_121.rolling(window = 12, min_periods = 12).std()
    df_MOM['roll_me1_pr2_122'] = df_MOM.me1_pr2_122.rolling(window = 12, min_periods = 12).mean()
    df_MOM['roll_me1_pr2_122_std'] = df_MOM.me1_pr2_122.rolling(window = 12, min_periods = 12).std()
    df_MOM['roll_me1_pr2_123'] = df_MOM.me1_pr2_123.rolling(window = 12, min_periods = 12).mean()
    df_MOM['roll_me1_pr2_123_std'] = df_MOM.me1_pr2_123.rolling(window = 12, min_periods = 12).std()
    df_MOM['roll_me2_pr2_121'] = df_MOM.me2_pr2_121.rolling(window = 12, min_periods = 12).mean()
    df_MOM['roll_me2_pr2_121_std'] = df_MOM.me2_pr2_121.rolling(window = 12, min_periods = 12).std()
    df_MOM['roll_me2_pr2_122'] = df_MOM.me2_pr2_122.rolling(window = 12, min_periods = 12).mean()
    df_MOM['roll_me2_pr2_122_std'] = df_MOM.me2_pr2_122.rolling(window = 12, min_periods = 12).std()
    df_MOM['roll_me2_pr2_123'] = df_MOM.me2_pr2_123.rolling(window = 12, min_periods = 12).mean()
    df_MOM['roll_me2_pr2_123_std'] = df_MOM.me2_pr2_123.rolling(window = 12, min_periods = 12).std()
    ff_MOM['roll_SMALL_LoPRIOR'] = ff_MOM['SMALL LoPRIOR'].rolling(window = 12, min_periods = 12).mean()
    ff_MOM['roll_SMALL_LoPRIOR_std'] = ff_MOM['SMALL LoPRIOR'].rolling(window = 12, min_periods = 12).std()
    ff_MOM['roll_ME1_PRIOR2'] = ff_MOM['ME1 PRIOR2'].rolling(window = 12, min_periods = 12).mean()
    ff_MOM['roll_ME1_PRIOR2_std'] = ff_MOM['ME1 PRIOR2'].rolling(window = 12, min_periods = 12).std()
    ff_MOM['roll_SMALL_HiPRIOR'] = ff_MOM['SMALL HiPRIOR'].rolling(window = 12, min_periods = 12).mean()
    ff_MOM['roll_SMALL_HiPRIOR_std'] = ff_MOM['SMALL HiPRIOR'].rolling(window = 12, min_periods = 12).std()
    ff_MOM['roll_BIG_LoPRIOR'] = ff_MOM['BIG LoPRIOR'].rolling(window = 12, min_periods = 12).mean()
    ff_MOM['roll_BIG_LoPRIOR_std'] = ff_MOM['BIG LoPRIOR'].rolling(window = 12, min_periods = 12).std()
    ff_MOM['roll_ME2_PRIOR2'] = ff_MOM['ME2 PRIOR2'].rolling(window = 12, min_periods = 12).mean()
    ff_MOM['roll_ME2_PRIOR2_std'] = ff_MOM['ME2 PRIOR2'].rolling(window = 12, min_periods = 12).std()
    ff_MOM['roll_BIG_HiPRIOR'] = ff_MOM['BIG HiPRIOR'].rolling(window = 12, min_periods = 12).mean()
    ff_MOM['roll_BIG_HiPRIOR_std'] = ff_MOM['BIG HiPRIOR'].rolling(window = 12, min_periods = 12).std()
    fig2, ax2 = plt.subplots(3, 2, figsize = (32, 18))
    ax2[0, 0].plot(ff_MOM['roll_SMALL_LoPRIOR'] / 100, label = 'ff')
    ax2[0, 0].plot(df_MOM['roll_me1_pr2_121'], label = 'mine')
    ax2[0, 0].legend()
    ax2[0, 0].set_title(f'Small Lo Prior: Corr = {corr_1}')
    ax2[0, 0].set_ylabel('Return')

    ax2[1, 0].plot(ff_MOM['roll_ME1_PRIOR2'] / 100, label = 'ff')
    ax2[1, 0].plot(df_MOM['roll_me1_pr2_122'], label = 'mine')
    ax2[1, 0].legend()
    ax2[1, 0].set_title(f'Small Med Prior: Corr = {corr_2}')
    ax2[1, 0].set_ylabel('Return')

    ax2[2, 0].plot(ff_MOM['roll_SMALL_HiPRIOR'] / 100, label = 'ff')
    ax2[2, 0].plot(df_MOM['roll_me1_pr2_123'], label = 'mine')
    ax2[2, 0].legend()
    ax2[2, 0].set_title(f'Small Hi Prior: Corr = {corr_3}')
    ax2[2, 0].set_ylabel('Return')

    ax2[0, 1].plot(ff_MOM['roll_BIG_LoPRIOR'] / 100, label = 'ff')
    ax2[0, 1].plot(df_MOM['roll_me2_pr2_121'], label = 'mine')
    ax2[0, 1].legend()
    ax2[0, 1].set_title(f'BIG Lo Prior: Corr = {corr_4}')
    ax2[0, 1].set_ylabel('Return')

    ax2[1, 1].plot(ff_MOM['roll_ME2_PRIOR2'] / 100, label = 'ff')
    ax2[1, 1].plot(df_MOM['roll_me2_pr2_122'], label = 'mine')
    ax2[1, 1].legend()
    ax2[1, 1].set_title(f'BIG Med Prior: Corr = {corr_5}')
    ax2[1, 1].set_ylabel('Return')
    ax2[2, 1].plot(ff_MOM['roll_BIG_HiPRIOR'] / 100, label = 'ff')
    ax2[2, 1].plot(df_MOM['roll_me2_pr2_123'], label = 'mine')
    ax2[2, 1].legend()
    ax2[2, 1].set_title(f'BIG Hi Prior: Corr = {corr_6}')
    ax2[2, 1].set_ylabel('Return')
    fig2, ax2 = plt.subplots(3, 2, figsize = (32, 18))
    ax2[0, 0].plot(ff_MOM['roll_SMALL_LoPRIOR_std'] / 100, label = 'ff')
    ax2[0, 0].plot(df_MOM['roll_me1_pr2_121_std'], label = 'mine')
    ax2[0, 0].legend()
    ax2[0, 0].set_title(f'Small Lo Prior')
    ax2[0, 0].set_ylabel('Return')

    ax2[1, 0].plot(ff_MOM['roll_ME1_PRIOR2_std'] / 100, label = 'ff')
    ax2[1, 0].plot(df_MOM['roll_me1_pr2_122_std'], label = 'mine')
    ax2[1, 0].legend()
    ax2[1, 0].set_title(f'Small Med Prior')
    ax2[1, 0].set_ylabel('Return')

    ax2[2, 0].plot(ff_MOM['roll_SMALL_HiPRIOR_std'] / 100, label = 'ff')
    ax2[2, 0].plot(df_MOM['roll_me1_pr2_123_std'], label = 'mine')
    ax2[2, 0].legend()
    ax2[2, 0].set_title(f'Small Hi Prior')
    ax2[2, 0].set_ylabel('Return')

    ax2[0, 1].plot(ff_MOM['roll_BIG_LoPRIOR_std'] / 100, label = 'ff')
    ax2[0, 1].plot(df_MOM['roll_me2_pr2_121_std'], label = 'mine')
    ax2[0, 1].legend()
    ax2[0, 1].set_title(f'BIG Lo Prior')
    ax2[0, 1].set_ylabel('Return')

    ax2[1, 1].plot(ff_MOM['roll_ME2_PRIOR2_std'] / 100, label = 'ff')
    ax2[1, 1].plot(df_MOM['roll_me2_pr2_122_std'], label = 'mine')
    ax2[1, 1].legend()
    ax2[1, 1].set_title(f'BIG Med Prior')
    ax2[1, 1].set_ylabel('Return')
    ax2[2, 1].plot(ff_MOM['roll_BIG_HiPRIOR_std'] / 100, label = 'ff')
    ax2[2, 1].plot(df_MOM['roll_me2_pr2_123_std'], label = 'mine')
    ax2[2, 1].legend()
    ax2[2, 1].set_title(f'BIG Hi Prior')
    ax2[2, 1].set_ylabel('Return')
    
    # Number of firms
    fig, ax = plt.subplots(3, 2, figsize = (32, 18))
    ax[0, 0].plot(ff_MOM_nf['SMALL LoPRIOR'], label = 'ff')
    ax[0, 0].plot(df_MOM.me1_pr2_121_num_firms, label = 'mine')
    ax[0, 0].legend()
    ax[0, 0].set_title(f'Small Lo PRIOR')
    ax[0, 0].set_ylabel('Number of Firms')

    ax[1, 0].plot(ff_MOM_nf['ME1 PRIOR2'], label = 'ff')
    ax[1, 0].plot(df_MOM.me1_pr2_122_num_firms, label = 'mine')
    ax[1, 0].legend()
    ax[1, 0].set_title(f'Small Med PRIOR')
    ax[1, 0].set_ylabel('Number of Firms')

    ax[2, 0].plot(ff_MOM_nf['SMALL HiPRIOR'], label = 'ff')
    ax[2, 0].plot(df_MOM.me1_pr2_123_num_firms, label = 'mine')
    ax[2, 0].legend()
    ax[2, 0].set_title(f'Small Hi PRIOR')
    ax[2, 0].set_ylabel('Number of Firms')

    ax[0, 1].plot(ff_MOM_nf['BIG LoPRIOR'], label = 'ff')
    ax[0, 1].plot(df_MOM.me2_pr2_121_num_firms, label = 'mine')
    ax[0, 1].legend()
    ax[0, 1].set_title(f'BIG Lo PRIOR')
    ax[0, 1].set_ylabel('Number of Firms')

    ax[1, 1].plot(ff_MOM_nf['ME2 PRIOR2'], label = 'ff')
    ax[1, 1].plot(df_MOM.me2_pr2_122_num_firms, label = 'mine')
    ax[1, 1].legend()
    ax[1, 1].set_title(f'BIG Med PRIOR')
    ax[1, 1].set_ylabel('Number of Firms')
    ax[2, 1].plot(ff_MOM_nf['BIG HiPRIOR'], label = 'ff')
    ax[2, 1].plot(df_MOM.me2_pr2_123_num_firms, label = 'mine')
    ax[2, 1].legend()
    ax[2, 1].set_title(f'BIG Hi PRIOR')
    ax[2, 1].set_ylabel('Number of Firms')
    df_MOM['MOM'] = (1/2) * (df_MOM.me1_pr2_123 + df_MOM.me2_pr2_123) - (1/2) * (df_MOM.me1_pr2_121 + df_MOM.me1_pr2_121)
    df_MOM['roll_MOM'] = df_MOM.MOM.rolling(window = 12, min_periods = 12).mean()
    df_MOM['roll_MOM_std'] = df_MOM.MOM.rolling(window = 12, min_periods = 12).std()
    ff_MOM['MOM'] = (1/2) * (ff_MOM['SMALL HiPRIOR'] + ff_MOM['BIG HiPRIOR']) - (1/2) * (ff_MOM['SMALL LoPRIOR'] + ff_MOM['BIG LoPRIOR'])
    ff_MOM['roll_MOM'] = ff_MOM.MOM.rolling(window = 12, min_periods = 12).mean()
    ff_MOM['roll_MOM_std'] = ff_MOM.MOM.rolling(window = 12, min_periods = 12).std()
    corr_MOM = ff_MOM.MOM.corr(df_MOM.MOM)
    corr_roll_MOM = ff_MOM.roll_MOM.corr(df_MOM.roll_MOM)
    corr_roll_std_MOM = ff_MOM.roll_MOM_std.corr(df_MOM.roll_MOM_std)
    print(df_MOM.describe() * 100)
    print(ff_MOM.describe())
    fig, ax = plt.subplots(3, 1, figsize = (32, 18))
    ax[0].plot(df_MOM.MOM, label = 'mine')
    ax[0].plot(ff_MOM.MOM / 100, label = 'ff')
    ax[0].legend()
    ax[0].set_title(f'Momentum: Corr = {corr_MOM}')
    ax[0].set_ylabel('Return')
    ax[1].plot(df_MOM.roll_MOM, label = 'mine')
    ax[1].plot(ff_MOM.roll_MOM / 100, label = 'ff')
    ax[1].legend()
    ax[1].set_title(f'Momentum: Corr = {corr_roll_MOM}')
    ax[1].set_ylabel('Return')
    
    ax[2].plot(df_MOM.roll_MOM_std, label = 'mine')
    ax[2].plot(ff_MOM.roll_MOM_std / 100, label = 'ff')
    ax[2].legend()
    ax[2].set_title(f'Momentum: Corr = {corr_roll_std_MOM}')
    ax[2].set_ylabel('Return')
    plt.show()

if __name__ == "__main__":
    main()