from __future__ import annotations

import wrds
import numpy
import pandas
import typing
import datetime
import pandas.tseries.offsets

class WebData:
    """
    Class to download data from WRDS.
    """
    
    def __init__(self, username: str):
        """
        Initialize a WebData instance and establish a WRDS connection.
        
        Parameters:
            username (str): The username used to authenticate with the WRDS database.
            
        Attributes:
            username (str): Stores the provided username.
            wrds_db: A WRDS Connection object initialized using the provided username.
            valid_fields (dict): A mapping of short field names to descriptive labels for data fields.
            
        Note:
            The WRDS connection is established using the provided username. Ensure that the username 
                has access to the required datasets: CRSP.MSEALL, CRSP.MSF, CRSP.DSF, CRSP.MSI, CRSP.DSI, 
                CRSP.CCMXPF_LINKTABLE, COMPA.FUNDQ.
                
            On first use, the user will be prompted to enter their WRDS password and then establish a .pgpass
                file. This is done by the wrds package and is not controlled by this class.
        """
        self.username = username
        self.wrds_db = wrds.Connection(username=self.username)
        self.valid_fields = {
            'ticker': 'Ticker',
            'date': 'Date',
            'permco': 'PERMCO',
            'comnam': 'Company Name',
            'cusip': 'CUSIP',
            'hsiccd': 'SIC Code',
            'shrcd': 'Share Code',
            'exchcd': 'Exchnage Code',
            'prc': 'Price',
            'me': 'Market equity (millions)',
            'shrout': 'Shares outsatanding',
            'ret': 'Return',
            'retx': 'Return sans dividends',
            'bidlo': 'Low price. For monthly data, this is the lowest price of the month. For daily data, this is the lowest price of the day',
            'askhi': 'Hi price. For monthly data, this is the highest price of the month. For daily data, this is the highest price of the day',
            'vol': 'Volume',
            'div': 'Dividend amount. Calculated as the difference between the return and the return sans dividends',
            'dp': 'Dividend yield. Calculated as a rolling yearly sum of dividends divided by the share price',
            'dps': 'Dividend per share. Calculated as a rolling yearly sum of dividends divided by shares outstanding',
            'pps': 'Price per share',
            'be': 'Book Equity',
            'earn': 'Earnings',
            'atq': 'Total Assets',
            'ltq': 'Total Liabilities',
            'bm': 'Book to Market',
            'bps': 'Book equity to share',
            'ep': 'Earnings to price. Calculateed as a rolling yearly sum of earnings divided by the market equity.',
            'eps': 'Earnings per share',
            'spindx': 'CRSP S&P 500 index',
            'sprtrn': 'CRSP S&P 500 index return',
            'vwretd': 'CRSP value weighted index return',
            'vwretx': 'CRSP value weighted index return sans dividends',
            'ewretd': 'CRSP equal weighted index return',
            'ewretx': 'CRSP equal weighted index return sans dividends',
            'totval': 'Nominal value of the CRSP indicies',
            'totcnt': 'Number of firms in the CRSP indicies'
        }
                
    def __del__(self):
        """
        Close the WRDS connection when the class is deleted.
        """
        self.wrds_db.close()
        
    def __repr__(self):
        """
        Return a string representation of the class with the WRDS username and nicely formatted valid fields.
        """
        fields_str = "\n".join([f"  {key}: {value}" for key, value in self.valid_fields.items()])
        return f"WRDS Username: {self.username}\nValid Fields:\n{fields_str}"
    
    def __str__(self):
        """
        Return a string representation of the class.
        """
        return self.__repr__()
    
    def getValidFields(self) -> dict[str, str]:
        """
        Return a dictionary of valid fields and their descriptions.
        
        Returns:
            dict: A dictionary where keys are field names and values are their descriptions.
        """
        return self.valid_fields
    
    def getData(
        self, 
        tickers: list[str], 
        fields: list[str] = None, 
        freq: str = 'M',
        start_date: typing.Any = None, 
        end_date: typing.Any = None) -> pandas.DataFrame:
        
        """
        Retrieves and processes financial data for given tickers and selected fields.
        This function gathers and merges several sources of financial data, cleans and transforms the data,
        and performs calculations such as rolling sums for earnings and ratios for book-to-market and earnings-to-price.
        Data is merged from multiple datasets including security, financial, compustat, and index information.
        
        Parameters:
            tickers (list[str]): A list of ticker symbols for which the data is to be retrieved.
            fields (list[str], optional): Specific fields to include in the final DataFrame. If None, defaults to the object's valid_fields.
            freq (str, optional): Frequency of the data. 'M' for monthly or 'D' for daily data. Default is 'M'.
            start_date (any, optional): Start date for the data. Any value convertible to a datetime will work. 
                                        If None, defaults to January 1, 1900.
            end_date (any, optional): End date for the data. Any value convertible to a datetime will work. 
                                      If None, defaults to the current date and time.
            
        Returns:
            pandas.DataFrame: A DataFrame containing the merged and processed financial data with the specified fields.
            
        Raises:
            ValueError: If the frequency is not 'M' or 'D', if provided fields are invalid or empty, if the tickers list is empty,
                        or if start_date/end_date cannot be converted to datetime.
            TypeError: If tickers or fields are not provided as lists or if elements in these lists are not strings.
        """
        
        if start_date is None:
            start_date = datetime.datetime(1900, 1, 1)
        
        if end_date is None:
            end_date = datetime.datetime.now()
        
        # Convert start_date and end_date to datetime objects if possible.
        try:
            start_date = pandas.to_datetime(start_date)
            end_date = pandas.to_datetime(end_date)
        except Exception as e:
            raise ValueError("start_date and end_date must be convertible to datetime objects.") from e
        
        # Check if start_date is before end_date
        if start_date > end_date:
            raise ValueError("start_date must be before end_date.")
            
        if fields is None:
            fields = list(self.valid_fields)
        
        if freq not in ['M', 'D']:
            raise ValueError('Frequency must be either M or D for monthly or daily data, respectively.')
        
        if not isinstance(tickers, list):
            raise TypeError('Tickers must be a list of strings.')
        
        if not all(isinstance(ticker, str) for ticker in tickers):
            raise TypeError('All tickers must be strings.')
        
        if not isinstance(fields, list):
            raise TypeError('Fields must be a list of strings.')
        
        if not all(isinstance(field, str) for field in fields):
            raise TypeError('All fields must be strings.')
        
        if not all(field in self.valid_fields for field in fields):
            raise ValueError('Invalid field(s) provided. Valid fields are: {}'.format(self.valid_fields))
        
        if len(tickers) == 0:
            raise ValueError('Tickers list is empty.')
        
        if len(fields) == 0:
            raise ValueError('Fields list is empty.')
            
        # convert to string
        start_date = '\'' + start_date.strftime('%Y-%m-%d') + '\''
        end_date  = '\'' + end_date.strftime('%Y-%m-%d') + '\''        
        
        mse_df = self._load_se_data(tickers, start_date, end_date, freq)
        
        # extract permcos
        permcos = [str(permco) for permco in mse_df['permco'].unique()]
        msf_df = self._load_sf_data(permcos, start_date, end_date, freq)
        
        # merge data
        res_df = msf_df.merge(mse_df, how = 'inner', on = ['date', 'permco'])
        
        # clean crsp data
        res_df = self._clean_crsp_data(res_df, freq)
        
        # load crsp comp link table
        link_df = self._load_ccm_link_data()
        
        # merge data
        res_df = res_df.merge(link_df, how = 'inner', on = ['permco'])
        
        # load compustat data
        gvkeys = [str(gvkey) for gvkey in res_df.gvkey.unique()]
        comp_df = self._load_comp_data(gvkeys, start_date, end_date)
        
        # merge data
        res_df = res_df.merge(comp_df, how = 'left', on = ['gvkey', 'date'])
        
        # set link bounds
        res_df = res_df[(res_df.date >= res_df.linkdt) & (res_df.date <= res_df.linkenddt)]
        
        # divide quarterly earnings by 3 to get monthly earnings
        if freq == 'M':
            res_df['earn'] = numpy.where(res_df.earn.isnull(), numpy.nan, res_df.earn / 3)
        else:
            res_df['earn'] = numpy.where(res_df.earn.isnull(), numpy.nan, res_df.earn / 63) # 63 trading days in a quarter
        
        # front fill compustat data
        res_df = res_df.sort_values(by = ['gvkey', 'date'])
        res_df[['be', 'earn', 'atq', 'ltq']] = res_df.groupby(by = 'gvkey', group_keys = False)[['be', 'earn', 'atq', 'ltq']].ffill()
        
        # book to market and share
        res_df['bm'] = numpy.where(res_df.me != 0, res_df.be / res_df.me, numpy.nan)
        res_df['bps'] = numpy.where(res_df.shrout != 0, res_df.be / res_df.shrout, numpy.nan)
        
        # earning to price and share
        # smooth earnings by summing over last 4 quarters
        res_df['earn'] = res_df.groupby('permco')['earn'].transform(
            lambda x: x.rolling(window=12, min_periods=7).sum()
        )  # annualize using a rolling sum
        
        res_df['ep'] = numpy.where(res_df.me != 0, res_df.earn / res_df.me, numpy.nan)
        res_df['eps'] = numpy.where(res_df.shrout != 0, res_df.earn / res_df.shrout, numpy.nan)
        
        res_df = res_df.drop(columns = ['linkdt', 'linkenddt'])
        res_df = res_df.drop_duplicates(subset = ['date', 'permco'])
        res_df = res_df.sort_values(by = ['permco', 'date'])
        
        # load index data
        start_date_idx = '\'' + res_df.date.min().strftime('%Y-%m-%d') + '\''
        end_date_idx  = '\'' + res_df.date.max().strftime('%Y-%m-%d') + '\''
        index_df = self._load_index_data(start_date_idx, end_date_idx, freq)
        
        # merge data
        res_df = res_df.merge(index_df, how = 'left', on = ['date'])
        
        required_fields = ['ticker', 'date', 'permco']
        for field in required_fields:
            if field not in fields:
                fields.insert(0, field)
                
        res_df = res_df[fields]
        res_df = res_df.reset_index(drop = True)
        
        return res_df
    
    def _load_index_data(
        self,
        start_date: str,
        end_date: str,
        freq: str) -> pandas.DataFrame:
        
        # load index data
        index_df = self.wrds_db.raw_sql(_build_sql_string(
            id_type=None,
            ids=None,
            fields=['date', 'spindx', 'sprtrn', 'vwretd', 'vwretx',
                    'ewretd', 'ewretx', 'totval', 'totcnt'],
            table_name=f'CRSP.{freq}SI',
            date_var='date',
            start_date=start_date,
            end_date=end_date))
        
        index_df.date = pandas.to_datetime(index_df.date)
        if freq == 'M':
            # add month end offset
            index_df.date += pandas.tseries.offsets.MonthEnd(0)
        
        return index_df
    
    def _load_comp_data(
        self,
        gvkeys: list[str],
        start_date: str,
        end_date: str) -> pandas.DataFrame:
        
        # load compustat data
        comp_df = self.wrds_db.raw_sql(_build_sql_string(
            id_type='gvkey', 
            ids=gvkeys, 
            fields=['gvkey', 'datadate', 'fyearq', 'seqq', 'txditcq', 'pstkrq',
                    'pstkq', 'ibq', 'atq', 'ltq'], 
            table_name='COMP.FUNDQ', 
            date_var='datadate',
            start_date=start_date, 
            end_date=end_date))
        
        comp_df = comp_df.rename(columns = {'datadate': 'date'})
        comp_df['date'] = pandas.to_datetime(comp_df['date'])
        comp_df.date += pandas.tseries.offsets.QuarterEnd(0)
        
        # preferrerd stock: 
        comp_df['ps'] = numpy.where(comp_df.pstkrq.isnull(), comp_df.pstkq, comp_df.pstkrq)
        comp_df.ps = numpy.where(comp_df.ps.isnull(), 0, comp_df.ps)
        
        # create book equity
        comp_df['be'] = numpy.where(comp_df.fyearq < 1993, 
                               comp_df.seqq + comp_df.txditcq - comp_df.ps, 
                               comp_df.seqq - comp_df.ps
                            ) 
        
        # earnings
        comp_df['earn'] = numpy.where(comp_df.ibq.isnull(), numpy.nan, comp_df.ibq)       
        
        comp_df = comp_df[['gvkey', 'date', 'be', 'atq', 'ltq', 'earn']]
        
        return comp_df
    
    def _load_ccm_link_data(self) -> pandas.DataFrame:
        
        # read data
        sql_str = """SELECT gvkey, lpermco, linktype, linkprim, linkdt, linkenddt FROM CRSP.CCMXPF_LINKTABLE"""
        df = self.wrds_db.raw_sql(sql_str)
    
        df = df.rename(columns = {'lpermco': 'permco'})

        # Link Type Code is a 2-character code providing additional detail on the usage of the link data available. Link Type Codes include:
        # 
        # linktype Code Description
        # LC    Link research complete. Standard connection between databases.
        # LU    Unresearched link to issue by CRSP.
        # LX    Link to a security that trades on another exchange system not included in CRSP data.
        # LD    Duplicate link to a security. Another GVKEY/IID is a better link to that CRSP record.
        # LS    Link valid for this security only. Other CRSP PERMNOs with the same PERMCO will link to other GVKEYs.
        # LN    Primary link exists but Compustat does not have prices.
        # LO    No link on issue level but company level link exists. Example includes Pre-FASB, Subsidiary, Consolidated, Combined, Pre-amend, Pro-Forma, or "-old".
        # NR    No link available. Confirmed by research.
        # NU    No link available, not yet confirmed.

        # keep only links with a starting L
        df = df[df.linktype.str.startswith('L')]

        # only keep linkprim of C or P
        df = df[(df.linkprim == 'C') | (df.linkprim == 'P')]

        # drop rows were permco is missing
        df = df.dropna(subset = 'permco')

        # if link end date is missing set it to THE YEAR 3000, NOT MUCH HAS CHANGED BUT WE LIVE UNDER WATER
        # i wanted to do the whole year 3000 thing but pandas wouldnt let me cause they only coded time
        # stamps up to the year 2200 :(
        df.linkenddt = pandas.to_datetime(df.linkenddt, errors = 'coerce')
        df.linkenddt = df.linkenddt.fillna(value = datetime.datetime(2200, 1, 1))
        
        df = df[['gvkey', 'permco', 'linkdt', 'linkenddt']]

        df = df.astype(dtype = {'gvkey': str,
                                'permco': 'Int32',
                                'linkdt': 'datetime64[ns]',
                                'linkenddt': 'datetime64[ns]'})

        df = df.drop_duplicates()
        
        return df

    def _clean_crsp_data(self, 
                         res_df: pandas.DataFrame,
                         freq: str) -> pandas.DataFrame:
        
        min_periods = 7 if freq == 'M' else 147 # trading days
        window = 12 if freq == 'M' else 252
        
        # absolute value of price
        res_df['prc'] = res_df['prc'].abs()
        res_df['bidlo'] = res_df['bidlo'].abs()
        res_df['askhi'] = res_df['askhi'].abs()
        
        # adjust for splits
        res_df['prc'] /= res_df['cfacpr']
        res_df['shrout'] *= res_df['cfacshr']
        res_df['shrout'] /= 1e3 # convert to millions
        res_df['bidlo'] /= res_df['cfacpr']
        res_df['askhi'] /= res_df['cfacpr']
        
        # calculate market equity (millions)
        res_df['me'] = res_df['prc'] * res_df['shrout']
        
        # calculate dividens
        res_df['ret'] = res_df['ret'].fillna(0)
        res_df['retx'] = res_df['retx'].fillna(0)
        res_df['div'] = (res_df.ret - res_df.retx) * res_df.prc.shift(1)
        res_df['div'] = res_df['div'].fillna(0)
        res_df['div_12m_sum'] = res_df.groupby('permco')['div'].transform(
            lambda x: x.rolling(window=window, min_periods=min_periods).sum()
        )  # annualize using a rolling sum
        res_df['dp'] = numpy.where(res_df.prc != 0, res_df['div_12m_sum'] / res_df['prc'], numpy.nan)
        res_df['dps'] = numpy.where(res_df.shrout != 0, res_df['div_12m_sum'] / res_df['shrout'], numpy.nan)

        # price per share
        res_df['pps'] = numpy.where(res_df.shrout == 0, numpy.nan, res_df.prc / res_df.shrout)
        
        # reorder columns
        res_df = res_df[['date', 'permco', 'ticker', 'comnam', 'cusip', 'hsiccd',
                        'shrcd', 'exchcd', 'prc', 'me', 'shrout', 'ret', 'retx', 
                        'bidlo', 'askhi', 'vol', 'div', 'dp', 'dps', 'pps']]

        
        return res_df

    def _load_se_data(self,
                       tickers: list[str], 
                       start_date: str,
                       end_date: str,
                       freq: str) -> pandas.DataFrame:
    
        # load mse names
        mse_df = self.wrds_db.raw_sql(_build_sql_string(
           id_type='ticker', 
           ids=tickers, 
           fields=['date', 'ticker', 'comnam', 'cusip', 'hsiccd',
                   'permco', 'shrcd', 'exchcd'], 
           table_name=f'CRSP.{freq}SEALL', 
           date_var='date',
           start_date=start_date, 
           end_date=end_date))
    
        mse_df = mse_df.sort_values(by = ['ticker', 'date'])
        
        freq = 'D' if freq == 'D' else 'ME'
    
        mse_df['date'] = pandas.to_datetime(mse_df['date'])
        mse_df = mse_df.drop_duplicates(subset=['ticker', 'date'])
        mse_df = mse_df.set_index('date')
        mse_df = mse_df.groupby(
           by = 'ticker', 
           group_keys = False
        ).resample(f'{freq}').ffill().reset_index()
       
        if(freq == 'ME'):
            mse_df.date += pandas.tseries.offsets.MonthEnd(0)
       
        mse_df = mse_df.astype(
           {'date': 'datetime64[ns]',
            'ticker': str,
            'comnam': str,
            'cusip': str,
            'hsiccd': 'Int64',
            'permco': 'Int64',
            'shrcd': 'Int64',
            'exchcd': 'Int64'}
        )
       
        return mse_df
   
   
    def _load_sf_data(self,
                       permcos: list[str], 
                       start_date: str,
                       end_date: str,
                       freq: str) -> pandas.DataFrame:
    
        # load mse names
        msf_df = self.wrds_db.raw_sql(_build_sql_string(
           id_type='permco', 
           ids=permcos, 
           fields=['date', 'permco', 'bidlo', 'askhi',
                   'cfacpr', 'cfacshr', 'prc', 'vol', 'ret', 'shrout', 'retx'], 
           table_name=f'CRSP.{freq}SF', 
           date_var='date',
           start_date=start_date, 
           end_date=end_date))
       
        msf_df['date'] = pandas.to_datetime(msf_df['date'])
        
        if freq == 'M':
            msf_df.date += pandas.tseries.offsets.MonthEnd(0)
        
        msf_df = msf_df.astype({
            'date': 'datetime64[ns]',
            'permco': 'Int64',
            'bidlo': float,
            'askhi': float,
            'cfacpr': float,
            'cfacshr': float,
            'prc': float,
            'vol': float,
            'ret': float,
            'shrout': float,
            'retx': float
        })
        
        msf_df = msf_df.drop_duplicates(subset=['permco', 'date'])
        
        return msf_df
               
    
def _build_sql_string(
    id_type: str, 
    ids: list[str], 
    fields: list[str], 
    table_name: str, 
    date_var: str,
    start_date: str, 
    end_date: str) -> str:
    
    # create argument string
    var_str = list_to_sql_str(fields)
    sql_str = f'SELECT {var_str} FROM {table_name}'
    
    # sub setting for date
    if(date_var is not None):
        sql_str += f' WHERE {date_var} BETWEEN {start_date} AND {end_date}'
    
    if id_type is not None:
        sql_str += f' AND {id_type} IN ({list_to_sql_str(ids, delimit = True)})'
    
    return sql_str

def list_to_sql_str(lst: list[typing.Any], table: str = None, delimit: bool = False) -> str:
    """
    Convert a list of values into a string representation for SQL queries.

    Parameters:
    - lst (list): The list of values.
    - table (str, optional): The table name to prefix the column names with. Default is None.
    - delimit (bool, optional): Whether to delimit values with single quotes. Default is False.

    Returns:
    str: A string representation of the list for SQL queries.

    Example:
    >>> list_to_sql_str(['name', 'age'], 'person', delimit=True)
    "'person.name', 'person.age'"
    """
    res = ''
    for var in lst:
        if(table is None):
            if(delimit):
                res += f'\'{var}\', '
            else:
                res += f'{var}, '
        else:
            res += f'{table}.{var}, '
    res = res[:-2]
    return(res)