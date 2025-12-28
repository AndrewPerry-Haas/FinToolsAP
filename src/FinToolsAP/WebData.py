from __future__ import annotations

import wrds as wrds_lib
import numpy
import pandas
import typing
import datetime
import pandas.tseries.offsets

from FinToolsAP.wrds.engine import WrdsEngine

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
        # NOTE: `FinToolsAP` has a subpackage named `FinToolsAP.wrds`.
        # If callers add `src/FinToolsAP` to sys.path (instead of `src`),
        # Python can accidentally import that folder as a top-level module named `wrds`,
        # shadowing the *external* `wrds` dependency.
        if not hasattr(wrds_lib, "Connection"):
            wrds_file = getattr(wrds_lib, "__file__", None)
            raise ImportError(
                "The imported module 'wrds' does not provide 'Connection'. "
                "This usually means you're shadowing the external 'wrds' package. "
                f"Imported wrds from: {wrds_file!r}. "
                "Fix: ensure you add '.../src' (not '.../src/FinToolsAP') to PYTHONPATH, "
                "and import via 'from FinToolsAP.WebData import WebData'."
            )

        self.wrds_db = wrds_lib.Connection(username=self.username)
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
        # Dependency map: defines minimal source requirements per output field
        # Structure: { field: { 'tables': set[str], 'crsp_se_cols': set[str], 'crsp_sf_cols': set[str],
        #                       'comp_cols': set[str], 'index_cols': set[str], 'builders': set[str] } }
        # Note: This is an initial scaffold; computations will be routed via builder registry.
        self._dep_map = {
            'ticker': {'tables': {'SE'}, 'crsp_se_cols': {'ticker'}, 'crsp_sf_cols': set(), 'comp_cols': set(), 'index_cols': set(), 'builders': set()},
            'date':   {'tables': {'SE'}, 'crsp_se_cols': {'date'}, 'crsp_sf_cols': set(), 'comp_cols': set(), 'index_cols': set(), 'builders': set()},
            'permco': {'tables': {'SE'}, 'crsp_se_cols': {'permco'}, 'crsp_sf_cols': set(), 'comp_cols': set(), 'index_cols': set(), 'builders': set()},
            'prc':    {'tables': {'SF'}, 'crsp_se_cols': set(), 'crsp_sf_cols': {'date','permco','prc','cfacpr'}, 'comp_cols': set(), 'index_cols': set(), 'builders': {'split_adjust'}},
            'shrout': {'tables': {'SF'}, 'crsp_se_cols': set(), 'crsp_sf_cols': {'date','permco','shrout','cfacshr'}, 'comp_cols': set(), 'index_cols': set(), 'builders': {'split_adjust'}},
            'me':     {'tables': {'SF'}, 'crsp_se_cols': set(), 'crsp_sf_cols': {'date','permco','prc','shrout','cfacpr','cfacshr'}, 'comp_cols': set(), 'index_cols': set(), 'builders': {'split_adjust','build_me'}},
            'ret':    {'tables': {'SF'}, 'crsp_sf_cols': {'date','permco','ret'}, 'crsp_se_cols': set(), 'comp_cols': set(), 'index_cols': set(), 'builders': set()},
            'retx':   {'tables': {'SF'}, 'crsp_sf_cols': {'date','permco','retx'}, 'crsp_se_cols': set(), 'comp_cols': set(), 'index_cols': set(), 'builders': set()},
            'vol':    {'tables': {'SF'}, 'crsp_sf_cols': {'date','permco','vol'}, 'crsp_se_cols': set(), 'comp_cols': set(), 'index_cols': set(), 'builders': set()},
            'bidlo':  {'tables': {'SF'}, 'crsp_sf_cols': {'date','permco','bidlo','cfacpr'}, 'builders': {'split_adjust'}, 'crsp_se_cols': set(), 'comp_cols': set(), 'index_cols': set()},
            'askhi':  {'tables': {'SF'}, 'crsp_sf_cols': {'date','permco','askhi','cfacpr'}, 'builders': {'split_adjust'}, 'crsp_se_cols': set(), 'comp_cols': set(), 'index_cols': set()},
            'div':    {'tables': {'SF'}, 'crsp_sf_cols': {'date','permco','ret','retx','prc'}, 'crsp_se_cols': set(), 'comp_cols': set(), 'index_cols': set(), 'builders': {'build_div'}},
            'dp':     {'tables': {'SF'}, 'crsp_sf_cols': {'date','permco','ret','retx','prc'}, 'crsp_se_cols': set(), 'comp_cols': set(), 'index_cols': set(), 'builders': {'build_div','build_dp'}},
            'dps':    {'tables': {'SF'}, 'crsp_sf_cols': {'date','permco','ret','retx','prc','shrout'}, 'crsp_se_cols': set(), 'comp_cols': set(), 'index_cols': set(), 'builders': {'split_adjust','build_div','build_dps'}},
            'pps':    {'tables': {'SF'}, 'crsp_sf_cols': {'date','permco','prc','shrout'}, 'crsp_se_cols': set(), 'comp_cols': set(), 'index_cols': set(), 'builders': {'split_adjust','build_pps'}},
            'be':     {'tables': {'COMP','LINK'}, 'crsp_se_cols': set(), 'crsp_sf_cols': set(), 'comp_cols': {'gvkey','datadate','fyearq','seqq','txditcq','pstkrq','pstkq'}, 'index_cols': set(), 'builders': {'build_be'}},
            'earn':   {'tables': {'COMP','LINK'}, 'crsp_se_cols': set(), 'crsp_sf_cols': set(), 'comp_cols': {'gvkey','datadate','ibq'}, 'index_cols': set(), 'builders': {'build_earn'}},
            'atq':    {'tables': {'COMP','LINK'}, 'crsp_se_cols': set(), 'crsp_sf_cols': set(), 'comp_cols': {'gvkey','datadate','atq'}, 'index_cols': set(), 'builders': set()},
            'ltq':    {'tables': {'COMP','LINK'}, 'crsp_se_cols': set(), 'crsp_sf_cols': set(), 'comp_cols': {'gvkey','datadate','ltq'}, 'index_cols': set(), 'builders': set()},
            'bm':     {'tables': {'SF','COMP','LINK'}, 'crsp_sf_cols': {'date','permco','prc','shrout','cfacpr','cfacshr'}, 'comp_cols': {'gvkey','datadate','fyearq','seqq','txditcq','pstkrq','pstkq'}, 'index_cols': set(), 'builders': {'split_adjust','build_me','build_be','build_bm'}},
            'bps':    {'tables': {'SF','COMP','LINK'}, 'crsp_sf_cols': {'date','permco','shrout','cfacshr'}, 'comp_cols': {'gvkey','datadate','fyearq','seqq','txditcq','pstkrq','pstkq'}, 'index_cols': set(), 'builders': {'split_adjust','build_be','build_bps'}},
            'ep':     {'tables': {'SF','COMP','LINK'}, 'crsp_sf_cols': {'date','permco','prc','shrout','cfacpr','cfacshr'}, 'comp_cols': {'gvkey','datadate','ibq'}, 'index_cols': set(), 'builders': {'split_adjust','build_me','build_earn_ann','build_ep'}},
            'eps':    {'tables': {'SF','COMP','LINK'}, 'crsp_sf_cols': {'date','permco','shrout','cfacshr'}, 'comp_cols': {'gvkey','datadate','ibq'}, 'index_cols': set(), 'builders': {'split_adjust','build_earn_ann','build_eps'}},
            'spindx': {'tables': {'INDEX'}, 'crsp_se_cols': set(), 'crsp_sf_cols': set(), 'comp_cols': set(), 'index_cols': {'date','spindx'}, 'builders': set()},
            'sprtrn': {'tables': {'INDEX'}, 'index_cols': {'date','sprtrn'}, 'crsp_se_cols': set(), 'crsp_sf_cols': set(), 'comp_cols': set(), 'builders': set()},
            'vwretd': {'tables': {'INDEX'}, 'index_cols': {'date','vwretd'}, 'crsp_se_cols': set(), 'crsp_sf_cols': set(), 'comp_cols': set(), 'builders': set()},
            'vwretx': {'tables': {'INDEX'}, 'index_cols': {'date','vwretx'}, 'crsp_se_cols': set(), 'crsp_sf_cols': set(), 'comp_cols': set(), 'builders': set()},
            'ewretd': {'tables': {'INDEX'}, 'index_cols': {'date','ewretd'}, 'crsp_se_cols': set(), 'crsp_sf_cols': set(), 'comp_cols': set(), 'builders': set()},
            'ewretx': {'tables': {'INDEX'}, 'index_cols': {'date','ewretx'}, 'crsp_se_cols': set(), 'crsp_sf_cols': set(), 'comp_cols': set(), 'builders': set()},
            'totval': {'tables': {'INDEX'}, 'index_cols': {'date','totval'}, 'crsp_se_cols': set(), 'crsp_sf_cols': set(), 'comp_cols': set(), 'builders': set()},
            'totcnt': {'tables': {'INDEX'}, 'index_cols': {'date','totcnt'}, 'crsp_se_cols': set(), 'crsp_sf_cols': set(), 'comp_cols': set(), 'builders': set()},
        }

        # Builder registry scaffold: maps builder name to method. Implementations added later.
        self._builder_registry = {
            'split_adjust': self._build_split_adjustments,
            'build_me': self._build_me,
            'build_div': self._build_div,
            'build_dp': self._build_dp,
            'build_dps': self._build_dps,
            'build_pps': self._build_pps,
            'build_be': self._build_be,
            'build_earn': self._build_earn,
            'build_earn_ann': self._build_earn_annualized,
            'build_bm': self._build_bm,
            'build_bps': self._build_bps,
            'build_ep': self._build_ep,
            'build_eps': self._build_eps,
        }

        # Builder dependency order ensures prerequisites are computed first
        # Define explicit ordering and dependency hints
        self._builder_order = [
            'split_adjust',         # prepares prc/shrout/bidlo/askhi
            'build_div',            # creates div
            'build_dp',             # uses div to create dp and rolling sum
            'build_dps',            # uses div_12m_sum and shrout
            'build_pps',            # uses prc and shrout
            'build_me',             # uses prc and shrout
            'build_be',             # Compustat
            'build_earn',           # Compustat
            'build_earn_ann',       # rolling annualization of earn //FIXME - annualization window review
            'build_bm',             # uses be and me
            'build_bps',            # uses be and shrout
            'build_ep',             # uses earn and me
            'build_eps',            # uses earn and shrout
        ]
        
        self.default_fields = [
            'date',
            'permco',
            'ticker',
            'prc',
            'me',
            'ret'
        ]
        
        self.default_filters = {
            'exchcd_filter': [1, 2, 3],  # NYSE, AMEX, NASDAQ
            'shrcd_filter': [10, 11]     # Common and ordinary shares
        }
                
    def __del__(self):
        """
        Close the WRDS connection when the class is deleted.
        """
        try:
            if getattr(self, "wrds_db", None) is not None:
                self.wrds_db.close()
        except Exception:
            # best-effort cleanup only
            pass
        
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
        tickers: list[str] | None, 
        fields: list[str] | None, 
        freq: str = 'M',
        start_date: typing.Any = None, 
        end_date: typing.Any = None,
        exchcd_filter: typing.Optional[list[int]] = None,
        shrcd_filter: typing.Optional[list[int]] = None) -> pandas.DataFrame:
        
        start_date = pandas.to_datetime(
            start_date or datetime.datetime(1900, 1, 1),
            errors="raise",
        )
        end_date = pandas.to_datetime(
            end_date or datetime.datetime.now(),
            errors="raise",
        )

        if start_date > end_date:
            raise ValueError("start_date must be before end_date.")
            
        if fields is None:
            fields = list(self.default_fields)
        else:
            if not isinstance(fields, list) or not all(isinstance(f, str) for f in fields):
                raise TypeError("Fields must be a list of strings.")
            if len(fields) == 0:
                raise ValueError("Fields list is empty.")
            invalid_fields = [f for f in fields if f not in self.valid_fields]
            if invalid_fields:
                raise ValueError(
                    f"Invalid field(s) provided: {invalid_fields}. "
                    f"Valid fields are: {list(self.valid_fields.keys())}"
                )
        
        if freq not in ['M', 'D']:
            raise ValueError('Frequency must be either M or D for monthly or daily data, respectively.')
        
        # tickers may be None to trigger universe pull
        if tickers is not None:
            if not isinstance(tickers, list):
                raise TypeError('Tickers must be a list of strings.')
            if not all(isinstance(ticker, str) for ticker in tickers):
                raise TypeError('All tickers must be strings.')

        # Maintain legacy behavior: always return identity fields.
        required_fields = ['ticker', 'date', 'permco']
        for field in required_fields:
            if field not in fields:
                fields.insert(0, field)

        engine = WrdsEngine(
            fetch_se=self._load_se_data,
            fetch_sf=self._load_sf_data,
            fetch_index=self._load_index_data,
            fetch_link=self._load_ccm_link_data,
            fetch_comp=self._load_comp_data,
            clean_inputs=self._clean_inputs,
        )

        return engine.get(
            features=fields,
            tickers=tickers,
            freq=freq,
            start_date=start_date,
            end_date=end_date,
            exchcd_filter=exchcd_filter,
            shrcd_filter=shrcd_filter,
            include_identity=False,
        )

    # --- Unified WRDS loader ---
    def _load_wrds(self,
                   table_alias: str,
                   columns: list[str],
                   start_date: str,
                   end_date: str,
                   id_type: typing.Optional[str] = None,
                   ids: typing.Optional[list[str]] = None,
                   predicates: typing.Optional[str] = None,
                   date_var: typing.Optional[str] = None) -> pandas.DataFrame:
        alias_to_table = {
            'CRSP.SEALL.M': 'CRSP.MSEALL',
            'CRSP.SEALL.D': 'CRSP.DSEALL',
            'CRSP.SF.M': 'CRSP.MSF',
            'CRSP.SF.D': 'CRSP.DSF',
            'CRSP.SI.M': 'CRSP.MSI',
            'CRSP.SI.D': 'CRSP.DSI',
            'CRSP.LINK': 'CRSP.CCMXPF_LINKTABLE',
            'COMP.FUNDQ': 'COMP.FUNDQ',
        }
        table_name = alias_to_table.get(table_alias, table_alias)
        # default date var selection
        if date_var is None:
            if table_alias.startswith('CRSP.SI') or table_alias.startswith('CRSP.SEALL') or table_alias.startswith('CRSP.SF'):
                date_var = 'date'
            elif table_alias.startswith('COMP.FUNDQ'):
                date_var = 'datadate'
            else:
                date_var = None
        sql = _build_sql_string(id_type=id_type,
                                ids=ids,
                                fields=columns,
                                table_name=table_name,
                                date_var=date_var,
                                start_date=start_date,
                                end_date=end_date,
                                predicates=predicates)
        return self.wrds_db.raw_sql(sql)

    # --- Centralized input cleaning ---
    def _clean_inputs(self,
                      se_df: pandas.DataFrame | None,
                      sf_df: pandas.DataFrame | None,
                      link_df: pandas.DataFrame | None,
                      comp_df: pandas.DataFrame | None,
                      idx_df: pandas.DataFrame | None,
                      freq: str):
        # SE: resample and month-end align
        if se_df is not None and not se_df.empty:
            se_df = se_df.sort_values(['ticker','date'])
            res_freq = 'D' if freq == 'D' else 'ME'
            se_df['date'] = pandas.to_datetime(se_df['date'])
            se_df = se_df.drop_duplicates(subset=['ticker','date'])
            se_df = se_df.set_index('date').groupby('ticker', group_keys=False).resample(res_freq).ffill().reset_index()
            if res_freq == 'ME':
                se_df['date'] = se_df['date'] + pandas.tseries.offsets.MonthEnd(0)
            # drop rows lacking permco early
            if 'permco' in se_df.columns:
                se_df = se_df.dropna(subset=['permco'])

        # SF: types and month-end align; drop missing permco
        if sf_df is not None and not sf_df.empty:
            sf_df['date'] = pandas.to_datetime(sf_df['date'])
            if freq == 'M':
                sf_df['date'] = sf_df['date'] + pandas.tseries.offsets.MonthEnd(0)
            if 'permco' in sf_df.columns:
                sf_df = sf_df.dropna(subset=['permco'])

        # LINK: prefer P already handled; ensure bounds types
        if link_df is not None and not link_df.empty:
            link_df['linkdt'] = pandas.to_datetime(link_df['linkdt'])
            link_df['linkenddt'] = pandas.to_datetime(link_df['linkenddt'])

        # COMP: rename and quarter-end align; do not compute be/earn here
        if comp_df is not None and not comp_df.empty:
            if 'datadate' in comp_df.columns:
                comp_df = comp_df.rename(columns={'datadate':'date'})
            comp_df['date'] = pandas.to_datetime(comp_df['date'])
            comp_df['date'] = comp_df['date'] + pandas.tseries.offsets.QuarterEnd(0)

        # INDEX: date parse and month-end align
        if idx_df is not None and not idx_df.empty:
            idx_df['date'] = pandas.to_datetime(idx_df['date'])
            if freq == 'M':
                idx_df['date'] = idx_df['date'] + pandas.tseries.offsets.MonthEnd(0)

        return se_df, sf_df, link_df, comp_df, idx_df
    # --- Builder orchestration ---
    def _apply_builders(self, df: pandas.DataFrame, builders: set[str], freq: str) -> pandas.DataFrame:
        # Apply builders in dependency-safe order
        for b in self._builder_order:
            if b in builders:
                func = self._builder_registry.get(b)
                if func is not None:
                    df = func(df, freq)
        return df

    # --- Builder stubs (initial implementations using existing logic pieces) ---
    def _build_split_adjustments(self, df: pandas.DataFrame, freq: str) -> pandas.DataFrame:
        if 'cfacpr' in df.columns:
            if 'prc' in df.columns: df['prc'] = df['prc'].abs() / df['cfacpr']
            if 'bidlo' in df.columns: df['bidlo'] = df['bidlo'].abs() / df['cfacpr']
            if 'askhi' in df.columns: df['askhi'] = df['askhi'].abs() / df['cfacpr']
        if 'cfacshr' in df.columns and 'shrout' in df.columns:
            df['shrout'] = df['shrout'] * df['cfacshr'] / 1e3
        return df

    def _build_me(self, df: pandas.DataFrame, freq: str) -> pandas.DataFrame:
        if 'prc' in df.columns and 'shrout' in df.columns:
            df['me'] = df['prc'] * df['shrout']
        return df

    def _build_div(self, df: pandas.DataFrame, freq: str) -> pandas.DataFrame:
        if 'ret' in df.columns and 'retx' in df.columns and 'prc' in df.columns:
            df['ret'] = df['ret'].fillna(0)
            df['retx'] = df['retx'].fillna(0)
            df['div'] = (df.ret - df.retx) * df.prc.shift(1)
            df['div'] = df['div'].fillna(0)
        return df

    def _build_dp(self, df: pandas.DataFrame, freq: str) -> pandas.DataFrame:
        if 'div' in df.columns and 'prc' in df.columns:
            window = 12 if freq == 'M' else 252  # //FIXME - dividend annualization window
            min_periods = 7 if freq == 'M' else 147
            df['div_12m_sum'] = df.groupby('permco')['div'].transform(lambda x: x.rolling(window=window, min_periods=min_periods).sum())
            df['dp'] = numpy.where(df.prc != 0, df['div_12m_sum'] / df['prc'], numpy.nan)
        return df

    def _build_dps(self, df: pandas.DataFrame, freq: str) -> pandas.DataFrame:
        if 'div_12m_sum' in df.columns and 'shrout' in df.columns:
            df['dps'] = numpy.where(df.shrout != 0, df['div_12m_sum'] / df['shrout'], numpy.nan)
        return df

    def _build_pps(self, df: pandas.DataFrame, freq: str) -> pandas.DataFrame:
        if 'prc' in df.columns and 'shrout' in df.columns:
            df['pps'] = numpy.where(df.shrout == 0, numpy.nan, df.prc / df.shrout)
        return df

    def _build_be(self, df: pandas.DataFrame, freq: str) -> pandas.DataFrame:
        # Assumes Compustat columns present and renamed later in _load_comp_data
        return df

    def _build_earn(self, df: pandas.DataFrame, freq: str) -> pandas.DataFrame:
        return df

    def _build_earn_annualized(self, df: pandas.DataFrame, freq: str) -> pandas.DataFrame:
        if 'earn' in df.columns:
            # Compustat earnings are quarterly; annualize via 4-quarter rolling sum
            window = 4  # //FIXME - earnings annualization window
            min_periods = 2  # allow initial partial, adjust if needed
            df['earn'] = df.groupby('permco')['earn'].transform(lambda x: x.rolling(window=window, min_periods=min_periods).sum())
        return df

    def _build_bm(self, df: pandas.DataFrame, freq: str) -> pandas.DataFrame:
        if 'be' in df.columns and 'me' in df.columns:
            df['bm'] = numpy.where(df.me != 0, df.be / df.me, numpy.nan)
        return df

    def _build_bps(self, df: pandas.DataFrame, freq: str) -> pandas.DataFrame:
        if 'be' in df.columns and 'shrout' in df.columns:
            df['bps'] = numpy.where(df.shrout != 0, df.be / df.shrout, numpy.nan)
        return df

    def _build_ep(self, df: pandas.DataFrame, freq: str) -> pandas.DataFrame:
        if 'earn' in df.columns and 'me' in df.columns:
            df['ep'] = numpy.where(df.me != 0, df.earn / df.me, numpy.nan)
        return df

    def _build_eps(self, df: pandas.DataFrame, freq: str) -> pandas.DataFrame:
        if 'earn' in df.columns and 'shrout' in df.columns:
            df['eps'] = numpy.where(df.shrout != 0, df.earn / df.shrout, numpy.nan)
        return df
    
    def _load_index_data(
        self,
        start_date: str,
        end_date: str,
        freq: str) -> pandas.DataFrame:
        # load index using unified loader
        table_alias = f'CRSP.SI.{"M" if freq=="M" else "D"}'
        cols = ['date', 'spindx', 'sprtrn', 'vwretd', 'vwretx', 'ewretd', 'ewretx', 'totval', 'totcnt']
        index_df = self._load_wrds(table_alias, cols, start_date, end_date)
        return index_df
    
    def _load_comp_data(
        self,
        gvkeys: list[str],
        start_date: str,
        end_date: str,
        select_cols: list[str] | None = None) -> pandas.DataFrame:
        # load compustat data (no derived fields here)
        base_fields = {'gvkey','datadate'}
        default_fields = {'fyearq','seqq','txditcq','pstkrq','pstkq','ibq','atq','ltq'}
        selected_set = set(select_cols) if select_cols is not None else default_fields
        fields_req = list(selected_set | base_fields)
        comp_df = self._load_wrds('COMP.FUNDQ', fields_req, start_date, end_date, id_type='gvkey', ids=gvkeys)
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
        # prefer primary links; keep C as fallback but prioritize P
        df = df[(df.linkprim == 'C') | (df.linkprim == 'P')]
        # sort to prefer P first
        df['linkprim_order'] = numpy.where(df.linkprim == 'P', 0, 1)
        df = df.sort_values(['permco', 'linkprim_order', 'linkdt'])

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

        # drop duplicates while preferring P
        df = df.drop_duplicates(subset=['permco','gvkey','linkdt','linkenddt'])
        
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
                       freq: str,
                       select_cols: list[str] | None = None,
                       exchcd_filter: typing.Optional[list[int]] = None,
                       shrcd_filter: typing.Optional[list[int]] = None) -> pandas.DataFrame:
        # load mse names using unified loader
        fields_req = select_cols if select_cols else ['date','ticker','comnam','cusip','hsiccd','permco','shrcd','exchcd']
        preds = []
        if exchcd_filter:
            exch_list = ', '.join(str(x) for x in exchcd_filter)
            preds.append(f"exchcd IN ({exch_list})")
        if shrcd_filter:
            shr_list = ', '.join(str(x) for x in shrcd_filter)
            preds.append(f"shrcd IN ({shr_list})")
        predicate_str = ' AND '.join(preds) if len(preds) > 0 else None
        table_alias = f'CRSP.SEALL.{"M" if freq=="M" else "D"}'
        mse_df = self._load_wrds(table_alias, fields_req, start_date, end_date,
                                 id_type=('ticker' if tickers is not None else None),
                                 ids=(tickers if tickers is not None else None),
                                 predicates=predicate_str,
                                 date_var='date')

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
       
        # astype based on available columns
        astype_map = {'date': 'datetime64[ns]','ticker': str,'comnam': str,'cusip': str,
                      'hsiccd': 'Int64','permco': 'Int64','shrcd': 'Int64','exchcd': 'Int64'}
        for c, t in list(astype_map.items()):
            if c in mse_df.columns:
                mse_df[c] = mse_df[c].astype(t)
       
        return mse_df
   
   
    def _load_sf_data(self,
                       permcos: list[str], 
                       start_date: str,
                       end_date: str,
                       freq: str,
                       select_cols: list[str] | None = None) -> pandas.DataFrame:
        default_cols = ['date','permco','bidlo','askhi','cfacpr','cfacshr','prc','vol','ret','shrout','retx']
        fields_req = select_cols if select_cols else default_cols
        table_alias = f'CRSP.SF.{"M" if freq=="M" else "D"}'
        msf_df = self._load_wrds(table_alias, fields_req, start_date, end_date, id_type='permco', ids=permcos, date_var='date')
       
        msf_df['date'] = pandas.to_datetime(msf_df['date'])
        
        if freq == 'M':
            msf_df.date += pandas.tseries.offsets.MonthEnd(0)
        
        # astype only for present columns
        astype_map = {'date': 'datetime64[ns]','permco': 'Int64','bidlo': float,'askhi': float,
                      'cfacpr': float,'cfacshr': float,'prc': float,'vol': float,'ret': float,'shrout': float,'retx': float}
        for c, t in list(astype_map.items()):
            if c in msf_df.columns:
                msf_df[c] = msf_df[c].astype(t)
        
        msf_df = msf_df.drop_duplicates(subset=['permco', 'date'])
        
        return msf_df
               
    
def _build_sql_string(
    id_type: str | None, 
    ids: list[str] | None, 
    fields: list[str], 
    table_name: str, 
    date_var: str | None,
    start_date: str, 
    end_date: str,
    predicates: typing.Optional[str] = None) -> str:
    
    # create argument string
    var_str = list_to_sql_str(fields)
    sql_str = f'SELECT {var_str} FROM {table_name}'
    
    # sub setting for date
    if(date_var is not None):
        sql_str += f' WHERE {date_var} BETWEEN {start_date} AND {end_date}'
    else:
        sql_str += ' WHERE 1=1'

    if predicates:
        sql_str += f' AND {predicates}'
    
    if id_type is not None and ids is not None:
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