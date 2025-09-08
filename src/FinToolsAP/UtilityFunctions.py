from __future__ import annotations

# standard imports
import os
import math
import tqdm
import numpy
import scipy
import typing
import polars
import pandas
import pathlib
import datetime
import functools
import time
import scipy.stats
import numpy.typing
import scipy.special
import statsmodels.api
import dateutil.relativedelta

def percentile_rank(
    df: typing.Union[pandas.DataFrame, polars.DataFrame],
    vr: typing.Union[str, typing.List[str]],
    gr: typing.Union[str, typing.List[str], None] = None,
    multiply_by_100: bool = True
) -> typing.Union[pandas.DataFrame, polars.DataFrame]:
    """
    Calculate percentile ranks for one or more columns in a pandas or polars DataFrame,
    optionally within groups defined by one or more columns.

    Parameters
    ----------
    df : pandas.DataFrame or polars.DataFrame
        The input DataFrame.
    vr : str or list of str
        Column name(s) to compute percentile ranks for.
    gr : str or list of str, optional
        Column name(s) to group by. If None, uses the entire DataFrame.
    multiply_by_100 : bool, optional
        If True, multiply the 'pct=True' ranks by 100 so the range is [0,100].
        Note that `.rank(pct=True)` naturally goes from 1/N up to 1.0. If True,
        the smallest item(s) will be near ~1*(100/N) and the largest exactly 100.

    Returns
    -------
    pd.DataFrame or pl.DataFrame
        A copy of `df` with new columns `<col>_pr` for each column in `vr`.
        These columns contain the percentile rank of each observation, ignoring NaNs.
        Rows where `vr` is NaN get NaN percentile ranks.
    """

    # Convert single strings to lists for uniform handling
    if isinstance(vr, str):
        vr = [vr]
    if isinstance(gr, str):
        gr = [gr]

    if isinstance(df, pandas.DataFrame):
        # Make a copy so we don't overwrite the original
        df = df.copy()

        if gr is None:
            # Global rank
            ranks = df[vr].rank(method='average', na_option='keep', pct=True)
        else:
            ranks = (
                df[vr + gr].groupby(by=gr).rank(method='average', na_option='keep', pct=True)
            )

        # Optionally multiply by 100 to get 1..100-scale
        if multiply_by_100:
            ranks = ranks * 100

        # Rename columns to original_col + "_pr"
        new_cols = {c: f"{c}_pr" for c in ranks.columns}
        ranks.rename(columns=new_cols, inplace=True)

        # Join back to the original DataFrame on the same index
        df = df.join(ranks)

    elif isinstance(df, polars.DataFrame):
        if gr is None:
            # Global rank
            ranks = df.select([polars.col(vr).rank("average").over(None).alias(f"{col}_pr") for col in vr])
        else:
            ranks = df.select([polars.col(vr).rank("average").over(gr).alias(f"{col}_pr") for col in vr])

        # Optionally multiply by 100 to get 1..100-scale
        if multiply_by_100:
            ranks = ranks.with_columns([(polars.col(f"{col}_pr") * 100).alias(f"{col}_pr") for col in vr])

        # Join back to the original DataFrame
        df = df.hstack(ranks)

    return df

def prior_returns(df: pandas.DataFrame, 
                  intervals: list[tuple[int, int]],
                  group: str | list[str] = None,
                  vars: str | list[str] = None, 
                  ) -> pandas.DataFrame:
    """
    Calculates cumulative returns over specified backward-looking time intervals.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the time series data.
        intervals (list[tuple[int, int]]): A list of tuples representing the time intervals
            for which cumulative returns should be calculated. Each tuple is of the form 
            (start_lag, end_lag).
        group (str | list[str], optional): Column name(s) for grouping the data before 
            calculating the cumulative returns (e.g., for panel data). If None, the entire 
            DataFrame is treated as a single group. Defaults to None.
        vars (str | list[str], optional): Column name(s) for which cumulative returns 
            should be calculated. If None, all columns in the DataFrame are used. Defaults 
            to None.

    Returns:
        pandas.DataFrame: The input DataFrame with new columns added for each variable and 
        time interval combination, named in the format `{var}_pr{start_lag}_{end_lag}`.

    Notes:
        - The function computes cumulative returns as the product of (1 + shifted values)
          over the specified interval, minus 1.
        - If `group` is provided, calculations are performed within each group defined 
          by the specified column(s).
        - If `vars` is not specified, the function applies to all columns in the DataFrame.
        - New columns are added directly to the input DataFrame.
        - Missing values introduced by the shift operation are handled implicitly.
    """
    
    grouped = group is not None
    
    if not isinstance(df, pandas.DataFrame):
        raise TypeError(f'df: expected type pandas.DataFrame, got {type(df).__name__!r}')

    if grouped:
        # type check gr
        if not (isinstance(group, str) or isinstance(group, list)):
            raise TypeError(f'group: expected type str or list[str], got {type(group).__name__!r}')
        if isinstance(group, list):
            if not all(isinstance(x, str) for x in group):
                raise TypeError(f'group: expected a list of only strings, got {type(group).__name__!r}')
        if isinstance(group, str):
            group = [group]
        
    if vars is not None:
        # type check vr
        if not (isinstance(vars, str) or isinstance(vars, list)):
            raise TypeError(f'vars: expected type str or list[str], got {type(vars).__name__!r}')
        if isinstance(vars, list):
            if not all(isinstance(x, str) for x in vars):
                raise TypeError(f'vars: expected a list of only str, got {type(vars).__name__!r}')
        if isinstance(vars, str):
            vars = [vars]
    else:
        vars = df.columns # apply to all columns
    
    for var in vars:
        for interval in intervals:
            name: str = f'{var}_pr{interval[0]}_{interval[1]}'
            df[name] = 1
            for i in range(interval[0], interval[1] + 1): 
                if grouped:
                    df[name] *= 1 + df.groupby(by = group)[var].shift(i)
                else:
                    df[name] *= 1 + df[var].shift(i)
            df[name] -= 1
    return df

def date_intervals(
    min_date: datetime.datetime,
    max_date: datetime.datetime,
    overlap: bool = False,
    **kwargs
) -> list[tuple[datetime.datetime, datetime.datetime]]:
    """
    Generate a list of date intervals between a minimum and maximum date with configurable overlap.

    Parameters:
    - min_date (datetime): The starting date of the intervals.
    - max_date (datetime): The ending date of the intervals.
    - overlap (bool): Whether the intervals should overlap. Default is False.
    - **kwargs: Additional arguments specifying the length of each interval,
      compatible with `dateutil.relativedelta.relativedelta` (e.g., months=1, years=1).

    Returns:
    - list[tuple[datetime, datetime]]: A list of tuples, where each tuple represents
      a date interval (start_date, end_date).

    Behavior:
    - The function creates intervals starting from `min_date` and incrementing by the specified `kwargs`.
    - If `overlap` is False, each interval's end date is adjusted to exclude overlap with the next interval.
    - The last interval always ends at `max_date`, regardless of the specified interval size.
    """
    blocks = []
    start_date = min_date

    while True:
        end_date = start_date + dateutil.relativedelta.relativedelta(**kwargs)
        if end_date >= max_date:
            end_date = max_date
            blocks.append((start_date, end_date))
            break

        end_date_adj = end_date
        if not overlap:
            end_date_adj -= dateutil.relativedelta.relativedelta(days=1)
        blocks.append((start_date, end_date_adj))
        start_date = end_date

    return blocks

def df_normalize(df: typing.Union[pandas.DataFrame, polars.DataFrame],
                 vr: typing.Union[str, list[str]],
                 gr: typing.Optional[typing.Union[str, list[str]]] = None, 
                 method: str = 'log',
                 **kwargs
                ) -> typing.Union[pandas.DataFrame, polars.DataFrame]:
    """
    Normalize specified columns in a DataFrame using various methods.

    Parameters
    ----------
    df : typing.Union[pandas.DataFrame, polars.DataFrame]
        The input DataFrame to normalize. Supports pandas and polars DataFrames.
    
    vr : typing.Union[str, list[str]]
        Specific columns to normalize. Can be a single column name or a list 
        of column names.
    
    gr : typing.Optional[typing.Union[str, list[str]]] = None
        Column(s) to group by before applying normalization. Can be a single 
        column name or a list of column names.
    
    method : str = 'log'
        The normalization method to use. Options are:
        - 'clip': Clips values to a specified range or quantile.
        - 'minmax': Scales values to a [0, 1] range.
        - 'zscore': Standardizes values to have zero mean and unit variance.
        - 'log': Applies a logarithmic transformation.
        - 'softmax': Applies the softmax function to values.
    
    **kwargs
        Additional arguments specific to the chosen normalization method:
        - For 'clip': Provide either 'bounds' (tuple of lower and upper bounds) 
                        or 'quantiles' (tuple of lower and upper quantiles).
        - For 'log': Provide 'base' to specify the logarithm base. 
                        Default is the natural logarithm (base e).

    Returns
    -------
    typing.Union[pandas.DataFrame, polars.DataFrame]
        The normalized DataFrame.
    
    Raises
    ------
    TypeError
        If the input DataFrame is not a pandas or polars DataFrame, 
            or if `gr` or `vr` are not of the expected types.
    
    ValueError
        If the method is not one of the allowed options, or if necessary 
            arguments for the chosen method are missing.
    
    NotImplementedError
        If the input DataFrame is a polars DataFrame, as this 
            functionality is not implemented for polars.
            
    Notes
    -----
    For 'log', the logartithmic transformation will only be applied
        to columns that are positive element-wise.
        
    For 'clip', columns where quantiles can't be computed are skipped.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1]})
    >>> df_normalize(df, method='minmax')
         A    B
      0  0.0  1.0
      1  0.25 0.75
      2  0.5  0.5
      3  0.75 0.25
      4  1.0  0.0
    
    >>> df_normalize(df, method='clip', bounds=(2, 4))
         A  B
      0  2  4
      1  2  4
      2  3  3
      3  4  2
      4  4  2
    """
    
    is_pandas = isinstance(df, pandas.DataFrame)
    is_polars = isinstance(df, polars.DataFrame)
    if(not (is_pandas or is_polars)):
        raise TypeError(f'df: expected type pandas.DataFrame or polars.DataFrame, got {type(df).__name__!r}')

    if(gr is not None):
        # type check gr
        if(not (isinstance(gr, str) or isinstance(gr, list))):
            raise TypeError(f'gr: expected type str or list[str], got {type(df).__name__!r}')
        if(isinstance(gr, list)):
            if(not all(isinstance(x, str) for x in gr)):
                raise TypeError(f'gr: expected a list of only strings, got {type(gr).__name__!r}')
        if(isinstance(gr, str)):
            gr = [gr]
        
    if(vr is not None):
        # type check vr
        if(not (isinstance(vr, str) or isinstance(vr, list))):
            raise TypeError(f'gr: expected type str or list[str], got {type(vr).__name__!r}')
        if(isinstance(vr, list)):
            if(not all(isinstance(x, str) for x in vr)):
                raise TypeError(f'gr: expected a list of only str, got {type(vr).__name__!r}')
        if(isinstance(vr, str)):
            vr = [vr]
            
    # options to normalize
    #   1. Clip (value or quantile)
    #   2. minmax (x' = (x - x_min) / (x_max - x_min))
    #   3. z-score (x' = (x - mean(x)) / std(x))
    #   4. log
    #   5. softmax
        
    if(is_pandas):
        
        _df_int = df.copy()
        
        
        if(method == 'clip'):

            # quantile or value clip
            if('bounds' not in kwargs and 'quantiles' not in kwargs):
                raise ValueError(f'if `method` = \'clip\', then either `bounds` or `quantiles` must be passed.') 
            
            for _col in vr:
                
                if('quantiles' in kwargs):
                    try:
                        _lower_bound_val = numpy.quantile(
                            _df_int[_col], 
                            q = 0.5
                        )
                    except:
                        # if a quantile cant be caluclated move on
                        continue
                
                if(gr is None):

                    # cutoffs
                    if('bounds' in kwargs):
                        _lower_bound_val = kwargs['bounds'][0]
                        _upper_bound_val = kwargs['bounds'][1]
                    else:
                        _lower_bound_val = numpy.quantile(
                            _df_int[_col], 
                            q = kwargs['quantiles'][0]
                        )
                        _upper_bound_val = numpy.quantile(
                            _df_int[_col], 
                            q = kwargs['quantiles'][1]
                        )
                        
                        
                    # condtions
                    _upper_condition = _df_int[_col] > _upper_bound_val
                    _lower_condition = _df_int[_col] < _lower_bound_val
                    
                    # clip
                    _df_int[_col] = numpy.where(_upper_condition, 
                                           _upper_bound_val, 
                                           _df_int[_col])
                    _df_int[_col] = numpy.where(_lower_condition, 
                                           _lower_bound_val, 
                                           _df_int[_col])
                        
                else:
                    
                    try:
                        _qtiles = [kwargs['quantiles'][0], 
                                   kwargs['quantiles'][1]
                                  ]
                    except:
                        raise ValueError('using a groupby requires the use of quantiles')
                    
                    _dfs_to_concat = []
                    for _, group in _df_int.groupby(by = gr):

                        _lower_bound_val = numpy.quantile(
                            group[_col], 
                            q = _qtiles[0]
                        )
                        _upper_bound_val = numpy.quantile(
                            group[_col], 
                            q = _qtiles[1]
                        )
                        
                        # condtions
                        _upper_condition = group[_col] > _upper_bound_val
                        _lower_condition = group[_col] < _lower_bound_val

                        # clip
                        group[_col] = numpy.where(_upper_condition, 
                                                  _upper_bound_val, 
                                                  group[_col])
                        group[_col] = numpy.where(_lower_condition, 
                                                  _lower_bound_val, 
                                                  group[_col])
                        
                        _dfs_to_concat.append(group)
                    
                    _df_int = pandas.concat(_dfs_to_concat)

            return(_df_int)

        elif(method == 'minmax' or method == 'zscore'):
            
            _minmax = lambda x: (x - x.min()) / (x.max() - x.min())
            _zscore = lambda x: (x - x.mean()) / (x.std())
            
            _func_to_apply = _minmax if(method == 'minmax') else _zscore
            
            if(gr is None):
                _df_int[vr] = _df_int[vr].apply(_func_to_apply)
            else: 
                
                # create index for merging purposes
                curr_index = _df_int.index
                _df_int = _df_int.reset_index(drop = True)
                
                _tmp = _df_int.groupby(by = gr)[vr].apply(_func_to_apply)
                _tmp = _tmp.reset_index(drop = False)
                _tmp = _tmp.drop(columns = gr)
                _tmp = _tmp.set_index('level_1')
                _df_int = _df_int.drop(columns = vr)
                _df_int = _df_int.merge(_tmp, 
                              left_index = True, 
                              right_index = True
                            )

                # replace old index
                _df_int.index = curr_index
                
            return(_df_int)
            
        elif(method == 'log'):
            
            n = kwargs['base'] if('base' in kwargs) else math.e
            
            # only apply to columns that are all positive
            _ge_zero_cols = list(
                _df_int[vr].loc[:, _df_int[vr].gt(0).all()].columns
            )
            vr = list_inter(vr, _ge_zero_cols)
        
            # compute log base n of columns    
            _df_int[vr] = numpy.emath.logn(n, _df_int[vr])
            
            return(_df_int)
            
        elif(method == 'softmax'):
            
            if(gr is None):
                _df_int[vr] = _df_int[vr].transform(
                    scipy.special.softmax
                )
            else:
                _df_int[vr] = _df_int.groupby(
                    by = gr
                )[vr].transform(
                    scipy.special.softmax
                )
                
            return(_df_int)
            
        else:
            raise ValueError(f'options for `method` are [\'clip\', \'minmax\', \'zscore\', \'log\', \'softmax\'], got {method}')
    elif(is_polars):
        raise NotImplementedError('`df_normalize` has not been implemented for polars.DataFrames')
    else:
        raise ValueError('rah')

def group_quantile(df: pandas.DataFrame | polars.DataFrame,
                   qtiles: int | list[float] | dict[str, list[float]],
                   gr: typing.Optional[typing.Union[str, list[str]]] = None, 
                   vr: typing.Optional[typing.Union[str, list[str]]] = None, 
                   interpolation: typing.Optional[str] = 'linear',
                   no_merge: typing.Optional[bool] = False,
                   set_index: typing.Optional[typing.Union[str, list[str]]] = None
                ) -> pandas.DataFrame | polars.DataFrame:
    
    """
    Computes quantiles for specified variables in a DataFrame, with optional grouping.

    Parameters:
    -----------
    df : pandas.DataFrame | polars.DataFrame
        The input DataFrame (either pandas or polars) containing data for quantile calculation.
    qtiles : int | list[float] | dict[str, list[float]]
        Specifies quantile thresholds:
            - int: Number of equal-sized quantile intervals.
            - list[float]: Custom quantile thresholds (e.g., [0.25, 0.5, 0.75]).
            - dict[str, list[float]]: Maps column names to custom quantile thresholds.
    gr : str | list[str], optional
        Column(s) to group by before calculating quantiles. If None, computes global quantiles.
    vr : str | list[str], optional
        Column(s) for which quantiles are calculated. Must be provided if `qtiles` is not a dictionary.
    interpolation : str, optional
        Interpolation method for quantile computation in polars (default is 'linear').
    no_merge : bool, optional
        If True, returns the quantile results without merging back to the original DataFrame.
    set_index : (str | list[str]), optional
        If given, the index of the returned DataFrame will be set to these columns.

    Returns:
    --------
    pandas.DataFrame | polars.DataFrame
        A DataFrame (pandas or polars) containing computed quantile values. If `grouped` is True,
        the result includes group-level quantiles; otherwise, it includes global quantiles.

    Raises:
    -------
    TypeError
        If input types do not conform to the expected types or missing required arguments.

    Notes:
    ------
    1. Supports both pandas and polars DataFrames.
    2. For grouped computations, the output includes quantiles for each group defined by `gr`.
    3. If `qtiles` is an int, quantiles are calculated as evenly spaced intervals (e.g., quartiles for 4).
    4. The `no_merge` option allows returning standalone quantile results without joining them back.
    5. Ensures robust type-checking to validate inputs and prevent errors.
    """

    # type check df
    if not (isinstance(df, pandas.DataFrame) or isinstance(df, polars.DataFrame)):
        raise TypeError(f'df: expected type pandas.DataFrame or polars.DataFrame, got {type(df).__name__!r}')
    
    if not (isinstance(qtiles, list) or isinstance(qtiles, int) or isinstance(qtiles, dict)):
        raise TypeError(f'qtiles: expected type int, list[float], or dict[str, list[float]], got {type(qtiles).__name__!r}')
    
    if not isinstance(qtiles, dict):
        if vr is None:
            raise TypeError(f'vr: if qtiles is not a dictionary vr must be provided.')
    
    if not (isinstance(vr, dict) or isinstance(vr, list)):
        raise TypeError(f'vr: expected type str or list[str], got {type(vr).__name__!r}')
    
    grouped: bool = gr is not None
    if grouped:
        if not (isinstance(gr, str) or isinstance(gr, list)):
            raise TypeError(f'gr: expected type str or list[str], got {type(gr).__name__!r}')
        if isinstance(gr, str):
            gr = [gr]
            
    dict_in: dict = {}
    if isinstance(qtiles, dict):
        dict_in = qtiles
    elif isinstance(qtiles, int):
        tiles: list = [i / qtiles for i in range(1, qtiles)]
        for var in vr:
            dict_in[var] = tiles
    else:
        for var in vr:
            dict_in[var] = qtiles
                
    # compute quantiles
    is_polars = isinstance(df, polars.DataFrame)
    is_pandas = isinstance(df, pandas.DataFrame)

    if(is_pandas):
        res = []
        for var, qtiles in dict_in.items():
            ptiles = [f'{int(100 * q)}%' for q in qtiles]
            if(grouped):
                comp = df.groupby(by = gr)[var].describe(percentiles = qtiles)
                comp = comp[ptiles]
                comp = comp.add_prefix(f'{var}_')
                res.append(comp)
            else:
                comp = df[var].describe(percentiles = qtiles)
                comp = comp.to_frame()    
                comp = comp.loc[ptiles]
                res.append(comp)
        if(grouped):
            fin = functools.reduce(lambda x, y: pandas.merge(x, y, 
                                                             right_index = True, 
                                                             left_index = True), 
                                   res
                                   )
            fin = fin.reset_index(drop = False)
            if(no_merge):
                if(set_index is not None):
                    if(isinstance(set_index, str)):
                        set_index = [set_index]
                    if(set_index):
                        fin = fin.set_index(set_index)
                        return(fin)
                return(fin)
            else:
                fin = df.merge(fin, how = 'inner', on = gr)
                if(set_index is not None):
                    if(isinstance(set_index, str)):
                        set_index = [set_index]
                    if(set_index):
                        fin = fin.set_index(set_index)
                        return(fin)
                return(fin)
        else:
            fin = functools.reduce(lambda x, y: pandas.merge(x, y, 
                                                             how = 'outer', 
                                                             right_index = True, 
                                                             left_index = True), 
                                   res
                                   )
            return(fin)
    elif(is_polars):
        res_list = []
        for var, qtiles in dict_in.items():
            if(grouped):
                for q in qtiles:
                    comp = df[gr + [var]].group_by(gr).quantile(q, 
                                                                interpolation = interpolation)
                    comp = comp.rename(mapping = {var: f'{var}_{int(100 * q)}%'})
                    comp = comp.to_pandas()
                    res_list.append(comp)
            else:
                res_dict = {}
                for q in qtiles:
                    comp = df[var].quantile(q, interpolation = interpolation)
                    res_dict[f'{int(100 * q)}%'] = [comp]
                res = pandas.DataFrame(res_dict)
                res = res.transpose()
                res = res.rename(columns = {0: var})
                res = res_list.append(res)

        if(grouped):
            fin = functools.reduce(lambda x, y: pandas.merge(x, y, 
                                                             how = 'inner', 
                                                             on = gr), 
                                   res_list
                                   )
            fin = fin.sort_values(by = gr)
            fin = fin.reset_index(drop = True)
            fin = polars.from_pandas(fin)
            if(no_merge):
                return(fin)
            else:
                fin = df.join(other = fin, how = 'inner', on = gr)
                return(fin)
        else:
            fin = functools.reduce(lambda x, y: pandas.merge(x, y, 
                                                             how = 'outer', 
                                                             right_index = True, 
                                                             left_index = True), 
                                   res_list
                                   )
            fin = fin.reset_index(drop = False)
            fin = fin.rename(columns = {'index': 'ptile'})
            fin = polars.from_pandas(fin)
            return(fin)
    else:
        # type error
        raise TypeError('idk how u got here fam')

def group_nunique(df: polars.DataFrame | pandas.DataFrame, 
                  gr: str | list[str], 
                  vr: str | list[str], 
                  name: typing.Optional[typing.Union[str, dict[str, str]]] = None, 
                  no_merge: typing.Optional[bool] = False, 
                  merge_how: typing.Optional[str] = 'left',
                  set_index: typing.Optional[typing.Union[str, list[str]]] = None
                ) -> polars.DataFrame | pandas.DataFrame:
        """
        Group by specified columns and calculate the number of unique values for each group.

        Parameters:
            df (polars.DataFrame or pandas.DataFrame): Input DataFrame.
            gr (str or list[str]): Grouping column(s).
            vr (str or list[str]): Column(s) on which to calculate the number of unique values.
            name (str or dict[str, str], optional): Custom name(s) for the resulting columns. 
                If a string is provided, it will be appended to the column names specified in `vr`.
                If a dictionary is provided, it should map original column names to new names.
                Defaults to None.
            no_merge (bool, optional): Whether to merge the result back to the original DataFrame. 
                on `gr`. Defaults to False.
            merge_how (str, optional): Method of merging if `no_merge` is False. 
                Possible values: 
                    Pandas: {`left`, `inner`, `outer`, `cross`, `right`}
                    Polars: {`left`, `inner`, `outer`, `cross`, `asof`, `semi`, `anti`}
                Defaults to 'left'.
            set_index (str | list[str], optional): If given, the index of the returned DataFrame will be set to these columns.

        Returns:
            polars.DataFrame or pandas.DataFrame: DataFrame with the grouped results.

        Raises:
            TypeError: If input `df` is not a pandas.DataFrame or polars.DataFrame.
            TypeError: If input `gr` is not a string or a list of strings.
            TypeError: If input `vr` is not a string or a list of strings.
            TypeError: If input `name` is not a string, a dictionary with string keys and string values, or None.
            TypeError: If input `no_merge` is not a boolean.
        """
        
        # type check df
        is_pandas = isinstance(df, pandas.DataFrame)
        if(not (is_pandas or isinstance(df, polars.DataFrame))):
            raise TypeError(f'df: expected type pandas.DataFrame or polars.DataFrame, got {type(df).__name__!r}')
        
        # type check gr
        if(not (isinstance(gr, str) or isinstance(gr, list))):
            raise TypeError(f'gr: expected type str or list[str], got {type(df).__name__!r}')
        
        # tpye check vr
        if(isinstance(gr, list)):
            if(not all(isinstance(x, str) for x in gr)):
                raise TypeError(f'gr: expected a list of only strings, got {type(gr).__name__!r}')
            
        # type check vr
        if(not (isinstance(vr, str) or isinstance(vr, list))):
            raise TypeError(f'gr: expected type str or list[str], got {type(vr).__name__!r}')
            
        if(isinstance(vr, list)):
            if(not all(isinstance(x, str) for x in vr)):
                raise TypeError(f'gr: expected a list of only str, got {type(vr).__name__!r}')
            
        if(isinstance(vr, str)):
            vr = [vr]
        
        # tpye check name
        if(name is not None):
            if(not (isinstance(name, str) or isinstance(name, dict))):
                raise TypeError(f'name: expected type str or dict[str:str], got {type(name).__name__!r}')
            
            if(isinstance(name, dict)):
                if(not all(isinstance(x, str) for x in name.keys())):
                    raise TypeError(f'name: expected all keys of type str, got {type(name).__name__!r}')
                if(not all(isinstance(x, str) for x in name.values())):
                    raise TypeError(f'name: expected all values of type str, got {type(name).__name__!r}')

        # type check no_merge     
        if(not isinstance(no_merge, bool)):
            raise TypeError(f'no_merge: expected type bool, got {type(no_merge).__name__!r}')

        names = {}
        if(isinstance(name, str)):
            # append name to column names in vr
            for col in vr:
                names[col] = f'{col}{name}'
        else:
            names = name

        if(is_pandas):
            res = df.groupby(by = gr)[vr].nunique()
            res = res.reset_index(drop = False)
            if(name is not None):
                res = res.rename(columns = names)
        else:
            res = df.group_by(gr).agg(polars.col(vr).n_unique())
            if(name is not None):
                res = res.rename(mapping = names)

        if(no_merge):
            if(set_index is not None and is_pandas):
                if(isinstance(set_index, str)):
                    set_index = [set_index]
                if(set_index):
                    fin = res.set_index(set_index)
                    return(fin)
            return(res)
        else:
            if(is_pandas):
                res = df.merge(res, how = merge_how, on = gr)
                if(set_index is not None):
                    if(isinstance(set_index, str)):
                        set_index = [set_index]
                    if(set_index):
                        fin = res.set_index(set_index)
                        return(fin)
                return(res)
            else:
                res = df.join(res, how = merge_how, on = gr)
            return(res)
        
def group_avg(df: pandas.DataFrame | polars.DataFrame, 
              gr: str | list[str], 
              vr: str | list[str], 
              wt: typing.Optional[str] = None,
              ignore_nan: typing.Optional[bool] = True,
              name: typing.Optional[typing.Union[str, list[str]]] = None,
              no_merge: typing.Optional[bool] = False,
              merge_how: typing.Optional[str] = 'left',
              suppress_output: typing.Optional[bool] = False,
              desc: typing.Optional[str] = None,
              set_index: typing.Optional[typing.Union[str, list[str]]] = None
            ) -> pandas.DataFrame | polars.DataFrame:
        """
        Compute group-wise averages (optionally weighted) on a pandas or Polars DataFrame.
    
        Parameters
        ----------
        df : pandas.DataFrame or polars.DataFrame
            Input DataFrame containing the data to aggregate.
        gr : str or list of str
            Column name or list of column names to group by.
        vr : str or list of str
            Column name or list of column names whose averages are computed.
        wt : str, optional
            Column name to use as weights for a weighted average. If None,
            an unweighted (arithmetic) mean is used. Default is None.
        ignore_nan : bool, optional
            If True (default), then groups with zero total weight fall back to 
            the unweighted mean; if False, such groups produce NaN.
        name : str or dict, optional
            If a string, appended to each output column name (e.g. `"_avg"`).
            If a dict, maps original column names to desired output names.
        no_merge : bool, optional
            If True, return only the aggregated table; if False (default), merge
            the aggregated columns back onto the original `df` on the group keys.
        merge_how : str, optional
            Merge strategy to use when `no_merge=False`. One of {'left', 'right', 
            'inner', 'outer'}. Default is 'left'.
        suppress_output : bool, optional
            If True, disable progress bars during computation. Default is False.
        desc : str, optional
            Custom description to display in the progress bar prefix. Default is None.
        set_index : str or list of str, optional
            Column name(s) to set as the index of the returned pandas DataFrame.
            Only applies when `no_merge=False` and `df` is a pandas DataFrame.
            Default is None.
    
        Returns
        -------
        pandas.DataFrame or polars.DataFrame
            If `no_merge=True`, returns a DataFrame with one row per group
            containing the computed averages. Otherwise returns the original
            `df` augmented with the new average columns.
    
        Raises
        ------
        TypeError
            If `df` is not a pandas or Polars DataFrame, or if any of the other
            parameters are of an invalid type (e.g. non-string column names,
            incorrect `name` format, etc.).
    
        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'group': ['A','A','B','B'],
        ...     'value': [10, 20, 30, 40],
        ...     'weight': [1, 2, 3, 4]
        ... })
        >>> # Unweighted group mean
        >>> group_avg(df, gr='group', vr='value')
          group  value
        0     A   15.0
        1     B   35.0
    
        >>> # Weighted group mean, with custom suffix
        >>> group_avg(df, gr='group', vr='value', wt='weight', name='_wavg')
          group  value_wavg
        0     A   16.666667
        1     B   36.666667
    
        >>> # Merge results back onto original DataFrame
        >>> merged = group_avg(df, gr='group', vr='value', wt='weight', name='_wavg')
        >>> merged.head()
          group  value  weight  value_wavg
        0     A     10       1    16.666667
        1     A     20       2    16.666667
        2     B     30       3    36.666667
        3     B     40       4    36.666667
    
        >>> # Polars example
        >>> import polars as pl
        >>> pldf = pl.DataFrame(df)
        >>> group_avg(pldf, gr='group', vr='value', wt='weight', name={'value':'avg_val'})
        shape: (4, 3)
        ┌───────┬───────┬─────────┐
        │ group ┆ value ┆ avg_val │
        │ ---   ┆ ---   ┆ ---     │
        │ str   ┆ i64   ┆ f64     │
        ╞═══════╪═══════╪═════════╡
        │ A     ┆ 10    ┆ 16.6667 │
        │ A     ┆ 20    ┆ 16.6667 │
        │ B     ┆ 30    ┆ 36.6667 │
        │ B     ┆ 40    ┆ 36.6667 │
        └───────┴───────┴─────────┘
        """
        
        # type check df
        is_pandas = isinstance(df, pandas.DataFrame)
        if(not (is_pandas or isinstance(df, polars.DataFrame))):
            raise TypeError(f'df: expected type pandas.DataFrame or polars.DataFrame, got {type(df).__name__!r}')
        
        # type check gr
        if(not (isinstance(gr, str) or isinstance(gr, list))):
            raise TypeError(f'gr: expected type str or list[str], got {type(df).__name__!r}')
        
        # tpye check gr
        if(isinstance(gr, list)):
            if(not all(isinstance(x, str) for x in gr)):
                raise TypeError(f'gr: expected a list of only strings, got {type(gr).__name__!r}')
            
        # type check vr
        if(not (isinstance(vr, str) or isinstance(vr, list))):
            raise TypeError(f'vr: expected type str or list[str], got {type(vr).__name__!r}')
            
        if(isinstance(vr, list)):
            if(not all(isinstance(x, str) for x in vr)):
                raise TypeError(f'vr: expected a list of only str, got {type(vr).__name__!r}')
            
        if(isinstance(vr, str)):
            vr = [vr]
            
        # type check wt
        if(wt is not None):
            if(not isinstance(wt, str)):
                raise TypeError(f'wt: expected type str, got {type(wt).__name__!r}')
        
        # tpye check name
        if(name is not None):
            if(not (isinstance(name, str) or isinstance(name, dict))):
                raise TypeError(f'name: expected type str or dict[str:str], got {type(name).__name__!r}')
            
            if(isinstance(name, dict)):
                if(not all(isinstance(x, str) for x in name.keys())):
                    raise TypeError(f'name: expected all keys of type str, got {type(name).__name__!r}')
                if(not all(isinstance(x, str) for x in name.values())):
                    raise TypeError(f'name: expected all values of type str, got {type(name).__name__!r}')

        # type check no_merge     
        if(not isinstance(no_merge, bool)):
            raise TypeError(f'no_merge: expected type bool, got {type(no_merge).__name__!r}')
        
        DESCRIPTION = 'Processing:'
        if(isinstance(desc, str)):
            DESCRIPTION = desc
                    
        # Weighted average
        # can be used with groupby:  df.groupby('col1').apply(wavg, 'avg_name', 'weight_name')
        # ML: corrected by ML to allow for missing values
        # AP: corrected by AP to remove RunTimeWarning double_scalars
        def _wavg_py(gr, vr, wt):
            x = gr[[vr, wt]].dropna()
            den = x[wt].sum()
            if(den == 0):
                if(not ignore_nan):
                    return(numpy.nan)
                else:
                    return(gr[vr].mean())
            else:
                return((x[vr] * x[wt]).sum() / den)

        if(name is not None):
            names = {}
            if(isinstance(name, str) and isinstance(vr, list)):
                # append name to column names in vr
                for col in vr:
                    names[col] = f'{col}{name}'
            if(isinstance(name, dict)):
                names = name    
        
        if(is_pandas):
            # wt is None equal weighted average
            if(wt is None):
                res = df.groupby(by = gr).mean(numeric_only = True)[vr]
                if(isinstance(res, pandas.Series)):
                    res = res.to_frame()
                res = res.reset_index(drop = False)
            else:
                dfs_to_merge = []
                for col in tqdm.tqdm(vr, disable = suppress_output):
                    res = df.groupby(by = gr).apply(_wavg_py, col, wt, include_groups = False)
                    if(isinstance(res, pandas.Series)):
                        res = res.to_frame()
                    res = res.reset_index(drop = False)
                    if(0 in list(res.columns)):
                        res = res.rename(columns = {0: col})
                    dfs_to_merge.append(res)
                res = functools.reduce(lambda x, y: pandas.merge(x, y, on = gr), 
                                       dfs_to_merge
                                    )
            if(name is not None):
                res = res.rename(columns = names)
        else:
            df = df.fill_nan(None)
            if(wt is None):
                res = df.group_by(gr, maintain_order = True).agg(polars.col(vr).mean())
            else:
                mask_wt = polars.col(wt) * polars.col(vr).is_not_null()
                wavg = (polars.col(vr) * polars.col(wt)).sum() / mask_wt.sum()
                if(not ignore_nan):
                    res = df.group_by(gr, maintain_order = True).agg(wavg)
                else:
                    res = df.group_by(gr, maintain_order = True).agg(
                        polars.when(polars.col(wt).filter(polars.col(vr).is_not_null()).sum() == 0)
                            .then(polars.col(vr).mean())
                            .otherwise(wavg))
            if(name is not None):
                res = res.rename(mapping = names)

        if(no_merge):
            if(set_index is not None and is_pandas):
                    if(isinstance(set_index, str)):
                        set_index = [set_index]
                    if(set_index):
                        fin = res.set_index(set_index)
                        return(fin)
            return(res)
        else:
            if(is_pandas):
                res = df.merge(res, how = merge_how, on = gr)
                if(set_index is not None):
                    if(isinstance(set_index, str)):
                        set_index = [set_index]
                    if(set_index):
                        fin = res.set_index(set_index)
                        return(fin)
                return(res)
            else:
                res = df.join(res, how = merge_how, on = gr)
            return(res)
        
def group_sum(df: pandas.DataFrame | polars.DataFrame, 
              gr: str | list[str], 
              vr: str | list[str], 
              wt: typing.Optional[str] = None,
              name: typing.Optional[typing.Union[str, list[str]]] = None,
              no_merge: typing.Optional[bool] = False,
              merge_how: typing.Optional[str] = 'left',
              set_index: typing.Optional[typing.Union[str, list[str]]] = None
            ) -> pandas.DataFrame | polars.DataFrame:
    """
    Groups the DataFrame by specified column(s), calculates the sum of specified column(s)
    for each group, and optionally merges the result back to the original DataFrame.

    Parameters:
    - df (pandas.DataFrame or polars.DataFrame): The input DataFrame.
    - gr (str or list[str]): The column(s) to group by.
    - vr (str or list[str]): The column(s) to sum within each group.
    - wt (str, optional): Weight column to use when aggregating.
    - name (str or dict[str, str], optional): A suffix to append to the column names in the output DataFrame
                                              or a dictionary mapping column names to their new names.
    - no_merge (bool, optional): If True, the result is not merged back to the original DataFrame.
                                 Default is False.
    - merge_how (str, optional): Method to use when merging the result back to the original DataFrame.
                                 Options are 'left', 'right', 'outer', 'inner'. Default is 'left'.
    - set_index (str or list[str], optional): If provided, sets the index of the returned DataFrame to these columns.
                                 

    Returns:
    - pandas/polars.DataFrame: The DataFrame with groups summarized by summing the specified columns,
                         optionally merged back to the original DataFrame.
    Raises:
    - TypeError: If the input types are invalid or not supported.

    Note:
    - If the input DataFrame is of type 'polars.DataFrame', it will use Polars library for aggregation,
      otherwise, it will use Pandas.
    - If 'name' is a string, it appends the name to the column names in 'vr'.
    - If 'no_merge' is True, the result DataFrame is returned without merging.
    - If 'no_merge' is False, the result DataFrame is merged back to the original DataFrame based on 'merge_how'.
    """

    # type check df
    is_pandas = isinstance(df, pandas.DataFrame)
    if(not (is_pandas or isinstance(df, polars.DataFrame))):
        raise TypeError(f'df: expected type pandas.DataFrame or polars.DataFrame, got {type(df).__name__!r}')
    
    # type check gr
    if(not (isinstance(gr, str) or isinstance(gr, list))):
        raise TypeError(f'gr: expected type str or list[str], got {type(df).__name__!r}')
    
    # tpye check vr
    if(isinstance(gr, list)):
        if(not all(isinstance(x, str) for x in gr)):
            raise TypeError(f'gr: expected a list of only strings, got {type(gr).__name__!r}')
        
    # type check vr
    if(not (isinstance(vr, str) or isinstance(vr, list))):
        raise TypeError(f'gr: expected type str or list[str], got {type(vr).__name__!r}')
        
    if(isinstance(vr, list)):
        if(not all(isinstance(x, str) for x in vr)):
            raise TypeError(f'gr: expected a list of only str, got {type(vr).__name__!r}')
        
    if(isinstance(vr, str)):
        vr = [vr]
        
    # type check wt
    if(wt is not None):
        if(not isinstance(wt, str)):
            raise TypeError(f'wt: expected type str, got {type(wt).__name__!r}')
    
    # tpye check name
    if(name is not None):
        if(not (isinstance(name, str) or isinstance(name, dict))):
            raise TypeError(f'name: expected type str or dict[str:str], got {type(name).__name__!r}')
        
        if(isinstance(name, dict)):
            if(not all(isinstance(x, str) for x in name.keys())):
                raise TypeError(f'name: expected all keys of type str, got {type(name).__name__!r}')
            if(not all(isinstance(x, str) for x in name.values())):
                raise TypeError(f'name: expected all values of type str, got {type(name).__name__!r}')
            
    # type check no_merge     
    if(not isinstance(no_merge, bool)):
        raise TypeError(f'no_merge: expected type bool, got {type(no_merge).__name__!r}')
    
    names = {}
    if(isinstance(name, str)):
        # append name to column names in vr
        for col in vr:
            names[col] = f'{col}{name}'
    else:
        names = name

    if(is_pandas):
        res = df.groupby(by = gr)[vr].sum()
        res = res.reset_index(drop = False)
        if(name is not None):
            res = res.rename(columns = names)
    else:
        df = df.fill_nan(None)
        res = df.group_by(gr).agg(polars.col(vr).sum())
        if(name is not None):
            res = res.rename(mapping = names)
        
    if(no_merge):
        if(set_index is not None and is_pandas):
            if(isinstance(set_index, str)):
                set_index = [set_index]
            if(set_index):
                fin = res.set_index(set_index)
                return(fin)
        return(res)
    else:
        if(is_pandas):
            res = df.merge(res, how = merge_how, on = gr)
            if(set_index is not None and is_pandas):
                if(isinstance(set_index, str)):
                    set_index = [set_index]
                if(set_index):
                    fin = res.set_index(set_index)
                    return(fin)
            return(res)
        else:
            res = df.join(res, how = merge_how, on = gr)
        return(res)
    
def compound(x: pandas.Series) -> float:
    """
    Calculate the compound return of a pandas Series representing a sequence of returns.

    This function takes a pandas Series of returns as input and calculates the compound return
    by multiplying all elements of the Series together after adding 1 to each element, then
    subtracting 1 from the product.

    Parameters:
        x (pandas.Series): A pandas Series containing a sequence of returns.

    Returns:
        float: The compound return calculated from the input Series.

    Example:
        >>> import pandas as pd
        >>> returns = pd.Series([0.1, 0.05, -0.02])
        >>> compound_return = compound(returns)
        >>> print(compound_return)
        0.1276
    """
    return((1 + x).prod() - 1)

def create_lags(df: typing.Union[pandas.DataFrame, polars.DataFrame],
                vr: typing.Optional[typing.Union[str, list[str]]] = None, 
                lag: typing.Optional[typing.Union[int, dict[str, int | list[int]]]] = 1,
                gr: typing.Optional[typing.Union[str, list[str]]] = None,
                operation: typing.Optional[str] = 'lag'
            ) -> pandas.DataFrame | polars.DataFrame:
    
    """
    Create lagged, differenced, or percentage change columns for 
        a given DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame or polars.DataFrame
        Input DataFrame.
    vr : str or list[str], optional
        Name or list of names of the variables for which to create lags.
    lag : int or dict[str, int | list[int]], optional
        Lag specification. If an integer, it specifies the number of lags for all variables.
        If a dictionary, it specifies the number of lags (or list of lags) for each variable.
    gr : str or list[str], optional
        Column name or list of column names to group by when creating lags.
    operation : str, optional
        Operation to perform: 'lag' for lagging, 'diff' for differencing, or 'pct_change' for percentage change.
        Default is 'lag'.
    
    Returns:
    --------
    pandas.DataFrame or polars.DataFrame
        DataFrame with the new lagged/differenced/percentage change columns added.
    
    Raises:
    -------
    ValueError
        If both `vr` and a dictionary `lag` are specified.
        If neither `vr` nor a dictionary `lag` are specified.
        If `vr` is specified and `lag` is not an integer.
        If `operation` is not one of 'lag', 'diff', or 'pct_change'.
    TypeError
        If `df` is not of type pandas.DataFrame or polars.DataFrame.
        
    Note:
    -----
        Will compute difference and pct_change on only numeric columns.
            Nonnumeric columns can be still be lagged.
    
    Examples:
    ---------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1]})
    >>> create_lags(df, vr='A', lag=2)
       A  B  A_L1  A_L2
    0  1  5   NaN   NaN
    1  2  4   1.0   NaN
    2  3  3   2.0   1.0
    3  4  2   3.0   2.0
    4  5  1   4.0   3.0
    """
     
    _is_pandas = isinstance(df, pandas.DataFrame)
    _is_polars = isinstance(df, polars.DataFrame)
    _grouped = gr is not None

    if(vr is not None and isinstance(lag, dict)):
        raise ValueError('vr and dictionary lag can not both be specified.')
    
    if(vr is None and (lag is None or isinstance(lag, int))):
        raise ValueError('one of vr or dictionary lag must be specified.')
    
    if(vr is not None and not isinstance(lag, int)):
        raise ValueError(f'if vr is specified then lag must be of type int, got {type(lag).__name__!r}')
        
    if(operation != 'diff' and operation != 'pct_change' and operation != 'lag'):
        raise ValueError(f'operation expected `diff`, `pct_change`, or `lag`, got {operation}')
    
    if(isinstance(vr, str)):
        vr = [vr]
        
    _var_int = vr if(vr is not None) else list(lag.keys())

    _lag_int = {}
    for var in _var_int:
        if(isinstance(lag, dict)):
            if(isinstance(lag[var], int)):
                _lag_int[var] = list(range(1, lag[var] + 1))
            else:
                _lag_int[var] = lag[var]
        else:
            _lag_int[var] = list(range(1, lag + 1))
    
    _is_lags = operation == 'lag'
    _is_diff = operation == 'diff'
    _is_pctc = operation == 'pct_change'
        
    _dfs_to_concat = []
    for var in _var_int:
        
        # only diff and pct_change to numeric columns
        if(_is_diff or _is_pctc):
            if(_is_pandas):
                __is_numeric = pandas.api.types.is_numeric_dtype(df[var])
            else:
                __is_numeric = df[var].is_numeric()
            
            if(not __is_numeric): continue
        
        for lg in _lag_int[var]:    
            
            # create new column name
            _new_col_name = None
            if(_is_lags):
                _new_col_name = f'{var}_L{lg}'
            elif(_is_diff):
                _new_col_name = f'{var}_D{lg}'
            else:
                # pct_change
                _new_col_name = f'{var}_PC{lg}'
                        
            if(_is_pandas):
                
                _tmp = None
                
                if(_grouped):
                    if(_is_lags):    
                        _tmp = df.groupby(by = gr)[var].shift(lg)
                    elif(_is_diff):
                        _tmp = df.groupby(by = gr)[var].diff(lg)
                    else:
                        _tmp = df.groupby(by = gr)[var].pct_change(lg)
                else:
                    if(_is_lags):    
                        _tmp = df[var].shift(lg)
                    elif(_is_diff):
                        _tmp = df[var].diff(lg)
                    else:
                        _tmp = df[var].pct_change(lg)
    
                # concat instead of insert to avoid fragmentation                
                _tmp = _tmp.to_frame()
                _tmp = _tmp.rename(columns = {var: _new_col_name})
                _dfs_to_concat.append(_tmp)
                
                
            elif(_is_polars):
                if(_grouped):
                    if(_is_lags):
                        df = df.with_columns(
                            polars.col(var).
                            shift(lg).
                            over(gr).
                            alias(_new_col_name)
                        )
                    elif(_is_diff):
                        df = df.with_columns(
                            polars.col(var).
                            diff(lg).
                            over(gr).
                            alias(_new_col_name)
                        )
                    else:
                        df = df.with_columns(
                            polars.col(var).
                            pct_change(lg).
                            over(gr).
                            alias(_new_col_name)
                        )
                                      
                else:
                    if(_is_lags):
                        df = df.with_columns(
                            polars.col(var).
                            shift(lg).
                            alias(_new_col_name)
                        )
                    elif(_is_diff):
                        df = df.with_columns(
                            polars.col(var).
                            diff(lg).
                            alias(_new_col_name)
                        )
                    else:
                        df = df.with_columns(
                            polars.col(var).
                            pct_change(lg).
                            alias(_new_col_name)
                        )
            else:
                raise TypeError(f'df: expected type pandas.DataFrame or polars.DataFrame, got {type(df).__name__!r}')
    
    if(_is_pandas):
        df = pandas.concat([df, *_dfs_to_concat], axis = 1)
    
    return(df)
    
def group_score(df: pandas.DataFrame | polars.DataFrame,
                gr: str | list[str] = None,
                vr: typing.Optional[typing.Union[str, list[str]]] = None,
                q: typing.Optional[typing.Union[int, dict[str, typing.Union[int, list[float]]]]] = 2,
                **kwargs
            ) -> pandas.DataFrame | polars.DataFrame:
    
    """
    Calculate group scores based on quantiles of specified variables.

    This function calculates group scores based on quantiles of specified variables 
        in a pandas DataFrame or a polars DataFrame.
    Group scores are assigned based on the quantiles, with each group assigned a score f
        rom 1 to the number of quantiles.

    Parameters:
        df (pandas.DataFrame or polars.DataFrame): The DataFrame containing the variables.
        gr (str or list of str, optional): The variable(s) to group by before calculating 
            group scores. Default is None.
        vr (str or list of str, optional): The name(s) of the variable(s) for which group 
            scores will be calculated. Default is None.
        q (int or dict, optional): The number of quantiles to divide the variable(s) into. 
            If a dict is provided, it should map variable names to either the number of 
            quantiles or a list of quantile boundaries. Default is 2.
        **kwargs: Additional keyword arguments to be passed to pandas.qcut or polars.qcut.

    Returns:
        pandas.DataFrame or polars.DataFrame: The DataFrame with group scores added.

    Raises:
        ValueError: If both 'vr' and dictionary 'q' are specified.
        TypeError: If 'df' is not a pandas.DataFrame or polars.DataFrame, or if keyword 
            arguments are incompatible with pandas.qcut or polars.qcut.

    Example:
        >>> import pandas as pd
        >>> data = {'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]}
        >>> df = pd.DataFrame(data)
        >>> df_grouped = group_score(df, vr=['A', 'B'], q=3)
        >>> print(df_grouped)
           A   B A_scr3 B_scr3
        0  1  10      1      1
        1  2  20      2      2
        2  3  30      3      3
        3  4  40      3      3
        4  5  50      3      3
    """

    is_pandas = isinstance(df, pandas.DataFrame)
    is_polars = isinstance(df, polars.DataFrame)
    grouped = gr is not None

    if(vr is not None and isinstance(q, dict)):
        raise ValueError('vr and dictionary quantiles can not both be specified.')
    
    if(isinstance(vr, str)):
        vr = [vr]

    vr_int = vr if(vr is not None) else list(q.keys())

    quantiles_int = {}
    if(isinstance(q, int)):
        for var in vr_int:
            quantiles_int[var] = q
    else:
        quantiles_int = q

    for var in vr_int:
        tmp_q = quantiles_int[var]
        if(is_pandas):
            n = tmp_q if(isinstance(tmp_q, int)) else (len(tmp_q) - 1)
            labels_int = range(1, n + 1)
            try:
                if(grouped):
                    df[f'{var}_scr{n}'] = df.groupby(by = gr)[var].transform(
                        lambda g: pandas.qcut(g, 
                                              q = tmp_q,
                                              labels = labels_int,
                                              **kwargs
                                           )
                                        )
                else:
                    df[f'{var}_scr{n}'] = pandas.qcut(df[var], 
                                                      q = tmp_q,
                                                      labels = labels_int,
                                                      **kwargs
                                                    )
            except Exception as e:
                print(e)
                raise TypeError(f'one of the keyword arguments is incompatible with pandas.qcut or pandas.qcut raised the exception above')
        elif(is_polars):
            n = tmp_q if(isinstance(tmp_q, int)) else (len(tmp_q) + 1)
            labels_int = range(1, n + 1)
            labels_int = [str(i) for i in labels_int]
            try:
                if(grouped):
                    df = df.with_columns(polars.col(var).qcut(quantiles = tmp_q, 
                                                              labels = labels_int,
                                                              **kwargs
                                                            ).over(gr).alias(f'{var}_scr{n}'))
                else:
                    df = df.with_columns(polars.col(var).qcut(quantiles = tmp_q, 
                                                              labels = labels_int,
                                                              **kwargs
                                                            ).alias(f'{var}_scr{n}'))
                df = df.cast({f'{var}_scr{n}': polars.Int64})
            except Exception as e:
                print(e)
                raise TypeError(f'one of the keyword arguments is incompatible with polars.qcut or polars.qcut raised the exception above')
        else:
            raise TypeError(f'df: expected type pandas.DataFrame or polars.DataFrame, got {type(df).__name__!r}')
    return(df)

def lh_regression(endog: typing.Union[list, numpy.ndarray, pandas.Series], 
                  exog: typing.Union[list, numpy.ndarray, pandas.Series], 
                  horizon: int, 
                  se_type: typing.Union[str, list[str]],
                  add_constant: bool = True
                ) -> typing.Tuple:
    
    """
    Perform linear regression with a rolling window approach.

    Parameters:
    - endog (Union[list, numpy.ndarray, pandas.Series]): The dependent variable.
    - exog (Union[list, numpy.ndarray, pandas.Series]): The independent variable.
    - horizon (int): The number of periods to roll the window.
    - se_type (Union[str, list[str]]): Type of standard error(s) to 
        calculate.
    - add_constant (bool, optional): Whether to add a constant to 
        the independent variable. Default is True.

    Returns:
    Tuple: A tuple containing:
        - float: The estimated coefficient for the independent variable.
        - dict: A dictionary containing standard error(s) with their 
            corresponding type as keys.
        - float: The R-squared value of the regression.
        - numpy.ndarray: The residuals of the regression.

    Notes:
    - If endog or exog is a pandas Series, it will be converted to a 
        numpy array for processing.
    - The rolling window approach is implemented by differencing the 
        dependent variable over the horizon.
    - Standard error types supported:
        - 'R': Robust standard errors (HC0).
        - 'NW': Newey-West standard errors (HAC) with maxlags set to the horizon.
        - 'HH': Heteroskedasticity and autocorrelation robust standard 
            errors (HAC) with maxlags set to the horizon and uniform kernel.
        - Any other string: Assumes the string corresponds to a specific 
            covariance type.
    """
    
    if(isinstance(endog, pandas.Series)):
        endog = endog.values

    if(isinstance(endog, pandas.Series)):
        exog = exog.values

    ### k-horizon difference
    _Y =(numpy.roll(endog, -horizon) - endog)[:-horizon]

    if(add_constant):
        _X = statsmodels.api.add_constant(exog[:-horizon])

    _model_fit = statsmodels.api.OLS(_Y, _X).fit()

    ### point estimate
    _beta = _model_fit.params.iloc[-1]
    
    if(isinstance(se_type, str)):
        se_type = [se_type]

    ### SEs:
    _ses = {}
    for _se in se_type:
        if(_se == 'R'):
            _ses[_se] = _model_fit.get_robustcov_results(cov_type = 'HC0').bse[-1]
        elif(_se == 'NW'):
            _ses[_se] = _model_fit.get_robustcov_results(cov_type = 'HAC', 
                                                         maxlags = horizon
                                                        ).bse[-1]
        elif(_se == 'HH'):
            _ses[_se] = _model_fit.get_robustcov_results(cov_type = 'HAC', 
                                                         maxlags = horizon, 
                                                         kernel = 'uniform'
                                                        ).bse[-1]
        else:
            _ses[_se] = _model_fit.get_robustcov_results(cov_type = _se).bse[-1]

    # R^2
    _r2 = _model_fit.rsquared

    # Resiudals
    _resid = _model_fit.resid
    
    return(_beta, _ses, _r2, _resid)

def long_horizon_regssions(endog: typing.Union[list, numpy.ndarray, pandas.Series], 
                           exog: typing.Union[list, numpy.ndarray, pandas.Series], 
                           se_type: typing.Union[str, list[str]],
                           horizons: typing.Union[int, list[int]] = 10,
                           add_constant: bool = True,
                           return_resid: bool = False
                        ) -> pandas.DataFrame:
    
    """
    Perform linear regression with a rolling window approach for multiple horizons.

    Parameters:
    - endog (Union[list, numpy.ndarray, pandas.Series]): The dependent variable.
    - exog (Union[list, numpy.ndarray, pandas.Series]): The independent variable.
    - se_type (Union[str, list[str]]): Type of standard error(s) to calculate.
    - horizons (Union[int, list[int]], optional): The list of horizons or maximum 
        horizon if integer provided. Default is 10.
    - add_constant (bool, optional): Whether to add a constant to the independent 
        variable. Default is True.
    - return_resid (bool, optional): Whether to return residuals in addition to 
        regression results. Default is False.

    Returns:
    pandas.DataFrame: A DataFrame containing regression results for each horizon.
    
    If return_resid is True, a tuple containing the DataFrame and a dictionary 
        of residuals for each horizon is returned.
    """
    
    if(isinstance(horizons, int)):
        horizons = list(range(1, horizons + 1))

    res = pandas.DataFrame(index = horizons)
    resids = {}
    for k in horizons:
        _b, _ses, _r2, _resid = lh_regression(endog = endog, 
                                              exog = exog, 
                                              horizon = k,
                                              se_type = se_type, 
                                              add_constant = add_constant
                                            )
        res.loc[k, 'b'] = _b
        for _se, _values in _ses.items():
            res.loc[k, _se] = _values
        res.loc[k, 'R2'] = _r2
        resids[k] = _resid

    if(return_resid):
        return(res, resids)
    else:
        return(res)
    
def list_dir(path: typing.Union[str, pathlib.Path],
             only_dir: bool = False,
             extension: str = None,
             only_files: bool = True,
             keep_hidden: bool = False,
             absolute_paths: bool = False
            ) -> list[typing.Union[str, pathlib.Path]]:
    """
    List directories and/or files in a given path with optional filtering.

    Parameters:
    - path (Union[str, pathlib.Path]): The path to list contents from.
    - only_dir (bool, optional): Whether to list only directories. Default is False.
    - extension (str, optional): Filter files by extension. Default is None.
    - only_files (bool, optional): Whether to list only files. Default is True.
    - keep_hidden (bool, optional): Whether to include hidden files. Default is False.
    - absolute_paths (bool, optional): Whether to return absolute paths. Default is False.

    Returns:
    list[str]: A list of directory and/or file names.

    Notes:
    - If only_dir is True, only directories will be included.
    - If only_files is True, only files will be included.
    - If keep_hidden is True, hidden files will be included.
    - If extension is provided, only files with that extension will be included.
    - If absolute_paths is True, the full paths of directories and/or files will be returned.
    """
    
    path = pathlib.Path(path)
    path = path.resolve()
    _tmp = os.listdir(path)

    res = []
    for _obj in _tmp:
        _obj_path = path / _obj
        if(_obj_path.is_dir() and only_dir):
            res.append(_obj)
        elif(_obj_path.is_file() and only_files):
            if(keep_hidden or not _obj_path.stem.startswith('.')):
                if(extension is None or _obj_path.suffix == extension):
                    res.append(_obj)
    if(absolute_paths):
        res = [path / f for f in res]

    return(res)

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

def columns_to_lower(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    Convert column names of a DataFrame to lowercase and strip 
        trailing/leading whitespace.

    Parameters:
    - df (pd.DataFrame): The DataFrame whose column names are to be converted.

    Returns:
    pd.DataFrame: The DataFrame with lowercase column names.

    Example:
    >>> import pandas as pd
    >>> data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
    >>> df = pd.DataFrame(data)
    >>> df = columns_to_lower(df)
    >>> print(df.columns)
    Index(['a', 'b'], dtype='object')
    """
    cols = df.columns
    cols = [col.lower().strip() for col in cols]
    df.columns = cols
    return(df)

def adf_test(input: typing.Union[pandas.DataFrame, numpy.typing.ArrayLike],
             vr: typing.Union[str, list[str]] = None,
             signif_level: str = '5%',
             print_null: bool = True,
             **kwargs
            ) -> pandas.DataFrame:
    
    """
    Perform Augmented Dickey-Fuller test for unit root.

    Parameters:
    - input (pandas.DataFrame or numpy.ndarray): The input data, either a pandas DataFrame or a numpy array.
    - vr (str or list of str, optional): The column(s) of the DataFrame to analyze. Required if input 
        is a DataFrame.
    - signif_level (str, optional): The significance level to use for the test. Default is '5%'. 
        Options are '1%', '5%', or '10%'.
    - print_null (bool, optional): Whether to print the null hypothesis. Default is True.
    - **kwargs: Additional keyword arguments to pass to statsmodels.tsa.stattools.adfuller().

    Returns:
    - pandas.DataFrame: A DataFrame containing the test results.

    Notes:
    - If the input is a pandas DataFrame, vr must be specified.
    - The 'Conclusion' column in the returned DataFrame indicates whether to reject or fail to reject 
        the null hypothesis.

    Example:
    >>> import pandas as pd
    >>> from statsmodels.tsa.stattools import adfuller
    >>> data = {'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1]}
    >>> df = pd.DataFrame(data)
    >>> adf_test(df, vr='A')
    """

    is_pandas = False
    if(isinstance(input, pandas.DataFrame)):
        is_pandas = True
        if(vr is None):
            raise ValueError('if input is type pandas.DataFrame vr must be specified')
        
    if(isinstance(vr, str)):
        vr = [vr]

    res = pandas.DataFrame()
    if(is_pandas):
        for v in vr:
            _in = input[v].dropna()
            output = statsmodels.tsa.stattools.adfuller(_in, **kwargs)
            res.loc[v, 'Test Stat.'] = output[0]
            res.loc[v, 'p-value'] = output[1]
            res.loc[v, 'Used Lag'] = int(output[2])
            res.loc[v, 'N Obs'] =    int(output[3])
            res.loc[v, '1%'] = output[4]['1%']
            res.loc[v, '5%'] = output[4]['5%']
            res.loc[v, '10%'] = output[4]['10%']
            res.loc[v, 'Information'] = output[5]
            res.loc[v, 'Conclusion'] = 'Reject' if(res.loc[v, 'Test Stat.'] < res.loc[v, signif_level]) else 'Fail to Reject'
    else:
        output = statsmodels.tsa.stattools.adfuller(input, **kwargs)
        res['Test Stat.'] =  [output[0]]
        res['p-value'] =     [output[1]]
        res['Used Lag'] =    [int(output[2])]
        res['N Obs'] =       [int(output[3])]
        res['1%'] =          [output[4]['1%']]
        res['5%'] =          [output[4]['5%']]
        res['10%'] =         [output[4]['10%']]
        res['Information'] = [output[5]]
        res['Conclusion'] = 'Reject' if(res['Test Stat.'].values[0] < res[signif_level].values[0]) else 'Fail to Reject'

    if(print_null):
        print('H0: A unit root is present in a time series sample.')

    return(res)

def pca(X: typing.Union[pandas.DataFrame, numpy.typing.NDArray],
        vr: list[str] = None,
        disp_corr: bool = False):
    """
    Perform Principal Component Analysis (PCA) on the given data.

    Parameters:
    - X (pandas.DataFrame or numpy.ndarray): The input data, either a pandas DataFrame 
        or a numpy array.
    - vr (list of str, optional): The column names to consider for PCA. 
        Required if input is a DataFrame.

    Returns:
    - ResultsClasses.PCAResults: PCA results including eigenvalues, 
        eigenvectors, principal components, and variance explained.

    Notes:
    - If the input is a pandas DataFrame, vr must be specified.
    - Assumes X is (N x P) where N is the number of observations and P is the number of features.
    - PCA is performed on the correlation matrix via eigenvalue decomposition.

    Example:
    >>> from FinToolsAP.UtilityFunctions import pca
    >>> numpy.random.seed(1)
    >>> X = numpy.random.normal(size = (100, 3))
    >>> results = pca(X)
    >>> results.variance_explained
    >>> [0.37271793 0.33867340 0.28860867]
    """

    class PCAResults:

        def __init__(self, L, Q, PC, var_expl, cum_var) -> None:
        
            self.eigenvalues =          numpy.diag(L)
            self.eigenvectors =         Q
            self.cumm_var_explained =   cum_var
            self.variance_explained =   var_expl
            self.principal_components = PC

        def __str__(self) -> str:
            content = {'Eigenvalues': list(self.eigenvalues), 
                       'Var. Explained': self.variance_explained,
                       'Cummulative Var. Explained': self.cumm_var_explained
                    }
            _res = pandas.DataFrame(content)
            return(_res.to_string())

    _data = None

    if(isinstance(X, pandas.DataFrame)):
        if(vr is None):
            raise ValueError('if input is type pandas.DataFrame vr must be specified')
        _data = X[vr].to_numpy()
    else:
        _data = X

    # preform PCA on the correlation matrix via eigenvalue decomposition

    # calculatec correlation matrix
    corr_mat = numpy.corrcoef(_data, rowvar = False)
    
    if(disp_corr):
        print(corr_mat)
    
    # calculate eigenvalue decomposition A = QLQ^{-1}
    eigenvalues, Q = numpy.linalg.eig(corr_mat)
    eigenvalues = numpy.real(eigenvalues)
    Q = numpy.real(Q)
    L = numpy.diag(eigenvalues)

    # principal compents
    PC = numpy.dot(_data, Q)

    # variance explained
    variance_explained = numpy.zeros(L.shape[0])
    _sum_of_egienvalues = numpy.sum(L)
    for i, l in enumerate(sorted(numpy.diag(L), reverse = True)):
        variance_explained[i] = l / _sum_of_egienvalues

    cumm_var_explained = numpy.cumsum(variance_explained)

    return(PCAResults(L, Q, PC, variance_explained, cumm_var_explained))

def list_diff(list1: typing.List[typing.Any], 
              list2: typing.List[typing.Any]
            ) -> list:
    """
    Compute the difference between two lists.

    This function returns a new list that contains 
        the elements from `list1` that are not 
        present in `list2`.

    Args:
        list1 (List[Any]): The first list from which 
                            elements will be taken.
        list2 (List[Any]): The second list whose 
                            elements will be excluded from the result.

    Returns:
        list: A list containing elements from `list1` 
            that are not in `list2`.

    Example:
        >>> list_diff([1, 2, 3, 4], [2, 4])
        [1, 3]
        
        >>> list_diff(['a', 'b', 'c'], ['b'])
        ['a', 'c']
    """
    res = [e for e in list1 if e not in list2]
    return(res)

def list_inter(list1: typing.List[typing.Any], 
               list2: typing.List[typing.Any]
            ) -> list:
    """
    Compute the intersection of two lists.

    This function returns a new list that contains the 
        elements present in both `list1` and `list2`.

    Args:
        list1 (List[Any]): The first list to be compared.
        list2 (List[Any]): The second list to be compared.

    Returns:
        list: A list containing elements that are common 
            to both `list1` and `list2`.

    Example:
        >>> list_inter([1, 2, 3, 4], [2, 4])
        [2, 4]
        
        >>> list_inter(['a', 'b', 'c'], ['b', 'd'])
        ['b']
    """
    res = [e for e in list1 if e in list2]
    return(res)
    
def sort_portfolios(df: pandas.DataFrame,
                    id_col: str, 
                    date_col: str,
                    return_col: str,
                    sorting_funcs: dict[str, typing.Callable],
                    breakpoints: dict[str, list[float]] | pandas.DataFrame,
                    weight_col: str = None,
                    rebalance_freq: str = 'A',
                    sort_month: int = 7, # July
                    drop_na: bool = False,
                    suppress_output: bool = False
                ) -> pandas.DataFrame:
    
    # only keep necessary columns
    
    _df_internal = df.copy()
    
    if sorting_funcs.keys() != breakpoints.keys():
        raise ValueError('keys of sorting_funcs and breakpoints must match.')
    
    needed_cols: list = [id_col, date_col, return_col]
    if weight_col is not None and weight_col not in sorting_funcs.keys():
        needed_cols += [weight_col]
    needed_cols += list(set(sorting_funcs.keys()))
    _df_internal = _df_internal[needed_cols]
    _df_internal = _df_internal.sort_values(by = [date_col, id_col])
    _df_internal = _df_internal.dropna()
        
    # cast date column to datetime64[ns]
    try: 
        _df_internal[date_col] = pandas.to_datetime(_df_internal[date_col], format = 'ISO8601')
    except Exception as e:
        print(e)
        raise TypeError('unable to cast `date_col` to datetime with format ISO8601.')
        
    # when to sort
    _df_internal['sort_month'] = _df_internal[date_col].dt.month    
    if(rebalance_freq == 'A'):
        rebalance_df = _df_internal[df.sort_month == sort_month]
        if rebalance_df.empty:
            raise ValueError(f'No data available for sorting in month {sort_month}.')
    else:
        rebalance_df = _df_internal
    
    # calculate breakpoints
    breakpoints_df = None
    if isinstance(breakpoints, dict):
        breakpoints_df = group_quantile(
            df = _df_internal,
            gr = date_col,
            vr = list(sorting_funcs.keys()),
            qtiles = breakpoints,
            no_merge = True
        )
    elif isinstance(breakpoints, pandas.DataFrame):
        breakpoints_df = breakpoints
    else:
        raise TypeError(f'breakpoints expected type dict[str, list[float]] or pandas.DataFrame, got {type(df).__name__!r}.')
    rebalance_df = breakpoints_df.merge(rebalance_df, how = 'inner', on = [date_col])

    # apply ranking to stocks
    rank_cols = []
    for char, func in sorting_funcs.items():
        rank_col = f'{char}_rank'
        rank_cols.append(rank_col)

        # Fast vectorized paths for built-in sorting functions; fallback to apply
        v = rebalance_df[char]

        # Initialize with '--fail' to mirror existing behavior on unmatched rows
        out = pandas.Series(numpy.full(len(rebalance_df), '--fail', dtype=object), index=rebalance_df.index)

        if func is sort_50:
            p50 = rebalance_df[f'{char}_50%']
            mask1 = v < p50
            mask2 = v >= p50
            out[mask1] = f'{char}1'
            out[mask2] = f'{char}2'
            rebalance_df[rank_col] = out

        elif func is sort_3070:
            p30 = rebalance_df[f'{char}_30%']
            p70 = rebalance_df[f'{char}_70%']
            mask1 = v < p30
            mask2 = (v >= p30) & (v < p70)
            mask3 = v >= p70
            out[mask1] = f'{char}1'
            out[mask2] = f'{char}2'
            out[mask3] = f'{char}3'
            rebalance_df[rank_col] = out

        elif func is sort_03070:
            p30 = rebalance_df[f'{char}_30%']
            p70 = rebalance_df[f'{char}_70%']
            mask1 = v <= 0
            mask2 = (v >= 0) & (v < p30)
            mask3 = (v >= p30) & (v < p70)
            mask4 = v >= p70
            out[mask1] = f'{char}1'
            out[mask2] = f'{char}2'
            out[mask3] = f'{char}3'
            out[mask4] = f'{char}4'
            rebalance_df[rank_col] = out

        elif func is sort_tercile:
            p33 = rebalance_df[f'{char}_33%']
            p66 = rebalance_df[f'{char}_66%']
            mask1 = v <= p33
            mask2 = (v > p33) & (v <= p66)
            mask3 = v > p66
            out[mask1] = f'{char}1'
            out[mask2] = f'{char}2'
            out[mask3] = f'{char}3'
            rebalance_df[rank_col] = out

        elif func is sort_quartile:
            p25 = rebalance_df[f'{char}_25%']
            p50 = rebalance_df[f'{char}_50%']
            p75 = rebalance_df[f'{char}_75%']
            mask1 = v <= p25
            mask2 = (v > p25) & (v <= p50)
            mask3 = (v > p50) & (v <= p75)
            mask4 = v > p75
            out[mask1] = f'{char}1'
            out[mask2] = f'{char}2'
            out[mask3] = f'{char}3'
            out[mask4] = f'{char}4'
            rebalance_df[rank_col] = out

        elif func is sort_quintile:
            p20 = rebalance_df[f'{char}_20%']
            p40 = rebalance_df[f'{char}_40%']
            p60 = rebalance_df[f'{char}_60%']
            p80 = rebalance_df[f'{char}_80%']
            mask1 = v <= p20
            mask2 = (v > p20) & (v <= p40)
            mask3 = (v > p40) & (v <= p60)
            mask4 = (v > p60) & (v <= p80)
            mask5 = v > p80
            out[mask1] = f'{char}1'
            out[mask2] = f'{char}2'
            out[mask3] = f'{char}3'
            out[mask4] = f'{char}4'
            out[mask5] = f'{char}5'
            rebalance_df[rank_col] = out

        elif func is sort_decile:
            p10 = rebalance_df[f'{char}_10%']
            p20 = rebalance_df[f'{char}_20%']
            p30 = rebalance_df[f'{char}_30%']
            p40 = rebalance_df[f'{char}_40%']
            p50 = rebalance_df[f'{char}_50%']
            p60 = rebalance_df[f'{char}_60%']
            p70 = rebalance_df[f'{char}_70%']
            p80 = rebalance_df[f'{char}_80%']
            p90 = rebalance_df[f'{char}_90%']
            masks = [
                v < p10,
                (v >= p10) & (v < p20),
                (v >= p20) & (v < p30),
                (v >= p30) & (v < p40),
                (v >= p40) & (v < p50),
                (v >= p50) & (v < p60),
                (v >= p60) & (v < p70),
                (v >= p70) & (v < p80),
                (v >= p80) & (v < p90),
                v >= p90,
            ]
            labels = [f'{char}{i}' for i in range(1, 11)]
            for m, lab in zip(masks, labels):
                out[m] = lab
            rebalance_df[rank_col] = out

        elif func is sort_050:
            p50 = rebalance_df[f'{char}_50%']
            mask1 = v < 0
            mask2 = (v >= 0) & (v < p50)
            mask3 = v >= p50
            out[mask1] = f'{char}1'
            out[mask2] = f'{char}2'
            out[mask3] = f'{char}3'
            rebalance_df[rank_col] = out

        else:
            # Fallback for custom callables
            rebalance_df[rank_col] = rebalance_df.apply(func, args=(char,), axis=1)

    # remove stocks that could not be sorted
    for rank_col in rank_cols:
        if('--fail' in rebalance_df[rank_col].unique()):
            if(not suppress_output):
                raise RuntimeWarning(f'There are stocks that could not be sorted by {rank_col}. They will be removed before constructing portfolios.')
            rebalance_df = rebalance_df[rebalance_df[rank_col] != '--fail']

    # create portfolio name
    rebalance_df['port_name'] = rebalance_df[rank_cols].agg('_'.join, axis = 1)

    # merge portfolio name back to input data
    fin = None
    if(rebalance_freq == 'A'):
        fin = df.merge(rebalance_df[[id_col, date_col, 'port_name']], 
                         how = 'left', 
                         on = [id_col, date_col]
                        )
    else:
        fin = rebalance_df
        
    fin = fin.sort_values(by = [id_col, date_col])
            
    # front fill portfolio name
    fin.port_name = fin.groupby(by = [id_col])[['port_name']].ffill()
        
    # create portfolio returns    
    fin = fin.dropna(subset = ['port_name'])
    rets = group_avg(df = fin, 
                     gr = [date_col, 'port_name'], 
                     vr = return_col, 
                     wt = weight_col,
                     no_merge = True,
                     suppress_output = suppress_output
                    )   
    
    # count number of firms in each portfolio 
    firm = group_nunique(fin, 
                         gr = [date_col, 'port_name'], 
                         vr = id_col, 
                         name = {id_col: 'num_firms'}, 
                         no_merge = True
                        )
    
    rets = rets.pivot(index = date_col, columns = 'port_name', values = return_col)
    firm = firm.pivot(index = date_col, columns = 'port_name', values = 'num_firms')
    firm = firm.add_suffix('_num_firms')
    res = rets.merge(firm, how = 'inner', on = [date_col])
    if(drop_na): 
        res = res.dropna()
    res = res.reset_index()
    return(res)

## Built in Sorting Functions -------------------------------------------------

def sort_50(row: pandas.Series, var: str) -> str:
    """
    Sorts a row of data into one of two groups based on whether a specified 
        variable's value is below or above its 50th percentile breakpoint.

    Args:
        - row (pd.Series): A row of data from a DataFrame.
        - var (str): The name of the variable to use for sorting.

    Returns:
        - str: A string indicating the group the row belongs to, formatted as 
                '{var}1' or '{var}2'. Returns '--fail' if sorting fails.
    """
    if(row[var] < row[f'{var}_50%']):
        res = f'{var}1'
    elif(row[var] >= row[f'{var}_50%']):
        res = f'{var}2'
    else:
        res = '--fail'
    return(res)

def sort_050(row: pandas.Series, var: str) -> str:
    """
    Sorts a row of data into one of three groups based on the value of a 
        specified variable relative to 0 and its 50th percentile breakpoint.

    Args:
        - row (pd.Series): A row of data from a DataFrame.
        - var (str): The name of the variable to use for sorting.

    Returns:
        - str: A string indicating the group the row belongs to, formatted as 
            '{var}1', '{var}2', or '{var}3'. Returns '--fail' if sorting fails.
    """
    if(row[var] < 0):
        res = f'{var}1'
    elif(row[var] >= 0 and row[var] < row[f'{var}_50%']):
        res = f'{var}2'
    elif(row[var] >= row[f'{var}_50%']):
        res = f'{var}3'
    else:
        res = '--fail'
    return(res)

def sort_3070(row: pandas.Series, var: str) -> str:
    """
    Sorts a given row based on the values of the specified variable and its 
        associated 30% and 70% thresholds.

    Args:
        - row (dict): A dictionary representing a data row.
        - var (str): The variable to be used for sorting.

    Returns:
        - str: A string indicating the sorting result:
    """
    if(row[var] < row[f'{var}_30%']):
        res = f'{var}1'
    elif(row[var] >= row[f'{var}_30%'] and row[var] < row[f'{var}_70%']):
        res = f'{var}2'
    elif(row[var] >= row[f'{var}_70%']):
        res = f'{var}3'
    else:
        res = '--fail'
    return(res)

def sort_03070(row: pandas.Series, var: str) -> str:
    """
    Sorts a given Pandas Series row based on the values of the specified 
        variable and its associated 30% and 70% thresholds.

    Args:
        - row (pd.Series): A Pandas Series representing a data row.
        - var (str): The variable to be used for sorting.

    Returns:
        - str: A string indicating the sorting result:
    """
    if(row[var] <= 0):
        res = f'{var}1'
    elif(row[var] >= 0 and row[var] < row[f'{var}_30%']):
        res = f'{var}2'
    elif(row[var] >= row[f'{var}_30%'] and row[var] < row[f'{var}_70%']):
        res = f'{var}3'
    elif(row[var] >= row[f'{var}_70%']):
        res = f'{var}4'
    else:
        res = '--fail'
    return(res)

def sort_tercile(row: pandas.Series, var: str) -> str:
    """
    Sorts a given Pandas Series row into terciles based on the values of the 
        specified variable and its associated 33% and 66% thresholds.

    Args:
        - row (pd.Series): A Pandas Series representing a data row.
        - var (str): The variable to be used for sorting.

    Returns:
        - str: A string indicating the tercile classification:
    """
    if(row[var] <= row[f'{var}_33%']):
        res = f'{var}1'
    elif(row[var] > row[f'{var}_33%'] and row[var] <= row[f'{var}_66%']):
        res = f'{var}2'
    elif(row[var] > row[f'{var}_66%']):
        res = f'{var}3'
    else:
        res = '--fail'
    return(res)

def sort_quartile(row: pandas.Series, var: str) -> str:
    """
    Sorts a given Pandas Series row into quartiles based on the values of 
        the specified variable and its associated 25%, 50%, and 75% thresholds.

    Args:
        - row (pd.Series): A Pandas Series representing a data row.
        - var (str): The variable to be used for sorting.

    Returns:
        - str: A string indicating the quartile classification:
    """
    if(row[var] <= row[f'{var}_25%']):
        res = f'{var}1'
    elif(row[var] > row[f'{var}_25%'] and row[var] <= row[f'{var}_50%']):
        res = f'{var}2'
    elif(row[var] > row[f'{var}_50%'] and row[var] <= row[f'{var}_75%']):
        res = f'{var}3'
    elif(row[var] > row[f'{var}_75%']):
        res = f'{var}4'
    else:
        res = '--fail'
    return(res)

def sort_quintile(row: pandas.Series, var: str) -> str:
    """
    Sorts a given Pandas Series row into quintiles based on the values of the 
        specified variable and its associated 20%, 40%, 60%, and 80% thresholds.

    Args:
        - row (pd.Series): A Pandas Series representing a data row.
        - var (str): The variable to be used for sorting.

    Returns:
        - str: A string indicating the quintile classification:
    """
    if(row[var] <= row[f'{var}_20%']):
        res = f'{var}1'
    elif(row[var] > row[f'{var}_20%'] and row[var] <= row[f'{var}_40%']):
        res = f'{var}2'
    elif(row[var] > row[f'{var}_40%'] and row[var] <= row[f'{var}_60%']):
        res = f'{var}3'
    elif(row[var] > row[f'{var}_60%'] and row[var] <= row[f'{var}_80%']):
        res = f'{var}4'
    elif(row[var] > row[f'{var}_80%']):
        res = f'{var}5'
    else:
        res = '--fail'
    return(res)

def sort_decile(row: pandas.Series, var: str) -> str:
    """
    Sorts a given Pandas Series row into deciles based on the values of the 
        specified variable and its associated 10%, 20%, ..., 90% thresholds.

    Args:
        - row (pd.Series): A Pandas Series representing a data row.
        - var (str): The variable to be used for sorting.

    Returns:
        - str: A string indicating the decile classification:
    """
    if(row[var] < row[f'{var}_10%']):
        res = f'{var}1'
    elif(row[var] >= row[f'{var}_10%'] and row[var] < row[f'{var}_20%']):
        res = f'{var}2'
    elif(row[var] >= row[f'{var}_20%'] and row[var] < row[f'{var}_30%']):
        res = f'{var}3'
    elif(row[var] >= row[f'{var}_30%'] and row[var] < row[f'{var}_40%']):
        res = f'{var}4'
    elif(row[var] >= row[f'{var}_40%'] and row[var] < row[f'{var}_50%']):
        res = f'{var}5'
    elif(row[var] >= row[f'{var}_50%'] and row[var] < row[f'{var}_60%']):
        res = f'{var}6'
    elif(row[var] >= row[f'{var}_60%'] and row[var] < row[f'{var}_70%']):
        res = f'{var}7'
    elif(row[var] >= row[f'{var}_70%'] and row[var] < row[f'{var}_80%']):
        res = f'{var}8'
    elif(row[var] >= row[f'{var}_80%'] and row[var] < row[f'{var}_90%']):
        res = f'{var}9'
    elif(row[var] >= row[f'{var}_90%']):
        res = f'{var}10'
    else:
        res = '--fail'
    return(res)
    






