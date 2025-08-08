from __future__ import annotations

"""
FinToolsAP.FactorModels
This module provides tools for assembling and formatting factor loading estimates
from a set of asset return regressions. It leverages pre‐computed regression outputs:
parameter estimates (`params`), their standard errors (`std_errors`), t‐statistics (`tstats`),
and p‐values (`pvalues`). The primary functionality is to produce a neatly formatted
pandas DataFrame that presents each asset’s factor loadings alongside corresponding
statistical measures—either standard errors or t‐statistics—with significance stars.
"""

__author__ = 'Andrew Maurice Perry'
__email__ = 'Andrewpe@berkeley.edu'
__date__ = '2025-01-13'
__status__ = 'Pre-production'

__all__ = ['tangency_portfolio', 'minimum_variance_portfolio', 
           'create_efficient_frontier', 'FamaMacBeth']

import numpy
import pandas
import itertools
import scipy.stats
import linearmodels.asset_pricing

def tangency_portfolio(returns: pandas.DataFrame, risk_free_rate: float = 0.0):
    """
    Calculate the tangency portfolio weights.
    
    Parameters
    ----------
    returns : pandas.DataFrame
        DataFrame of asset returns.
    risk_free_rate : float
        Risk-free rate (default is 0.0).
    
    Returns
    -------
    pandas.Series
        Weights of the tangency portfolio.
    """
    returns = returns.values
    excess_returns = returns - risk_free_rate
    expected_excess_returns = numpy.mean(excess_returns, axis=0)

    sigma = numpy.cov(returns, rowvar=False)
    sigma_inv = numpy.linalg.inv(sigma)
    
    ones = numpy.ones(shape = (sigma.shape[0], 1))

    _num = sigma_inv @ expected_excess_returns
    _den = ones.T @ sigma_inv @ expected_excess_returns
    weights = _num / _den
    
    weights = weights.reshape(-1, 1)
    
    return weights

def minimum_variance_portfolio(returns: pandas.DataFrame):
    """
    Calculate the minimum variance portfolio weights.
    
    Parameters
    ----------
    returns : pandas.DataFrame
        DataFrame of asset returns.
    
    Returns
    -------
    pandas.Series
        Weights of the minimum variance portfolio.
    """
    returns = returns.values
    sigma = numpy.cov(returns, rowvar=False)
    sigma_inv = numpy.linalg.inv(sigma)
    
    ones = numpy.ones(shape = (sigma.shape[0], 1))

    _num = sigma_inv @ ones
    _den = ones.T @ sigma_inv @ ones
    weights = _num / _den
    
    weights = weights.reshape(-1, 1)
    
    return weights

def create_efficient_frontier(returns: pandas.DataFrame, 
                              rf: float = 0.0, 
                              num_points: int = 100,
                              max_range = 10,
                              frontier_range: tuple = (-1, 1)):
    
    """
    Compute the efficient frontier and capital market line (CML) for a set of asset returns.
    This function calculates:
        1. The mean returns and standard deviations of each asset.
        2. The tangency portfolio (maximum Sharpe ratio) and its risk/return.
        3. The minimum‐variance portfolio and its risk/return.
        4. A parametrized efficient frontier by convex combinations of the tangency and minimum‐variance portfolios.
        5. The capital market line (CML) extending from the risk‐free rate through the tangency portfolio.
    Args:
            returns (pandas.DataFrame): Historical returns for each asset (rows: observations, columns: assets).
            rf (float, optional): Risk‐free rate (default: 0.0).
            num_points (int, optional): Number of points to compute on the frontier and CML (default: 100).
            max_range (float, optional): Maximum portfolio standard deviation for plotting the CML (default: 10).
            frontier_range (tuple of float, optional): Range of interpolation coefficients φ ∈ [φ_min, φ_max]
                    used to mix tangency and minimum‐variance portfolios (default: (-1, 1)).
    Returns:
            tuple:
                    mu (numpy.ndarray): 1D array of mean returns for each asset.
                    stds (numpy.ndarray): 1D array of standard deviations for each asset.
                    frontier (numpy.ndarray): 2D array of shape (num_points, 2) giving
                            [portfolio_std, portfolio_return] along the efficient frontier.
                    cml (numpy.ndarray): 2D array of shape (num_points, 2) giving
                            [portfolio_std, portfolio_return] along the capital market line.
                    tangency_stats (tuple): (sigma_T, mu_T) risk and return of the tangency portfolio.
                    minimum_variance_stats (tuple): (sigma_M, mu_M) risk and return of the minimum‐variance portfolio.
                    weights (tuple): (w_T, w_M, w_EF) weight vectors for the tangency, minimum‐variance,
                            and last efficient‐frontier portfolio respectively.
    """
    
    mu = returns.mean(axis = 0).values
    sigma = returns.cov().to_numpy()
    stds = numpy.sqrt(numpy.diag(sigma))    
    
    # tangency portfolio
    w_T = tangency_portfolio(returns, rf)
    mu_T = (w_T.T @ mu).item()
    sigma_T = numpy.sqrt(w_T.T @ sigma @ w_T).item()

    # minimum variance portfolio
    w_M = minimum_variance_portfolio(returns)
    mu_M = w_M.T @ mu
    sigma_M = numpy.sqrt(w_M.T @ sigma @ w_M)
    
    # efficient frontier
    phis = numpy.linspace(frontier_range[0], frontier_range[1], num_points)
    frontier = numpy.zeros((len(phis), 2))
    for i, phi in enumerate(phis):
        w_EF = phi * w_T + (1 - phi) * w_M
        mu_EF = w_EF.T @ mu
        sigma_EF = numpy.sqrt(w_EF.T @ sigma @ w_EF)
        frontier[i, 0] = sigma_EF.item()
        frontier[i, 1] = mu_EF.item()

    # cml
    sigmas_CML = numpy.linspace(0, max_range, num_points)
    cml = numpy.zeros((len(sigmas_CML), 2))
    for i, sigma_CML in enumerate(sigmas_CML):
        cml[i, 0] = sigma_CML
        cml[i, 1] = rf + (mu_T - rf) / sigma_T * sigma_CML
    
    return mu, stds, frontier, cml, (sigma_T, mu_T), (sigma_M, mu_M), (w_T, w_M, w_EF)

def FamaMacBeth(test_assets: pandas.Dataframe | numpy.ndarray, 
                factors: pandas.Dataframe | numpy.ndarray,
                riskfree: pandas.Dataframe | numpy.ndarray, 
                estimation: str = '2SLS', 
                disp: int = 0,
                bandwidth: int = 12,
                shanken_correction: bool = True):
    """
    Compute Fama–MacBeth estimates of risk premia using either 2SLS or GMM.
    Parameters
    ----------
    test_assets : numpy.ndarray or pandas.DataFrame
        T×N array of asset returns where T is the number of time periods
        and N is the number of test assets. If a DataFrame is provided,
        column names are preserved for output.
    factors : numpy.ndarray or pandas.DataFrame
        T×K array of risk factor returns where K is the number of factors.
        If a DataFrame is provided, column names are preserved for output.
    riskfree : numpy.ndarray or pandas.DataFrame
        T×1 array of risk‐free returns. Must have the same time dimension
        as `test_assets` and `factors`.
    estimation : str, optional, default='2SLS'
        Estimation method to use:
          - '2SLS': two‐stage least squares
          - 'GMM' : generalized method of moments
    disp : int, optional, default=0
        Display flag for the GMM optimizer. Only used when `estimation='GMM'`.
    bandwidth : int, optional, default=12
        Bandwidth parameter for Newey–West (Bartlett) covariance estimation.
    shanken_correction : bool, optional, default=True
        Whether to apply the Shanken (1992) correction for errors-in-variables
        bias in standard errors. The correction accounts for the fact that betas
        are estimated in the first stage and used in the second stage.
    Returns
    -------
    FamaMacBethResults
        An object encapsulating estimated factor loadings (betas),
        risk premia, standard errors, t‐statistics, and optional
        names of assets and factors.
    Raises
    ------
    ValueError
        If `estimation` is not in {'2SLS', 'GMM'} or if the time dimensions
        of inputs do not match.
    TypeError
        If any of `test_assets`, `factors`, or `riskfree` is not a numpy array
        or pandas DataFrame.
    Notes
    -----
    This function subtracts the risk-free rate from asset returns, fits
    either a 2SLS or GMM linear factor model using linearmodels
    (`LinearFactorModel` or `LinearFactorModelGMM`), and returns a
    consolidated results object for downstream analysis.
    
    The Shanken correction addresses the errors-in-variables problem arising
    from using estimated betas in the second-stage cross-sectional regression.
    The correction factor is: (1 + λ'Σ_f^(-1)λ) where λ is the vector of
    risk premia and Σ_f is the factor covariance matrix.
    """
    
    if estimation not in ['2SLS', 'GMM']:
        raise ValueError("estimation must be either '2SLS' or 'GMM'.")
    
    if not (isinstance(test_assets, numpy.ndarray) or isinstance(test_assets, pandas.DataFrame)):
        raise TypeError("test_assets must be a numpy array or pandas DataFrame.")
    
    if not (isinstance(factors, numpy.ndarray) or isinstance(factors, pandas.DataFrame)):
        raise TypeError("factors must be a numpy array or pandas DataFrame.")
    
    if not (isinstance(riskfree, numpy.ndarray) or isinstance(riskfree, pandas.DataFrame)):
        raise TypeError("riskfree must be a numpy array or pandas DataFrame.")
    
    # Convert to numpy arrays if they are pandas DataFrames
    test_assets_names = None
    if isinstance(test_assets, pandas.DataFrame):
        test_assets_names = test_assets.columns
        test_assets = test_assets.to_numpy()
    
    factors_names = None
    if isinstance(factors, pandas.DataFrame):
        factors_names = factors.columns
        factors = factors.to_numpy()
        
    if isinstance(riskfree, pandas.DataFrame):
        riskfree = riskfree.to_numpy()
    
    T_ta, N = test_assets.shape
    T_f, K = factors.shape
    T_rf = riskfree.shape[0]
    
    # Check dimensions
    T = None
    if T_ta == T_f and T_ta == T_rf:
        T = T_ta
    else:
        raise ValueError("The time dimensions of test_assets, factors, and riskfree do not match.")
    
    excess_returns = test_assets - riskfree
    
    results = None
    if estimation == '2SLS':
        # 2SLS estimation
        results = linearmodels.asset_pricing.LinearFactorModel(
            portfolios = excess_returns, 
            factors = factors, 
            risk_free = False
        ).fit(cov_type = 'kernel', kernel = 'bartlett', bandwidth = bandwidth)
    else:
        # GMM estimation
        results = linearmodels.asset_pricing.LinearFactorModelGMM(
            portfolios = excess_returns, 
            factors = factors, 
            risk_free = False
        ).fit(steps = 10, disp = disp, cov_type = 'kernel', kernel = 'bartlett', bandwidth = bandwidth)
        
    return FamaMacBethResults(results, test_assets_names, factors_names, 
                              factors, shanken_correction)
        
class FamaMacBethResults:
    
    def __init__(self, results, test_assets_names, factors_names, 
                 factors_data=None, shanken_correction=True):
        """
        A container for organizing multi‐factor regression outputs into pandas objects.
        Parameters:
            results (object):
                A regression results object providing attributes:
                - risk_premia
                - risk_premia_se
                - risk_premia_tstats
                - nobs
                - cov
                - alphas
                - betas
                - rsquared
                - params
                - pvalues
                - tstats
                - std_errors
            test_assets_names (list of str or None):
                Names/identifiers of the test assets. If provided, asset‐level metrics
                (alphas, betas, parameters, etc.) will be indexed by these names.
            factors_names (list of str or None):
                Names/identifiers of the factors. If provided, factor‐level metrics
                (risk premia, standard errors, t‐stats, p‐values) will be indexed
                by these names.
            factors_data (numpy.ndarray or None):
                The original factor data used for computing the Shanken correction.
            shanken_correction (bool):
                Whether to apply the Shanken (1992) correction to standard errors.
        Attributes:
            test_assets_names (list of str or None):
                Same as the input parameter.
            factors_names (list of str or None):
                Same as the input parameter.
            linearmodels_results (object):
                Raw regression results object.
            risk_premia (pandas.Series):
                Estimated factor risk premia (λ) indexed by factor names.
            risk_premia_std_errors (pandas.Series):
                Standard errors of the estimated risk premia (Shanken-corrected if enabled).
            risk_premia_tstats (pandas.Series):
                t‐statistics of the estimated risk premia (Shanken-corrected if enabled).
            risk_premia_pvalues (pandas.Series):
                Two‐sided p‐values for the risk premia estimates (Shanken-corrected if enabled).
            covariance (pandas.DataFrame):
                Covariance matrix of alpha and lambda estimates, indexed by
                combinations of assets and factors.
            alphas (pandas.Series):
                Estimated asset alphas indexed by asset names.
            betas (pandas.DataFrame):
                Estimated factor betas for each asset.
            r_squared (array‐like):
                R² values from each asset regression.
            params (pandas.DataFrame):
                Regression parameter estimates (alpha and betas) for each asset.
            pvalues (pandas.DataFrame):
                p‐values associated with each parameter estimate.
            tstats (pandas.DataFrame):
                t‐statistics associated with each parameter estimate.
            std_errors (pandas.DataFrame):
                Standard errors associated with each parameter estimate.
            shanken_correction_applied (bool):
                Whether the Shanken correction was applied to the results.
            shanken_correction_factor (float or None):
                The Shanken correction factor (1 + λ'Σ_f^(-1)λ) if correction was applied.
        """
        self.test_assets_names = test_assets_names
        self.factors_names = factors_names
        self.linearmodels_results = results
        self.risk_premia = results.risk_premia
        self.risk_premia_std_errors = results.risk_premia_se
        self.risk_premia_tstats = results.risk_premia_tstats
        self.shanken_correction_applied = shanken_correction
        self.shanken_correction_factor = None
        
        # Apply Shanken correction if requested and factors data is available
        if shanken_correction and factors_data is not None:
            # Compute factor covariance matrix
            factor_cov = numpy.cov(factors_data, rowvar=False)
            
            # Get risk premia vector
            lambda_vec = self.risk_premia.values.reshape(-1, 1)
            
            # Compute Shanken correction factor: (1 + λ'Σ_f^(-1)λ)
            try:
                factor_cov_inv = numpy.linalg.inv(factor_cov)
                correction_factor = 1 + (lambda_vec.T @ factor_cov_inv @ lambda_vec).item()
                self.shanken_correction_factor = correction_factor
                
                # Apply correction to standard errors
                self.risk_premia_std_errors = self.risk_premia_std_errors * numpy.sqrt(correction_factor)
                
                # Recompute t-statistics with corrected standard errors
                self.risk_premia_tstats = self.risk_premia / self.risk_premia_std_errors
                
            except numpy.linalg.LinAlgError:
                # If factor covariance matrix is singular, don't apply correction
                self.shanken_correction_applied = False
                print("Warning: Factor covariance matrix is singular. Shanken correction not applied.")
        else:
            self.shanken_correction_applied = False
        
        # Compute p-values using the (potentially corrected) t-statistics
        self.risk_premia_pvalues = 2*(1-scipy.stats.t.cdf(abs(self.risk_premia_tstats), results.nobs - 1))
        self.covariance = results.cov
        self.alphas = results.alphas
        self.betas = results.betas
        self.r_squared = results.rsquared
        self.params = results.params
        self.pvalues = results.pvalues
        self.tstats = results.tstats
        self.std_errors = results.std_errors
                
        if self.test_assets_names is not None and self.factors_names is not None:
            
            self.factor_asset_combinations = list(itertools.product(self.test_assets_names, self.factors_names.insert(0, 'Alpha')))
            self.factor_asset_combinations = [f'{asset}:{factor}' for asset, factor in self.factor_asset_combinations]
            self.factor_asset_combinations.extend([f'Lambda:{factor}' for factor in self.factors_names])
            
            self.risk_premia = pandas.Series(self.risk_premia.values, index = self.factors_names)
            self.risk_premia_std_errors = pandas.Series(self.risk_premia_std_errors.values, index = self.factors_names)
            self.risk_premia_tstats = pandas.Series(self.risk_premia_tstats.values, index = self.factors_names)
            self.risk_premia_pvalues = pandas.Series(self.risk_premia_pvalues, index = self.factors_names)
            self.covariance = pandas.DataFrame(self.covariance.values, index = self.factor_asset_combinations, columns = self.factor_asset_combinations)
            self.alphas = pandas.Series(self.alphas.values, index = self.test_assets_names)
            self.betas = pandas.DataFrame(self.betas.values, index = self.test_assets_names, columns = self.factors_names)
            self.params = pandas.DataFrame(self.params.values, index = self.test_assets_names, columns = self.factors_names.insert(0, 'Alpha'))
            self.pvalues = pandas.DataFrame(self.pvalues.values, index = self.test_assets_names, columns = self.factors_names.insert(0, 'Alpha'))
            self.tstats = pandas.DataFrame(self.tstats.values, index = self.test_assets_names, columns = self.factors_names.insert(0, 'Alpha'))
            self.std_errors = pandas.DataFrame(self.std_errors.values, index = self.test_assets_names, columns = self.factors_names.insert(0, 'Alpha'))
            
    def __repr__(self):
        
        width = 80
        
        print_str = "=" * width + "\n"
        if self.shanken_correction_applied:
            print_str += "Fama-MacBeth Results (Shanken-Corrected)\n"
        else:
            print_str += "Fama-MacBeth Results\n"
        print_str += "=" * width + "\n\n"
        print_str += f"Risk Premia:\n{self.riskPremia()}\n\n"
        print_str += f"Factor Loadings:\n{self.factorLoadings()}\n\n"
        print_str += f"=" * width + "\n"
        
        
        
        lines = [
            'Number of Assets: ' + str(self.params.shape[0]),
            'Number of Factors: ' + str(self.params.shape[1] - 1),
            'Number of Observations: ' + str(self.linearmodels_results.nobs),
            'R-Squared: ' + f"{self.r_squared:.3f}",
        ]
        
        if self.shanken_correction_applied and self.shanken_correction_factor is not None:
            lines.append(f'Shanken Correction Factor: {self.shanken_correction_factor:.3f}')
        
        j_lines = str(self.linearmodels_results.j_statistic).splitlines()
        
        # Pad j_lines if we have more lines than j_statistic output
        while len(j_lines) < len(lines):
            j_lines.append('')
        
        for i, line in enumerate(lines):
            offset = 40 - len(line)
            lines[i] = line + ' ' * offset + j_lines[i] if i < len(j_lines) else line
            print_str += lines[i] + '\n'
        
        return print_str
    
    def __str__(self):
        return self.__repr__()
            
    def riskPremia(self, tstat: bool = False) -> pandas.DataFrame:
        """
        Compute and format estimated risk premia for each factor, optionally including
        t-statistics or standard errors.
        Parameters
        ----------
        tstat : bool, default False
            If True, include t-statistics for the estimated risk premia in the output.
            If False, include standard errors instead.
        Returns
        -------
        pandas.DataFrame
            A formatted table of risk premia and their corresponding inference statistics.
            The DataFrame has two rows:
            - 'Risk Premia': point estimates of the risk premia with significance stars,
              formatted to three decimal places (*** p < 0.01, ** p < 0.05, * p < 0.1).
            - 't-stat' or 'Std Error': in square brackets for t-statistics or in
              parentheses for standard errors, formatted to three decimal places.
            Columns correspond to the model's factor names.
        Notes
        -----
        - Significance stars are appended to the risk premia estimates based on
          p-values stored in `self.risk_premia_pvalues`.
        - The underlying numeric values are taken from `self.risk_premia`,
          `self.risk_premia_tstats`, and `self.risk_premia_std_errors`.
        """
        
        rp_res = self.risk_premia.to_frame().transpose()
        
        if tstat:
            rp_res = pandas.concat([rp_res, self.risk_premia_tstats.to_frame().transpose()], axis = 0)
            rp_res.index = ['Risk Premia', 't-stat']
        else:
            rp_res = pandas.concat([rp_res, self.risk_premia_std_errors.to_frame().transpose()], axis = 0)
            rp_res.index = ['Risk Premia', 'Std Error']
        # Format risk premia as strings with significance stars
        def format_with_stars(val, p):
            stars = ''
            if p < 0.01:
                stars = '***'
            elif p < 0.05:
                stars = '**'
            elif p < 0.1:
                stars = '*'
            return f"{val:.3f}{stars}"
        
        rp_res_str = rp_res.copy()
        rp_res_str = rp_res_str.astype(str)
        
        for factor in self.factors_names:
            pval = self.risk_premia_pvalues.loc[factor]
            rp_res_str.loc['Risk Premia', factor] = format_with_stars(rp_res.loc['Risk Premia', factor], pval)
            # format t‐stat (or Std Error) to 4 decimals
            if tstat:
                rp_res_str.loc['t-stat', factor] = f"[{rp_res.loc['t-stat', factor]:.3f}]"
            else:
                rp_res_str.loc['Std Error', factor] = f"({rp_res.loc['Std Error', factor]:.3f})"
        return rp_res_str
    
    def factorLoadings(self, tstat: bool = False) -> pandas.DataFrame:
        """
        Compile and format the factor loadings for a collection of test assets.
        This method constructs a two‐row display for each asset and factor:
          1. The estimated coefficient, formatted to three decimal places with
             significance stars: 
               *** for p < 0.01, ** for p < 0.05, * for p < 0.10.
          2. The accompanying uncertainty measure in brackets—either the t‐statistic
             (if `tstat` is True) or the standard error (if `tstat` is False),
             also formatted to three decimal places.
        Parameters:
            tstat (bool, optional):
                If True, include t‐statistics in the second row for each asset.
                If False (default), include standard errors instead.
        Returns:
            pandas.DataFrame:
                A multi‐indexed table where each asset appears twice in succession:
                  - Row 0, 2, 4, ...: coefficient estimates with significance stars.
                  - Row 1, 3, 5, ...: corresponding t‐statistics (in square brackets)
                                     or standard errors (in parentheses).
                Columns correspond to all factors, with 'Alpha' inserted as the first column.
        """
        
        load_res = self.params
        
        load_res_str = load_res.copy()
        load_res_str = pandas.DataFrame()
        
        for i, asset in enumerate(self.test_assets_names):
            est = load_res.loc[asset, :].to_frame().transpose()
            sig = None
            if tstat:
                sig = self.tstats.loc[asset, :].to_frame().transpose()
                sig.index = ['t-stat']
            else:
                sig = self.std_errors.loc[asset, :].to_frame().transpose()
                sig.index = ['Std Error']
            est = est.astype(str)
            sig = sig.astype(str)
            
            load_res_str = pandas.concat([load_res_str, est, sig], axis = 0)
            

        # Format risk premia as strings with significance stars
        def format_with_stars(val, p):
            stars = ''
            if p < 0.01:
                stars = '***'
            elif p < 0.05:
                stars = '**'
            elif p < 0.1:
                stars = '*'
            return f"{val:.3f}{stars}"
        
        for factor in self.factors_names.insert(0, 'Alpha'):
            for asset in self.test_assets_names:
                pval = self.pvalues.loc[asset, factor]
                load_res_str.loc[asset, factor] = format_with_stars(load_res.loc[asset, factor], pval)
                
                # get numeric positions for a given row‐label and column‐label
                row_idx = load_res_str.index.get_loc(asset)
                col_idx = load_res_str.columns.get_loc(factor)
                
                if tstat:
                    load_res_str.iloc[row_idx + 1, col_idx] = f"[{self.tstats.loc[asset, factor]:.3f}]"
                else:
                    load_res_str.iloc[row_idx + 1, col_idx] = f"({self.std_errors.loc[asset, factor]:.3f})"
        
        return load_res_str
    
    def shanken_info(self) -> dict:
        """
        Return information about the Shanken correction applied to the results.
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'correction_applied': bool indicating if Shanken correction was applied
            - 'correction_factor': float with the correction factor (1 + λ'Σ_f^(-1)λ)
            - 'interpretation': str explaining the correction
        """
        info = {
            'correction_applied': self.shanken_correction_applied,
            'correction_factor': self.shanken_correction_factor,
            'interpretation': None
        }
        
        if self.shanken_correction_applied:
            if self.shanken_correction_factor is not None:
                info['interpretation'] = (
                    f"Shanken (1992) correction applied. Standard errors multiplied by "
                    f"√{self.shanken_correction_factor:.3f} = {numpy.sqrt(self.shanken_correction_factor):.3f} "
                    f"to account for errors-in-variables bias from using estimated betas."
                )
            else:
                info['interpretation'] = "Shanken correction was requested but could not be computed."
        else:
            info['interpretation'] = "No Shanken correction applied. Standard errors may be understated."
            
        return info

        
        
        
        
