import numpy
import scipy.stats


class MertonModel:
    """
    A class to implement the Merton (1974) model for credit risk analysis.
    
    ON THE PRICING OF CORPORATE DEBT: THE RISK STRUCTURE OF INTEREST RATES
    Robert C. Merton (1974, JF)
    https://onlinelibrary.wiley.com/doi/epdf/10.1111/j.1540-6261.1974.tb03058.x

    The Merton model treats a firm's equity as a call option on its assets
    and estimates default probabilities based on asset value dynamics.
    """

    def __init__(self, 
                 V: float, 
                 sigma: float, 
                 r: float, 
                 T: float, 
                 D: float,
                 mu: float = None
                 ) -> None:
        """
        Initialize the Merton Model parameters.

        :param V: Firm's total asset value
        :param sigma: Volatility of the firm's assets (annualized)
        :param r: Risk-free interest rate (annualized)
        :param T: Time to debt maturity (in years)
        :param D: Face value of the firm's debt
        :param mu: The drift of firm's assets (annualized)
        """
        self.V = V
        self.sigma = sigma
        self.r = r
        self.T = T
        self.D = D
        self.mu = mu
        
        # Debt Coverage Ratio
        self.DCR = self.V / self.D
        
    def calculate_d1_d2(self) -> tuple[float, float]:
        """
        Compute d1 and d2 for the Black-Scholes model.

        :return: A tuple containing d1 and d2
        """        
        
        merton_drift = self.r + 0.5 * self.sigma ** 2
        
        d1 = (numpy.log(self.DCR) + merton_drift * self.T) / (self.sigma * numpy.sqrt(self.T))
        d2 = d1 - self.sigma * numpy.sqrt(self.T)
        return d1, d2

    def black_scholes_call(self) -> float:
        """
        Calculate the price of a European call option using the Black-Scholes formula.

        :return: Call option price
        """
        d1, d2 = self.calculate_d1_d2()
        _term1 = self.V * scipy.stats.norm.cdf(d1)
        _term2 = self.D * numpy.exp(-self.r * self.T) * scipy.stats.norm.cdf(d2)
        call_price = _term1 - _term2
        return call_price

    def equity_value(self) -> float:
        """
        Compute the value of equity as a call 
            option on the firm's assets.

        :return: Value of equity
        """
        return self.black_scholes_call()

    def debt_value(self) -> float:
        """
        Compute the value of debt as the firm's 
            total value minus equity.

        :return: Value of debt
        """
        return self.D

    def distance_to_default(self, mu: float = None, method: str = 'merton') -> float:
        """
        Compute the distance to default, measured 
            in standard deviations.

        :return: Distance to default
        """
        dd = None
        if(True):
            _num = numpy.log(self.DCR) + (mu - 0.5 * self.sigma ** 2) * self.T
            _den = self.sigma * numpy.sqrt(self.T)
            dd = _num / _den
        else:
            raise ValueError('')
        return dd
    
    def default_probability(self, mu: float = None, method: str = 'merton') -> float:
        """
        Compute the probability of default using the 
            CDF of the normal distribution.

        :return: Probability of default
        """
        dd = self.distance_to_default(mu = mu, method = method)
        return scipy.stats.norm.cdf(dd)

    def update_parameters(self, 
                          V: float = None, 
                          sigma: float = None, 
                          r: float = None, T: float = None, 
                          D: float = None) -> None:
        """
        Update model parameters dynamically.

        :param V: Firm's total asset value
        :param sigma: Volatility of the firm's assets
        :param r: Risk-free interest rate
        :param T: Time to debt maturity
        :param D: Face value of the firm's debt
        """
        if V is not None:
            self.V = V
        if sigma is not None:
            self.sigma = sigma
        if r is not None:
            self.r = r
        if T is not None:
            self.T = T
        if D is not None:
            self.D = D

    def simulate(self, 
                 mu: float, 
                 n_steps: int = 252, 
                 n_simulations: int = 1000) -> numpy.ndarray:
        """
        Simulate the firm's asset value over time using geometric Brownian motion.

        :param mu: Expected return (drift) of the firm's assets
        :param n_steps: Number of time steps (e.g., 252 for daily steps in one year)
        :param n_simulations: Number of simulation paths
        :return: A 2D NumPy array of simulated paths, where each column is a simulation
        """
        dt = self.T / n_steps
        paths = numpy.zeros((n_steps + 1, n_simulations))
        paths[0] = self.V  # Initial value of the firm's assets

        for t in range(1, n_steps + 1):
            z = numpy.random.standard_normal(n_simulations)  # Random normal shocks
            paths[t] = paths[t - 1] * numpy.exp(
                (mu - 0.5 * self.sigma**2) * dt + self.sigma * numpy.sqrt(dt) * z
            )

        return paths
