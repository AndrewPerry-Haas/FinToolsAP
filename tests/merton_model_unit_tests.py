import sys
import unittest
import numpy as np

# add source directory to path
sys.path.insert(0, '../src/FinToolsAP/')

from MertonModel import MertonModel


class TestMertonModel(unittest.TestCase):
    """
    Unit tests for the MertonModel class.
    """

    def setUp(self) -> None:
        """
        Set up test cases with sample data.
        """
        self.model = MertonModel(
            V=100.0,
            sigma=0.2,
            r=0.05,
            T=1.0,
            D=80.0
        )

    def test_calculate_d1_d2(self):
        """
        Test the calculation of d1 and d2.
        """
        d1, d2 = self.model.calculate_d1_d2()
        self.assertAlmostEqual(d1, 0.636, places=3)
        self.assertAlmostEqual(d2, 0.436, places=3)

    def test_black_scholes_call(self):
        """
        Test the Black-Scholes call option price calculation.
        """
        call_price = self.model.black_scholes_call()
        self.assertAlmostEqual(call_price, 25.198, places=3)

    def test_equity_value(self):
        """
        Test the equity value calculation.
        """
        equity = self.model.equity_value()
        self.assertAlmostEqual(equity, 25.198, places=3)

    def test_debt_value(self):
        """
        Test the debt value calculation.
        """
        debt = self.model.debt_value()
        self.assertAlmostEqual(debt, 80, places=3)

    def test_distance_to_default(self):
        """
        Test the distance to default calculation.
        """
        dd = self.model.distance_to_default(mu=0.08)
        self.assertAlmostEqual(dd, 0.927, places=3)

    def test_default_probability(self):
        """
        Test the default probability calculation.
        """
        default_prob = self.model.default_probability(mu=0.08)
        self.assertAlmostEqual(default_prob, 0.176, places=3)

    def test_update_parameters(self):
        """
        Test dynamic parameter updates.
        """
        self.model.update_parameters(V=120.0, sigma=0.25)
        self.assertEqual(self.model.V, 120.0)
        self.assertEqual(self.model.sigma, 0.25)

    def test_simulation_shape(self):
        """
        Test the shape of the simulated paths.
        """
        paths = self.model.simulate(mu=0.08, n_steps=252, n_simulations=100)
        self.assertEqual(paths.shape, (253, 100))

    def test_simulation_values(self):
        """
        Test the validity of simulated path values.
        """
        paths = self.model.simulate(mu=0.08, n_steps=10, n_simulations=1)
        self.assertTrue(np.all(paths > 0))


if __name__ == "__main__":
    unittest.main()
