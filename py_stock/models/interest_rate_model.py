"""
Interest Rate Stochastic Models - CIR and Vasicek models for rate forecasting
"""
import numpy as np
from typing import Tuple, List
from dataclasses import dataclass
from .stock_params import InterestRateParams


@dataclass
class InterestRateProjection:
    """Interest rate path projections"""
    years: List[float]
    mean_rates: List[float]
    std_rates: List[float]
    paths: np.ndarray  # Shape: (num_paths, num_steps)
    expected_final_rate: float
    confidence_interval: Tuple[float, float]  # 95% CI


class CIRModel:
    """Cox-Ingersoll-Ross (CIR) interest rate model"""
    
    def __init__(self, params: InterestRateParams):
        """
        Initialize CIR model
        
        Args:
            params: InterestRateParams object with CIR parameters
        """
        self.r0 = params.r0
        self.kappa = params.kappa  # Mean reversion speed
        self.theta = params.theta  # Long-term mean
        self.sigma = params.sigma  # Volatility
        
    def simulate_paths(self, 
                      T: float = 5,
                      num_steps: int = 252,
                      num_paths: int = 1000) -> InterestRateProjection:
        """
        Simulate interest rate paths using CIR model
        
        Args:
            T: Time horizon (years)
            num_steps: Number of time steps
            num_paths: Number of simulation paths
            
        Returns:
            InterestRateProjection with simulated paths
        """
        dt = T / num_steps
        time_grid = np.linspace(0, T, num_steps + 1)
        
        # Initialize paths
        paths = np.zeros((num_paths, num_steps + 1))
        paths[:, 0] = self.r0
        
        # CIR dynamics: dr = kappa(theta - r)dt + sigma*sqrt(r)dW
        for step in range(num_steps):
            dW = np.random.normal(0, np.sqrt(dt), num_paths)
            r = paths[:, step]
            
            # Euler scheme with reflection to ensure positivity
            dr = self.kappa * (self.theta - r) * dt + self.sigma * np.sqrt(np.maximum(r, 0)) * dW
            r_new = r + dr
            
            # Reflect if becomes negative (Feller condition)
            r_new = np.maximum(r_new, 0)
            paths[:, step + 1] = r_new
        
        # Calculate statistics
        mean_rates = np.mean(paths, axis=0)
        std_rates = np.std(paths, axis=0)
        
        # Confidence interval at final time
        final_rates = paths[:, -1]
        ci_lower = np.percentile(final_rates, 2.5)
        ci_upper = np.percentile(final_rates, 97.5)
        
        return InterestRateProjection(
            years=list(time_grid),
            mean_rates=list(mean_rates),
            std_rates=list(std_rates),
            paths=paths,
            expected_final_rate=float(np.mean(final_rates)),
            confidence_interval=(ci_lower, ci_upper)
        )
    
    def closed_form_expected_rate(self, t: float) -> float:
        """
        Closed-form solution for expected rate at time t
        E[r(t)] = theta + (r0 - theta)*exp(-kappa*t)
        """
        return self.theta + (self.r0 - self.theta) * np.exp(-self.kappa * t)
    
    def closed_form_variance(self, t: float) -> float:
        """
        Closed-form solution for variance at time t
        """
        numerator = self.sigma**2 * (1 - np.exp(-2*self.kappa*t))
        denominator = 2 * self.kappa
        return numerator / denominator * self.theta


class VasicekModel:
    """Vasicek interest rate model"""
    
    def __init__(self, params: InterestRateParams):
        """
        Initialize Vasicek model
        
        Args:
            params: InterestRateParams object with Vasicek parameters
        """
        self.r0 = params.r0
        self.kappa = params.kappa  # Mean reversion speed
        self.theta = params.theta  # Long-term mean
        self.sigma = params.sigma  # Volatility
        
    def simulate_paths(self,
                      T: float = 5,
                      num_steps: int = 252,
                      num_paths: int = 1000) -> InterestRateProjection:
        """
        Simulate interest rate paths using Vasicek model
        
        Args:
            T: Time horizon (years)
            num_steps: Number of time steps
            num_paths: Number of simulation paths
            
        Returns:
            InterestRateProjection with simulated paths
        """
        dt = T / num_steps
        time_grid = np.linspace(0, T, num_steps + 1)
        
        # Initialize paths
        paths = np.zeros((num_paths, num_steps + 1))
        paths[:, 0] = self.r0
        
        # Vasicek dynamics: dr = kappa(theta - r)dt + sigma*dW
        for step in range(num_steps):
            dW = np.random.normal(0, np.sqrt(dt), num_paths)
            r = paths[:, step]
            
            # Exact solution for Vasicek
            dr = self.kappa * (self.theta - r) * dt + self.sigma * dW
            r_new = r + dr
            paths[:, step + 1] = r_new
        
        # Calculate statistics
        mean_rates = np.mean(paths, axis=0)
        std_rates = np.std(paths, axis=0)
        
        # Confidence interval at final time
        final_rates = paths[:, -1]
        ci_lower = np.percentile(final_rates, 2.5)
        ci_upper = np.percentile(final_rates, 97.5)
        
        return InterestRateProjection(
            years=list(time_grid),
            mean_rates=list(mean_rates),
            std_rates=list(std_rates),
            paths=paths,
            expected_final_rate=float(np.mean(final_rates)),
            confidence_interval=(ci_lower, ci_upper)
        )
    
    def closed_form_expected_rate(self, t: float) -> float:
        """
        Closed-form solution for expected rate at time t
        E[r(t)] = theta + (r0 - theta)*exp(-kappa*t)
        """
        return self.theta + (self.r0 - self.theta) * np.exp(-self.kappa * t)
    
    def closed_form_variance(self, t: float) -> float:
        """
        Closed-form solution for variance at time t
        Var[r(t)] = sigma^2 / (2*kappa) * (1 - exp(-2*kappa*t))
        """
        numerator = self.sigma**2 * (1 - np.exp(-2*self.kappa*t))
        denominator = 2 * self.kappa
        return numerator / denominator


class InterestRateAnalyzer:
    """Analyze interest rate forecasts and their impact"""
    
    def __init__(self, params: InterestRateParams):
        self.params = params
        
        if params.model_type.upper() == "CIR":
            self.model = CIRModel(params)
        else:
            self.model = VasicekModel(params)
    
    def forecast(self, T: float = 5) -> InterestRateProjection:
        """Generate interest rate forecast"""
        return self.model.simulate_paths(T=T, num_steps=252, num_paths=1000)
    
    def rate_scenario_analysis(self) -> dict:
        """
        Analyze different interest rate scenarios
        Returns scenarios: Bull (lower rates), Base, Bear (higher rates)
        """
        T = 5
        projection = self.forecast(T)
        
        base_final = projection.expected_final_rate
        
        # Simple scenarios based on mean reversion
        bull_scenario = base_final * 0.8  # 20% lower
        bear_scenario = base_final * 1.2  # 20% higher
        
        return {
            'base_case': base_final,
            'bull_case': bull_scenario,  # Lower rates
            'bear_case': bear_scenario,  # Higher rates
            'current_rate': self.params.r0,
            'long_term_mean': self.params.theta,
            'mean_reversion_speed': self.params.kappa
        }
    
    def impact_on_dividend_yield(self, dividend_yield: float) -> dict:
        """
        Analyze impact of rate changes on dividend yield attractiveness
        """
        projection = self.forecast()
        base_rate = self.params.r0
        expected_rate = projection.expected_final_rate
        
        rate_change = expected_rate - base_rate
        
        # Higher rates reduce attractiveness of dividend stocks
        # (Opportunity cost increases)
        
        return {
            'current_risk_free_rate': base_rate,
            'expected_rate_5y': expected_rate,
            'rate_change': rate_change,
            'dividend_yield': dividend_yield,
            'excess_yield': dividend_yield - expected_rate,
            'rate_impact': 'Negative' if rate_change > 0 else 'Positive'
        }
