"""
TorchBSM: A Differentiable Black-Scholes-Merton Pricing Engine

This module implements the Black-Scholes-Merton model using PyTorch tensors,
enabling automatic differentiation for Greek calculations.

The Black-Scholes formula prices European options under these assumptions:
    - Log-normal stock price distribution
    - Constant volatility and risk-free rate
    - No dividends, no transaction costs
    - Continuous trading

Key insight: Instead of memorizing analytical Greek formulas, we use
torch.autograd.grad() to compute derivatives automatically. This is:
    1. Less error-prone
    2. Extensible to exotic payoffs
    3. Educational (shows how gradients flow)
"""

import torch
from torch.distributions import Normal
from typing import Tuple, Optional


# Standard normal distribution for CDF/PDF calculations
NORMAL = Normal(torch.tensor(0.0), torch.tensor(1.0))


def compute_d1_d2(
    spot: torch.Tensor,
    strike: torch.Tensor,
    time_to_maturity: torch.Tensor,
    rate: torch.Tensor,
    volatility: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute d1 and d2 terms for the Black-Scholes formula.
    
    The d1/d2 terms are the standardized distances to the strike price
    in log-space, adjusted for drift and volatility.
    
    d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
    d2 = d1 - σ√T
    
    Args:
        spot: Current stock price (S)
        strike: Option strike price (K)
        time_to_maturity: Time to expiration in years (T)
        rate: Risk-free interest rate (r)
        volatility: Annualized volatility (σ)
    
    Returns:
        Tuple of (d1, d2) tensors
    """
    sqrt_t = torch.sqrt(time_to_maturity)
    
    d1 = (
        torch.log(spot / strike) 
        + (rate + 0.5 * volatility ** 2) * time_to_maturity
    ) / (volatility * sqrt_t)
    
    d2 = d1 - volatility * sqrt_t
    
    return d1, d2


def price_call(
    spot: torch.Tensor,
    strike: torch.Tensor,
    time_to_maturity: torch.Tensor,
    rate: torch.Tensor,
    volatility: torch.Tensor,
) -> torch.Tensor:
    """
    Price a European call option using Black-Scholes.
    
    Call Price = S·N(d1) - K·e^(-rT)·N(d2)
    
    Where N(x) is the cumulative normal distribution function.
    
    Intuition:
        - S·N(d1): Expected stock value if option is exercised, discounted by probability
        - K·e^(-rT)·N(d2): Present value of strike payment, weighted by exercise probability
    """
    d1, d2 = compute_d1_d2(spot, strike, time_to_maturity, rate, volatility)
    
    call_price = (
        spot * NORMAL.cdf(d1) 
        - strike * torch.exp(-rate * time_to_maturity) * NORMAL.cdf(d2)
    )
    
    return call_price


def price_put(
    spot: torch.Tensor,
    strike: torch.Tensor,
    time_to_maturity: torch.Tensor,
    rate: torch.Tensor,
    volatility: torch.Tensor,
) -> torch.Tensor:
    """
    Price a European put option using Black-Scholes.
    
    Put Price = K·e^(-rT)·N(-d2) - S·N(-d1)
    
    This follows from put-call parity: P = C - S + K·e^(-rT)
    """
    d1, d2 = compute_d1_d2(spot, strike, time_to_maturity, rate, volatility)
    
    put_price = (
        strike * torch.exp(-rate * time_to_maturity) * NORMAL.cdf(-d2)
        - spot * NORMAL.cdf(-d1)
    )
    
    return put_price


def compute_greeks(
    spot: torch.Tensor,
    strike: torch.Tensor,
    time_to_maturity: torch.Tensor,
    rate: torch.Tensor,
    volatility: torch.Tensor,
    option_type: str = "call",
) -> dict:
    """
    Compute option Greeks using automatic differentiation.
    
    This is the key insight of the module: instead of implementing
    analytical formulas for each Greek, we use PyTorch's autograd
    to compute derivatives of the price function.
    
    Greeks computed:
        - Delta (∂Price/∂Spot): Sensitivity to underlying price
        - Gamma (∂²Price/∂Spot²): Rate of change of Delta
        - Vega (∂Price/∂Volatility): Sensitivity to volatility
        - Theta (∂Price/∂Time): Time decay (negative for long options)
        - Rho (∂Price/∂Rate): Sensitivity to interest rates
    
    Args:
        spot, strike, time_to_maturity, rate, volatility: BSM inputs
        option_type: "call" or "put"
    
    Returns:
        Dictionary with Greek values
    """
    # Ensure inputs require gradients for autograd
    spot = spot.clone().requires_grad_(True)
    volatility = volatility.clone().requires_grad_(True)
    time_to_maturity = time_to_maturity.clone().requires_grad_(True)
    rate = rate.clone().requires_grad_(True)
    
    # Price the option
    if option_type == "call":
        price = price_call(spot, strike, time_to_maturity, rate, volatility)
    else:
        price = price_put(spot, strike, time_to_maturity, rate, volatility)
    
    # First-order Greeks via autograd
    delta = torch.autograd.grad(
        price, spot, create_graph=True, retain_graph=True
    )[0]
    
    vega = torch.autograd.grad(
        price, volatility, create_graph=True, retain_graph=True
    )[0]
    
    theta = torch.autograd.grad(
        price, time_to_maturity, create_graph=True, retain_graph=True
    )[0]
    
    rho = torch.autograd.grad(
        price, rate, create_graph=True, retain_graph=True
    )[0]
    
    # Second-order Greek: Gamma (derivative of Delta w.r.t. Spot)
    gamma = torch.autograd.grad(
        delta, spot, retain_graph=True
    )[0]
    
    return {
        "price": price.detach(),
        "delta": delta.detach(),
        "gamma": gamma.detach(),
        "vega": vega.detach(),
        "theta": -theta.detach(),  # Negative because we want decay to be negative
        "rho": rho.detach(),
    }


# =============================================================================
# Implied Volatility Solver (Newton-Raphson)
# =============================================================================

def implied_volatility(
    market_price: torch.Tensor,
    spot: torch.Tensor,
    strike: torch.Tensor,
    time_to_maturity: torch.Tensor,
    rate: torch.Tensor,
    option_type: str = "call",
    initial_guess: float = 0.2,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> Tuple[torch.Tensor, bool]:
    """
    Calculate Implied Volatility using Newton-Raphson iteration.
    
    The Problem:
        We observe a market price (e.g., $5.50 for a call option).
        We want to find the volatility σ that makes BSM(σ) = market_price.
        
    The Solution (Newton-Raphson):
        1. Start with an initial guess for σ (e.g., 20%)
        2. Compute: f(σ) = BSM_price(σ) - market_price
        3. Compute: f'(σ) = Vega (sensitivity of price to volatility)
        4. Update: σ_new = σ_old - f(σ) / f'(σ)
        5. Repeat until |f(σ)| < tolerance
    
    Why Newton-Raphson?
        - Converges quadratically (very fast) when close to solution
        - We already have Vega from autograd, so f'(σ) is free!
        
    Args:
        market_price: Observed option price in the market
        spot: Current stock price
        strike: Option strike price
        time_to_maturity: Time to expiration in years
        rate: Risk-free interest rate
        option_type: "call" or "put"
        initial_guess: Starting volatility estimate (default 20%)
        max_iterations: Maximum Newton-Raphson iterations
        tolerance: Convergence threshold for price difference
    
    Returns:
        Tuple of (implied_volatility, converged_flag)
    """
    # Initialize volatility guess
    sigma = torch.tensor(initial_guess, dtype=torch.float32, requires_grad=True)
    
    # Choose pricing function
    price_fn = price_call if option_type == "call" else price_put
    
    for i in range(max_iterations):
        # Compute model price at current sigma
        model_price = price_fn(spot, strike, time_to_maturity, rate, sigma)
        
        # Price difference (we want this to be zero)
        price_diff = model_price - market_price
        
        # Check convergence
        if abs(price_diff.item()) < tolerance:
            return sigma.detach(), True
        
        # Compute Vega (∂Price/∂σ) via autograd
        # This is f'(σ) in Newton-Raphson: the first derivative of price w.r.t. σ
        # Note: Vega itself is a first derivative, and we use it as f'(σ) here.
        # The second derivative would be Vomma (∂²Price/∂σ² = ∂Vega/∂σ), but
        # Newton-Raphson only needs the first derivative.
        vega = torch.autograd.grad(model_price, sigma, retain_graph=True)[0]
        
        # Avoid division by zero (Vega near zero means we're in trouble)
        if abs(vega.item()) < 1e-10:
            break
        
        # Newton-Raphson update: σ_new = σ_old - f(σ)/f'(σ)
        # Where f(σ) = price_diff and f'(σ) = Vega
        with torch.no_grad():
            sigma -= price_diff / vega
            
            # Keep sigma in reasonable bounds (0.01 to 5.0 = 1% to 500%)
            sigma.clamp_(0.01, 5.0)
        
        # Re-enable gradient tracking
        sigma.requires_grad_(True)
    
    # Did not converge
    return sigma.detach(), False



# =============================================================================
# Convenience class wrapping all functionality
# =============================================================================

class TorchBSM:
    """
    Black-Scholes-Merton pricing engine with automatic Greek calculation.
    
    Example usage:
        >>> bsm = TorchBSM()
        >>> greeks = bsm.price_and_greeks(
        ...     spot=100.0, strike=100.0, time_to_maturity=0.25,
        ...     rate=0.05, volatility=0.2, option_type="call"
        ... )
        >>> print(f"Price: {greeks['price']:.2f}, Delta: {greeks['delta']:.3f}")
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the BSM engine.
        
        Args:
            device: PyTorch device ("cpu" or "cuda"). Auto-detects if None.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
    
    def _to_tensor(self, value: float) -> torch.Tensor:
        """Convert a scalar to a tensor on the correct device."""
        return torch.tensor(value, dtype=torch.float32, device=self.device)
    
    def price_option(
        self,
        spot: float,
        strike: float,
        time_to_maturity: float,
        rate: float,
        volatility: float,
        option_type: str = "call",
    ) -> float:
        """
        Price an option (call or put) using Black-Scholes.
        
        Args:
            spot: Current stock price
            strike: Option strike price
            time_to_maturity: Years until expiration
            rate: Risk-free rate (annualized)
            volatility: Implied volatility (annualized)
            option_type: "call" or "put"
        
        Returns:
            Option price as a float
        """
        # Convert inputs to tensors
        S = self._to_tensor(spot)
        K = self._to_tensor(strike)
        T = self._to_tensor(time_to_maturity)
        r = self._to_tensor(rate)
        sigma = self._to_tensor(volatility)
        
        # Price the option
        if option_type == "call":
            price = price_call(S, K, T, r, sigma)
        else:
            price = price_put(S, K, T, r, sigma)
        
        return price.item()
    
    def price_and_greeks(
        self,
        spot: float,
        strike: float,
        time_to_maturity: float,
        rate: float,
        volatility: float,
        option_type: str = "call",
    ) -> dict:
        """
        Price an option and compute all Greeks.
        
        Args:
            spot: Current stock price
            strike: Option strike price
            time_to_maturity: Years until expiration
            rate: Risk-free rate (annualized, e.g., 0.05 for 5%)
            volatility: Implied volatility (annualized, e.g., 0.2 for 20%)
            option_type: "call" or "put"
        
        Returns:
            Dictionary with 'price', 'delta', 'gamma', 'vega', 'theta', 'rho'
        """
        # Convert inputs to tensors
        S = self._to_tensor(spot)
        K = self._to_tensor(strike)
        T = self._to_tensor(time_to_maturity)
        r = self._to_tensor(rate)
        sigma = self._to_tensor(volatility)
        
        # Compute Greeks
        greeks = compute_greeks(S, K, T, r, sigma, option_type)
        
        # Convert to Python floats for convenience
        return {k: v.item() for k, v in greeks.items()}
    
    def implied_volatility(
        self,
        market_price: float,
        spot: float,
        strike: float,
        time_to_maturity: float,
        rate: float,
        option_type: str = "call",
        initial_guess: float = 0.2,
    ) -> Tuple[float, bool]:
        """
        Calculate Implied Volatility from an observed market price.
        
        This is the "inverse" of pricing: given a price, find the volatility.
        
        Example:
            >>> bsm = TorchBSM()
            >>> iv, converged = bsm.implied_volatility(
            ...     market_price=5.50, spot=100, strike=100,
            ...     time_to_maturity=0.25, rate=0.05
            ... )
            >>> print(f"Implied Vol: {iv:.1%}")  # e.g., "Implied Vol: 22.3%"
        
        Args:
            market_price: Observed option price
            spot: Current stock price
            strike: Option strike price
            time_to_maturity: Years until expiration
            rate: Risk-free rate
            option_type: "call" or "put"
            initial_guess: Starting volatility estimate
        
        Returns:
            Tuple of (implied_volatility, converged_flag)
        """
        # Convert inputs to tensors
        price = self._to_tensor(market_price)
        S = self._to_tensor(spot)
        K = self._to_tensor(strike)
        T = self._to_tensor(time_to_maturity)
        r = self._to_tensor(rate)
        
        # Run Newton-Raphson solver
        iv, converged = implied_volatility(
            price, S, K, T, r, option_type, initial_guess
        )
        
        return iv.item(), converged


# =============================================================================
# Quick test / demonstration
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TorchBSM: Black-Scholes with Automatic Greeks")
    print("=" * 60)
    
    # Create the pricing engine
    bsm = TorchBSM()
    
    # Example: ATM call option
    # Stock at $100, Strike at $100, 3 months to expiry, 5% rate, 20% vol
    params = {
        "spot": 100.0,
        "strike": 100.0,
        "time_to_maturity": 0.25,  # 3 months
        "rate": 0.05,              # 5% risk-free rate
        "volatility": 0.20,        # 20% annualized volatility
    }
    
    print("\nOption Parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    
    # Price call
    call_greeks = bsm.price_and_greeks(**params, option_type="call")
    print("\nCall Option Greeks:")
    for greek, value in call_greeks.items():
        print(f"  {greek.capitalize():6s}: {value:8.4f}")
    
    # Price put
    put_greeks = bsm.price_and_greeks(**params, option_type="put")
    print("\nPut Option Greeks:")
    for greek, value in put_greeks.items():
        print(f"  {greek.capitalize():6s}: {value:8.4f}")
    
    # Verify put-call parity: C - P = S - K*e^(-rT)
    parity_lhs = call_greeks["price"] - put_greeks["price"]
    parity_rhs = params["spot"] - params["strike"] * torch.exp(
        torch.tensor(-params["rate"] * params["time_to_maturity"])
    ).item()
    
    print(f"\nPut-Call Parity Check:")
    print(f"  C - P = {parity_lhs:.4f}")
    print(f"  S - K·e^(-rT) = {parity_rhs:.4f}")
    print(f"  Difference: {abs(parity_lhs - parity_rhs):.6f} (should be ~0)")
    
    # =========================================================================
    # Implied Volatility Demo
    # =========================================================================
    print("\n" + "=" * 60)
    print("Implied Volatility Solver (Newton-Raphson)")
    print("=" * 60)
    
    # Scenario: You see a call trading at $5.50. What's the implied vol?
    observed_price = 5.50
    print(f"\nScenario: Call option trading at ${observed_price:.2f}")
    print(f"  (Same params: S=100, K=100, T=0.25, r=5%)")
    
    iv, converged = bsm.implied_volatility(
        market_price=observed_price,
        spot=100.0,
        strike=100.0,
        time_to_maturity=0.25,
        rate=0.05,
        option_type="call",
    )
    
    print(f"\nResult:")
    print(f"  Implied Volatility: {iv:.2%}")
    print(f"  Converged: {converged}")
    
    # Verify: price the option with the recovered IV
    verification = bsm.price_and_greeks(
        spot=100.0, strike=100.0, time_to_maturity=0.25,
        rate=0.05, volatility=iv, option_type="call"
    )
    print(f"\nVerification (pricing with recovered IV):")
    print(f"  BSM Price: ${verification['price']:.4f}")
    print(f"  Market Price: ${observed_price:.2f}")
    print(f"  Difference: ${abs(verification['price'] - observed_price):.6f}")
    
    # Round-trip test: start with known vol, price it, recover vol
    print("\n" + "-" * 40)
    print("Round-Trip Test:")
    print("-" * 40)
    true_vol = 0.35  # 35%
    print(f"  True volatility: {true_vol:.0%}")
    
    # Price with known vol
    price_at_35 = bsm.price_and_greeks(
        spot=100.0, strike=100.0, time_to_maturity=0.25,
        rate=0.05, volatility=true_vol, option_type="call"
    )["price"]
    print(f"  BSM price at 35% vol: ${price_at_35:.4f}")
    
    # Recover vol from price
    recovered_vol, _ = bsm.implied_volatility(
        market_price=price_at_35,
        spot=100.0, strike=100.0, time_to_maturity=0.25,
        rate=0.05, option_type="call"
    )
    print(f"  Recovered volatility: {recovered_vol:.2%}")
    print(f"  Error: {abs(recovered_vol - true_vol):.6f} (should be ~0)")

