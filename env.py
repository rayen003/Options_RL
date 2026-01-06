"""
MacroOptionsEnv: A Gymnasium environment for Options Trading with RL.

This environment simulates trading ATM call and put options. The agent
observes market state (prices, Greeks) and portfolio state (positions, PnL),
then decides to Buy/Sell calls or puts, or Hold.

State Space:
    The observation is a vector of continuous features that capture
    everything the agent needs to make trading decisions.

MDP Structure:
    - State: [spot, tte, call_greeks, put_greeks, call_pos, put_pos, pnls, regime, iv]
    - Action: {0: BUY_CALL, 1: BUY_PUT, 2: SELL_CALL, 3: SELL_PUT, 4: HOLD}
    - Reward: Change in portfolio value - transaction costs
    - Episode: 60 trading days (default)
    
Educational Note:
    With both calls and puts, agent can learn:
    - Hedging (protective puts, covered calls)
    - Spreads (bull call, bear put, iron condor)
    - Synthetic positions (synthetic long = long call + short put)
    - Put-call parity
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any

# Import our BSM pricing engine
from bsm import TorchBSM


class OptionsEnv(gym.Env):
    """
    Options trading environment for Reinforcement Learning.
    
    The agent trades both ATM call and put options over a 60-day episode.
    Stock price evolves via Geometric Brownian Motion (GBM).
    
    NEW: With both calls AND puts, agent can learn sophisticated strategies:
        - Hedging: Protective puts, covered calls
        - Spreads: Bull call, bear put, iron condor, straddles
        - Synthetics: Synthetic long (call + short put), conversion/reversal
        - Put-Call Parity: C - P = S - K*e^(-rT)
    
    Observation Space (~18 features, dynamically sized):
    ┌────────────────────────────────────────────────────────────────┐
    │ Index │ Feature              │ Description                    │
    ├───────┼──────────────────────┼────────────────────────────────┤
    │   0   │ spot_normalized      │ Stock price / initial price    │
    │   1   │ time_to_expiry       │ Years remaining                │
    │   2   │ call_delta           │ Call option delta (0 to 1)     │
    │   3   │ call_gamma           │ Call option gamma (≥ 0)        │
    │   4   │ call_vega            │ Call option vega (≥ 0)         │
    │   5   │ call_theta           │ Call option theta (≤ 0)        │
    │   6   │ put_delta            │ Put option delta (-1 to 0)     │
    │   7   │ put_gamma            │ Put option gamma (≥ 0)         │
    │   8   │ put_vega             │ Put option vega (≥ 0)          │
    │   9   │ put_theta            │ Put option theta (≤ 0)         │
    │  10   │ call_position        │ Call position (-1, 0, +1)      │
    │  11   │ put_position         │ Put position (-1, 0, +1)       │
    │  12   │ call_pnl_normalized  │ Call PnL / initial cash        │
    │  13   │ put_pnl_normalized   │ Put PnL / initial cash         │
    │  14   │ regime               │ Market regime (-1 bear, +1 bull)│
    │  15   │ implied_volatility   │ Extracted IV (Newton-Raphson)  │
    └────────────────────────────────────────────────────────────────┘
    
    Action Space (5 discrete actions):
        0: BUY_CALL  - Buy 1 call contract (or close short call)
        1: BUY_PUT   - Buy 1 put contract (or close short put)
        2: SELL_CALL - Sell 1 call contract (or close long call)
        3: SELL_PUT  - Sell 1 put contract (or close long put)
        4: HOLD      - Do nothing
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        initial_cash: float = 10_000.0,
        initial_spot: float = 100.0,
        strike: float = 100.0,
        risk_free_rate: float = 0.05,
        volatility: float = 0.20,
        time_to_expiry: float = 0.33,       # ~4 months in years (80 trading days)
        episode_length: int = 60,            # Trading days (default: 60 = ~3 months)
        transaction_cost: float = 0.02,      # 2% round-trip cost
        seed: Optional[int] = None,
        # ==================== Market Regime Parameters ====================
        use_regime: bool = True,             # Enable market regimes
        bull_drift: float = 0.30,            # Annual drift in bull market (30%) - STRONGER
        bear_drift: float = -0.30,           # Annual drift in bear market (-30%) - STRONGER
        regime_switch_prob: float = 0.03,    # Daily probability of regime switch (3%) - less switching
        # ==================== IV Dynamics Parameters ====================
        use_stochastic_vol: bool = True,     # Enable stochastic volatility
        vol_of_vol: float = 0.30,            # Volatility of volatility (30% annual)
        vol_mean_reversion: float = 2.0,     # Mean reversion speed (higher = faster reversion)
        vol_long_term_mean: float = 0.20,    # Long-term mean volatility (20%)
        # ==================== Reward Shaping Parameters ====================
        use_reward_shaping: bool = True,     # Enable advanced reward shaping
        delta_penalty_weight: float = 0.001, # Penalty for non-neutral delta
        iv_threshold: float = 0.30,          # IV threshold for expensive options (30%)
        iv_penalty_weight: float = 0.002,    # Penalty for buying expensive options
        iv_bonus_weight: float = 0.002,      # Bonus for selling expensive options
    ):
        """
        Initialize the options trading environment.
        
        Args:
            initial_cash: Starting cash balance
            initial_spot: Initial stock price
            strike: Option strike price (we use ATM, so strike = spot)
            risk_free_rate: Annual risk-free rate
            volatility: Annual volatility for GBM simulation
            time_to_expiry: Initial time to expiry in years (e.g., 0.25 = 3 months)
            episode_length: Number of trading days per episode
                           (default: 60 days = ~3 months, matches option expiry)
                           Longer episodes allow agent to see full option lifecycle
            transaction_cost: Proportional transaction cost (bid-ask spread)
            seed: Random seed for reproducibility
            
            NEW - Market Regime Parameters:
            use_regime: If True, market switches between bull/bear phases
            bull_drift: Annual drift rate during bull markets
            bear_drift: Annual drift rate during bear markets
            regime_switch_prob: Daily probability of switching regimes
        """
        super().__init__()
        
        # Store environment parameters
        self.initial_cash = initial_cash
        self.initial_spot = initial_spot
        self.strike = strike
        self.rate = risk_free_rate
        self.volatility = volatility
        self.episode_length = episode_length
        self.transaction_cost = transaction_cost
        
        # Ensure option expiry is at least as long as episode (with some buffer)
        # Convert episode_length (trading days) to years: days / 252 trading days per year
        min_tte = (episode_length + 10) / 252  # Add 10 day buffer
        self.initial_tte = max(time_to_expiry, min_tte)
        
        # Market regime parameters
        self.use_regime = use_regime
        self.bull_drift = bull_drift
        self.bear_drift = bear_drift
        self.regime_switch_prob = regime_switch_prob
        
        # IV dynamics parameters (stochastic volatility)
        self.use_stochastic_vol = use_stochastic_vol
        self.vol_of_vol = vol_of_vol
        self.vol_mean_reversion = vol_mean_reversion
        self.vol_long_term_mean = vol_long_term_mean
        self.initial_volatility = volatility  # Store initial vol for reset
        
        # Reward shaping parameters
        self.use_reward_shaping = use_reward_shaping
        self.delta_penalty_weight = delta_penalty_weight
        self.iv_threshold = iv_threshold
        self.iv_penalty_weight = iv_penalty_weight
        self.iv_bonus_weight = iv_bonus_weight
        
        # REAL-LIFE SETUP:
        # - true_volatility: Hidden vol that market maker uses (agent doesn't see)
        # - market_price: Price computed from true_volatility
        # - implied_volatility: Agent extracts this using Newton-Raphson
        self.true_volatility = volatility    # Hidden from agent
        self.implied_volatility = volatility  # What agent extracts from prices
        
        # Initialize BSM pricing engine
        self.bsm = TorchBSM()
        
        # =====================================================================
        # Define State Space (Observation Space)
        # =====================================================================
        # All features are continuous. We use Box space with reasonable bounds.
        # The bounds are "soft" - values can exceed them slightly, but this
        # gives the RL algorithm a sense of scale.
        
        # Build observation space dynamically based on features enabled
        # Base: 14 features (spot, tte, call_greeks, put_greeks, positions, pnls)
        # + optional regime signal + optional IV
        obs_low = [
            0.5,    # 0: spot_normalized: Stock won't drop below 50%
            0.0,    # 1: time_to_expiry: Can reach 0
            # Call Greeks
            0.0,    # 2: call_delta: Call delta ranges 0 to 1
            0.0,    # 3: call_gamma: Always positive
            0.0,    # 4: call_vega: Always positive
            -50.0,  # 5: call_theta: Can be very negative near expiry
            # Put Greeks
            -1.0,   # 6: put_delta: Put delta ranges -1 to 0
            0.0,    # 7: put_gamma: Always positive
            0.0,    # 8: put_vega: Always positive
            -50.0,  # 9: put_theta: Can be very negative near expiry
            # Positions
            -1.0,   # 10: call_position: -1 (short), 0 (flat), +1 (long)
            -1.0,   # 11: put_position: -1 (short), 0 (flat), +1 (long)
            # PnL
            -1.0,   # 12: call_pnl_normalized: Can lose up to 100%
            -1.0,   # 13: put_pnl_normalized: Can lose up to 100%
        ]
        obs_high = [
            2.0,    # 0: spot_normalized: Stock won't double
            0.5,    # 1: time_to_expiry: Can be up to 6 months
            # Call Greeks
            1.0,    # 2: call_delta: Call delta maxes at 1
            1.0,    # 3: call_gamma: Bounded in practice
            50.0,   # 4: call_vega: High but bounded
            0.0,    # 5: call_theta: Always negative for long options
            # Put Greeks
            0.0,    # 6: put_delta: Put delta maxes at 0
            1.0,    # 7: put_gamma: Bounded in practice
            50.0,   # 8: put_vega: High but bounded
            0.0,    # 9: put_theta: Always negative for long options
            # Positions
            1.0,    # 10: call_position: Long 1 contract max
            1.0,    # 11: put_position: Long 1 contract max
            # PnL
            2.0,    # 12: call_pnl_normalized: Can make up to 200%
            2.0,    # 13: put_pnl_normalized: Can make up to 200%
        ]
        
        # Add regime signal if using regimes
        if self.use_regime:
            obs_low.append(-1.0)   # regime_signal: -1 = bear
            obs_high.append(1.0)   # regime_signal: +1 = bull
        
        # Add IV if using stochastic volatility
        if self.use_stochastic_vol:
            obs_low.append(0.05)   # iv_normalized: Min ~5% vol
            obs_high.append(1.0)   # iv_normalized: Max ~100% vol
        
        self.observation_space = spaces.Box(
            low=np.array(obs_low, dtype=np.float32),
            high=np.array(obs_high, dtype=np.float32),
            dtype=np.float32
        )
        
        # =====================================================================
        # Define Action Space
        # =====================================================================
        # Discrete: 0 = BUY_CALL, 1 = BUY_PUT, 2 = SELL_CALL, 3 = SELL_PUT, 4 = HOLD
        self.action_space = spaces.Discrete(5)
        self.ACTION_BUY_CALL = 0
        self.ACTION_BUY_PUT = 1
        self.ACTION_SELL_CALL = 2
        self.ACTION_SELL_PUT = 3
        self.ACTION_HOLD = 4
        
        # =====================================================================
        # Initialize State Variables (will be set in reset())
        # =====================================================================
        self.spot = initial_spot
        self.tte = time_to_expiry
        self.cash = initial_cash
        
        # NEW: Separate positions for calls and puts
        self.call_position = 0      # -1, 0, or +1 contracts
        self.put_position = 0       # -1, 0, or +1 contracts
        self.call_entry_price = 0.0 # Price we bought/sold call at
        self.put_entry_price = 0.0  # Price we bought/sold put at
        
        # NEW: Bond investment (uninvested cash earns risk-free rate)
        # This represents cash sitting in a money market account
        self.total_interest_earned = 0.0  # Cumulative interest from bonds
        
        self.step_count = 0
        self.call_price = 0.0
        self.put_price = 0.0
        self.call_greeks = {}
        self.put_greeks = {}
        
        # NEW: Market regime (1 = bull, -1 = bear)
        self.regime = 1  # Start in bull market
        
        # Random number generator
        self.np_random = np.random.default_rng(seed)
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct the observation vector from current state.
        
        REAL-LIFE WORKFLOW:
        ===================
        1. Market maker prices BOTH call and put using true_volatility (hidden)
        2. Agent sees market_prices on screen
        3. Agent uses Newton-Raphson to extract implied_volatility (from call price)
        4. Agent computes Greeks for BOTH call and put using extracted IV
        5. Agent makes decisions based on extracted IV, NOT true vol
        
        This is realistic - in real life you never know the "true" volatility!
        """
        # =====================================================================
        # STEP 1: Market maker prices BOTH options using TRUE volatility
        # =====================================================================
        if self.use_stochastic_vol:
            # Market prices are set by the "market maker" using true vol
            self.call_price = self.bsm.price_option(
                spot=self.spot,
                strike=self.strike,
                time_to_maturity=max(self.tte, 0.001),
                rate=self.rate,
                volatility=self.true_volatility,  # TRUE vol - agent doesn't see this
                option_type="call"
            )
            
            self.put_price = self.bsm.price_option(
                spot=self.spot,
                strike=self.strike,
                time_to_maturity=max(self.tte, 0.001),
                rate=self.rate,
                volatility=self.true_volatility,  # TRUE vol - agent doesn't see this
                option_type="put"
            )
            
            # =====================================================================
            # STEP 2: Agent extracts IV using Newton-Raphson (REAL-LIFE PROCESS)
            # =====================================================================
            # Extract IV from call price (could also use put price)
            self._extract_implied_volatility()
            
            # =====================================================================
            # STEP 3: Agent computes Greeks for BOTH using EXTRACTED IV
            # =====================================================================
            # Agent uses their extracted IV estimate to compute Greeks
            self.call_greeks = self.bsm.price_and_greeks(
                spot=self.spot,
                strike=self.strike,
                time_to_maturity=max(self.tte, 0.001),
                rate=self.rate,
                volatility=self.implied_volatility,  # EXTRACTED IV
                option_type="call"
            )
            
            self.put_greeks = self.bsm.price_and_greeks(
                spot=self.spot,
                strike=self.strike,
                time_to_maturity=max(self.tte, 0.001),
                rate=self.rate,
                volatility=self.implied_volatility,  # EXTRACTED IV
                option_type="put"
            )
        else:
            # No stochastic vol: use fixed volatility
            self.call_greeks = self.bsm.price_and_greeks(
                spot=self.spot,
                strike=self.strike,
                time_to_maturity=max(self.tte, 0.001),
                rate=self.rate,
                volatility=self.initial_volatility,
                option_type="call"
            )
            self.call_price = self.call_greeks["price"]
            
            self.put_greeks = self.bsm.price_and_greeks(
                spot=self.spot,
                strike=self.strike,
                time_to_maturity=max(self.tte, 0.001),
                rate=self.rate,
                volatility=self.initial_volatility,
                option_type="put"
            )
            self.put_price = self.put_greeks["price"]
        
        # Calculate unrealized PnL for both call and put positions
        if self.call_position != 0:
            # Mark-to-market: what's our call position worth now?
            call_unrealized_pnl = self.call_position * (self.call_price - self.call_entry_price) * 100
        else:
            call_unrealized_pnl = 0.0
            
        if self.put_position != 0:
            # Mark-to-market: what's our put position worth now?
            put_unrealized_pnl = self.put_position * (self.put_price - self.put_entry_price) * 100
        else:
            put_unrealized_pnl = 0.0
        
        # Construct observation vector
        base_obs = [
            self.spot / self.initial_spot,              # 0: Normalized spot
            self.tte,                                    # 1: Time to expiry
            # Call Greeks
            self.call_greeks["delta"],                   # 2: Call delta
            self.call_greeks["gamma"],                   # 3: Call gamma
            self.call_greeks["vega"] / 100.0,            # 4: Call vega (scaled)
            self.call_greeks["theta"] / 100.0,           # 5: Call theta (scaled)
            # Put Greeks
            self.put_greeks["delta"],                    # 6: Put delta
            self.put_greeks["gamma"],                    # 7: Put gamma
            self.put_greeks["vega"] / 100.0,             # 8: Put vega (scaled)
            self.put_greeks["theta"] / 100.0,            # 9: Put theta (scaled)
            # Positions
            float(self.call_position),                   # 10: Call position
            float(self.put_position),                    # 11: Put position
            # PnL
            call_unrealized_pnl / self.initial_cash,     # 12: Call PnL normalized
            put_unrealized_pnl / self.initial_cash,      # 13: Put PnL normalized
        ]
        
        # Add regime signal if using regimes
        if self.use_regime:
            base_obs.append(float(self.regime))          # 14: -1 = bear, +1 = bull
        
        # Add EXTRACTED IV if using stochastic volatility
        # Agent sees their Newton-Raphson extracted IV, NOT the true vol!
        if self.use_stochastic_vol:
            base_obs.append(self.implied_volatility)     # 15: EXTRACTED IV
        
        obs = np.array(base_obs, dtype=np.float32)
        
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Return auxiliary information (not used by agent, but useful for debugging).
        
        NEW: With both calls and puts, we track separate prices, positions, and Greeks.
        
        REAL-LIFE SETUP:
        - true_vol: Hidden volatility (market maker's secret)
        - implied_vol: What agent extracted using Newton-Raphson
        - iv_error: Difference between true and extracted (numerical error)
        """
        iv_error = abs(self.true_volatility - self.implied_volatility) if self.use_stochastic_vol else 0.0
        
        # Calculate portfolio delta for debugging
        call_delta = self.call_greeks.get("delta", 0.5)
        put_delta = self.put_greeks.get("delta", -0.5)
        portfolio_delta = self.call_position * call_delta + self.put_position * put_delta
        
        return {
            "spot": self.spot,
            # Call info
            "call_price": self.call_price,
            "call_position": self.call_position,
            "call_greeks": self.call_greeks,
            # Put info
            "put_price": self.put_price,
            "put_position": self.put_position,
            "put_greeks": self.put_greeks,
            # Portfolio info
            "portfolio_delta": portfolio_delta,
            "cash": self.cash,
            "total_interest_earned": self.total_interest_earned,  # NEW: Bond interest
            "step": self.step_count,
            "tte": self.tte,
            "regime": self.regime,
            "regime_name": "BULL" if self.regime == 1 else "BEAR",
            # REAL-LIFE IV SETUP
            "true_vol": self.true_volatility,         # Hidden from agent (market maker's)
            "implied_vol": self.implied_volatility,   # Agent's extracted IV
            "iv_error": iv_error,                     # How accurate was Newton-Raphson?
            "true_vol_pct": f"{self.true_volatility:.1%}",
            "implied_vol_pct": f"{self.implied_volatility:.1%}",
        }
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Called at the start of each episode.
        
        Returns:
            observation: Initial state vector
            info: Auxiliary information
        """
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        
        # Reset state variables
        self.spot = self.initial_spot
        self.tte = self.initial_tte
        self.cash = self.initial_cash
        
        # NEW: Reset separate call and put positions
        self.call_position = 0
        self.put_position = 0
        self.call_entry_price = 0.0
        self.put_entry_price = 0.0
        
        # NEW: Reset bond interest tracker
        self.total_interest_earned = 0.0
        
        self.step_count = 0
        
        # Reset market regime (random start: 50% bull, 50% bear)
        if self.use_regime:
            self.regime = 1 if self.np_random.random() < 0.5 else -1
        else:
            self.regime = 1  # Default to bull if regimes disabled
        
        # Reset volatility (REAL-LIFE SETUP)
        if self.use_stochastic_vol:
            # TRUE volatility: Market maker's hidden vol (with random start)
            self.true_volatility = self.initial_volatility * (1 + 0.1 * self.np_random.standard_normal())
            self.true_volatility = np.clip(self.true_volatility, 0.05, 1.0)
            
            # IMPLIED volatility: What agent extracts (starts at initial guess)
            # Agent will update this using Newton-Raphson on first observation
            self.implied_volatility = self.initial_volatility
        else:
            self.true_volatility = self.initial_volatility
            self.implied_volatility = self.initial_volatility
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    # =========================================================================
    # STEP FUNCTION - The Core of the RL Environment
    # =========================================================================
    
    def _simulate_true_volatility_change(self) -> None:
        """
        Simulate the TRUE volatility using Ornstein-Uhlenbeck process.
        
        REAL-LIFE SETUP:
        ================
        The "market maker" has access to this true volatility.
        The AGENT does NOT see this - they only see market prices.
        
        The Ornstein-Uhlenbeck (OU) process is mean-reverting:
            dσ = κ(θ - σ)dt + ξ·dW
        
        Where:
            - σ = current true volatility
            - κ = mean reversion speed (vol_mean_reversion)
            - θ = long-term mean (vol_long_term_mean)
            - ξ = volatility of volatility (vol_of_vol)
            - dW = Wiener process increment
        """
        if not self.use_stochastic_vol:
            return
        
        dt = 1 / 252  # One trading day
        
        # Mean-reverting drift (pulls IV toward long-term mean)
        mean_reversion = self.vol_mean_reversion * (self.vol_long_term_mean - self.true_volatility) * dt
        
        # Random shock to volatility
        Z = self.np_random.standard_normal()
        vol_shock = self.vol_of_vol * np.sqrt(dt) * Z
        
        # Update TRUE volatility (hidden from agent)
        self.true_volatility = self.true_volatility + mean_reversion + vol_shock
        
        # Keep volatility in reasonable bounds (5% to 100%)
        self.true_volatility = np.clip(self.true_volatility, 0.05, 1.0)
    
    def _extract_implied_volatility(self) -> float:
        """
        NEWTON-RAPHSON IV EXTRACTION (Real-Life Workflow)
        
        This is what traders actually do:
        1. See a market price on their screen (call price)
        2. Use numerical methods to find the IV that matches the price
        3. Make trading decisions based on extracted IV
        
        The agent sees ONLY:
        - Market price (computed from hidden true_volatility)
        - Extracted IV (via Newton-Raphson, has numerical error)
        
        The agent NEVER sees:
        - The true_volatility (that's the market maker's secret)
        
        Note: We extract IV from the call price. By put-call parity, the put
        price should imply the same IV (up to numerical error).
        """
        if not self.use_stochastic_vol:
            return self.initial_volatility
        
        # Use the call market price (already computed in _get_observation)
        # Agent uses Newton-Raphson to EXTRACT IV from market price
        # This is the real-life process!
        extracted_iv, converged = self.bsm.implied_volatility(
            market_price=self.call_price,  # Market maker's call price (using true_volatility)
            spot=self.spot,
            strike=self.strike,
            time_to_maturity=max(self.tte, 0.001),
            rate=self.rate,
            option_type="call",
            initial_guess=self.implied_volatility,  # Use previous IV as guess
        )
        
        if converged:
            self.implied_volatility = extracted_iv
        # If didn't converge, keep previous IV estimate
        
        return self.implied_volatility
    
    def _simulate_price_movement(self) -> None:
        """
        Simulate one day of stock price movement using Geometric Brownian Motion.
        
        GBM Formula:
            S(t+dt) = S(t) * exp((μ - σ²/2)*dt + σ*√dt*Z)
        
        Where:
            - μ = drift (depends on market regime: bull or bear)
            - σ = volatility (annualized, now stochastic!)
            - dt = time step (1/252 years = 1 trading day)
            - Z = standard normal random variable
        
        Market Regimes:
            - Bull market: drift = bull_drift (e.g., +30% annual)
            - Bear market: drift = bear_drift (e.g., -30% annual)
            - Regimes switch randomly with probability regime_switch_prob
        
        Stochastic Volatility:
            - IV changes each step via Ornstein-Uhlenbeck process
            - Mean-reverts to long-term average
        """
        dt = 1 / 252  # One trading day in years
        
        # =====================================================================
        # Regime switching
        # =====================================================================
        if self.use_regime:
            # Check if regime should switch
            if self.np_random.random() < self.regime_switch_prob:
                self.regime *= -1  # Flip: bull -> bear or bear -> bull
            
            # Use regime-dependent drift
            if self.regime == 1:  # Bull market
                mu = self.bull_drift
            else:  # Bear market
                mu = self.bear_drift
        else:
            # No regimes: use risk-free rate as drift
            mu = self.rate
        
        # =====================================================================
        # Stochastic volatility (TRUE volatility dynamics - hidden from agent)
        # =====================================================================
        self._simulate_true_volatility_change()
        
        # Random shock (standard normal)
        Z = self.np_random.standard_normal()
        
        # GBM update with regime-dependent drift and TRUE volatility
        # Stock moves based on true realized volatility (hidden from agent)
        drift = (mu - 0.5 * self.true_volatility ** 2) * dt
        diffusion = self.true_volatility * np.sqrt(dt) * Z
        
        self.spot = self.spot * np.exp(drift + diffusion)
        
        # Reduce time to expiry
        self.tte = max(0.0, self.tte - dt)
    
    def _execute_action(self, action: int) -> float:
        """
        Execute a trading action and return the transaction cost.
        
        NEW: With both calls and puts, we have 5 actions:
        
        Actions:
            0 (BUY_CALL):  If flat → go long 1 call
                           If short call → close short call
                           If long call → do nothing
            
            1 (BUY_PUT):   If flat → go long 1 put
                           If short put → close short put
                           If long put → do nothing
            
            2 (SELL_CALL): If flat → go short 1 call
                           If long call → close long call
                           If short call → do nothing
            
            3 (SELL_PUT):  If flat → go short 1 put
                           If long put → close long put
                           If short put → do nothing
            
            4 (HOLD):      Do nothing
        
        Returns:
            Transaction cost incurred (0 if no trade)
            
        Educational Note:
            - BUY_CALL + BUY_PUT = Long Straddle (profit from big moves either way)
            - SELL_CALL + BUY_PUT = Protective Put (downside protection)
            - BUY_CALL + SELL_PUT = Synthetic Long (replicates long stock)
        """
        transaction_cost = 0.0
        
        # ====================== CALL ACTIONS ======================
        if action == self.ACTION_BUY_CALL:
            if self.call_position <= 0:  # Flat or short → buy
                # Close any short call position first
                if self.call_position == -1:
                    # Buying back short call: pay the ask price
                    buy_price = self.call_price * (1 + self.transaction_cost / 2)
                    pnl = (self.call_entry_price - buy_price) * 100  # Short profit/loss
                    self.cash += pnl
                    transaction_cost = self.call_price * self.transaction_cost / 2 * 100
                
                # Go long call
                self.call_position = 1
                self.call_entry_price = self.call_price * (1 + self.transaction_cost / 2)
                transaction_cost += self.call_price * self.transaction_cost / 2 * 100
                
        elif action == self.ACTION_SELL_CALL:
            if self.call_position >= 0:  # Flat or long → sell
                # Close any long call position first
                if self.call_position == 1:
                    # Selling long call: receive the bid price
                    sell_price = self.call_price * (1 - self.transaction_cost / 2)
                    pnl = (sell_price - self.call_entry_price) * 100  # Long profit/loss
                    self.cash += pnl
                    transaction_cost = self.call_price * self.transaction_cost / 2 * 100
                
                # Go short call
                self.call_position = -1
                self.call_entry_price = self.call_price * (1 - self.transaction_cost / 2)
                transaction_cost += self.call_price * self.transaction_cost / 2 * 100
        
        # ====================== PUT ACTIONS ======================
        elif action == self.ACTION_BUY_PUT:
            if self.put_position <= 0:  # Flat or short → buy
                # Close any short put position first
                if self.put_position == -1:
                    # Buying back short put: pay the ask price
                    buy_price = self.put_price * (1 + self.transaction_cost / 2)
                    pnl = (self.put_entry_price - buy_price) * 100  # Short profit/loss
                    self.cash += pnl
                    transaction_cost = self.put_price * self.transaction_cost / 2 * 100
                
                # Go long put
                self.put_position = 1
                self.put_entry_price = self.put_price * (1 + self.transaction_cost / 2)
                transaction_cost += self.put_price * self.transaction_cost / 2 * 100
                
        elif action == self.ACTION_SELL_PUT:
            if self.put_position >= 0:  # Flat or long → sell
                # Close any long put position first
                if self.put_position == 1:
                    # Selling long put: receive the bid price
                    sell_price = self.put_price * (1 - self.transaction_cost / 2)
                    pnl = (sell_price - self.put_entry_price) * 100  # Long profit/loss
                    self.cash += pnl
                    transaction_cost = self.put_price * self.transaction_cost / 2 * 100
                
                # Go short put
                self.put_position = -1
                self.put_entry_price = self.put_price * (1 - self.transaction_cost / 2)
                transaction_cost += self.put_price * self.transaction_cost / 2 * 100
        
        # ACTION_HOLD: do nothing, transaction_cost stays 0
        
        return transaction_cost
    
    def _calculate_portfolio_value(self) -> float:
        """
        Calculate current portfolio value (cash + mark-to-market positions).
        
        Portfolio Value = Cash + Call Position Value + Put Position Value
        
        Position Value = Position * Current Option Price * 100
        The *100 is because one option contract = 100 shares.
        
        NEW: With both calls and puts, we need to sum both position values.
        """
        call_position_value = self.call_position * self.call_price * 100
        put_position_value = self.put_position * self.put_price * 100
        return self.cash + call_position_value + put_position_value
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step of the environment.
        
        This is the main function called by the RL agent:
            obs, reward, terminated, truncated, info = env.step(action)
        
        The step sequence:
            1. Record portfolio value BEFORE action
            2. Execute the action (buy/hold/sell)
            3. Simulate stock price movement (GBM)
            4. Record portfolio value AFTER
            5. Calculate reward = change in value - transaction costs
            6. Check if episode is done
            7. Return new observation
        
        Args:
            action: 0 (BUY), 1 (HOLD), or 2 (SELL)
        
        Returns:
            observation: New state vector (8 features)
            reward: Profit/loss this step minus transaction costs
            terminated: True if episode ended naturally (e.g., expiration)
            truncated: True if episode ended early (e.g., time limit)
            info: Debug information
        """
        # 1. Portfolio value before action
        value_before = self._calculate_portfolio_value()
        
        # 2. Execute the action
        transaction_cost = self._execute_action(action)
        
        # 3. Simulate stock price movement
        self._simulate_price_movement()
        self.step_count += 1
        
        # 3.5 NEW: Accrue interest on uninvested cash (money market account)
        # Cash earns risk-free rate per day: r_daily = r_annual / 252
        # This is realistic - brokers automatically sweep idle cash into money market funds
        daily_interest = self.cash * (self.rate / 252)
        self.cash += daily_interest
        self.total_interest_earned += daily_interest
        
        # 4. Get new observation (also updates self.option_price)
        obs = self._get_observation()
        
        # 5. Portfolio value after
        value_after = self._calculate_portfolio_value()
        
        # 6. Calculate reward
        # Reward = Change in portfolio value - Transaction costs
        # We normalize by initial cash to keep rewards in a reasonable range
        raw_pnl = value_after - value_before
        reward = (raw_pnl - transaction_cost) / self.initial_cash
        
         # Track reward components for debugging
        reward_components = {
            "base_pnl": reward,
            "regime_penalty": 0.0,
            "delta_penalty": 0.0,
            "iv_penalty": 0.0,
        }
        
        # Regime-conditional penalty
        # Penalize positions that go against the market regime
        # NEW: Check both call and put positions
        if self.use_regime:
            # Long calls in bear market = bad (losing money fighting the trend)
            if self.call_position == 1 and self.regime == -1:
                reward -= 0.005
                reward_components["regime_penalty"] = -0.005
            # Short calls in bull market = bad (missing the upside)
            elif self.call_position == -1 and self.regime == 1:
                reward -= 0.005
                reward_components["regime_penalty"] = -0.005
            # Long puts in bull market = bad (puts lose value in bull markets)
            if self.put_position == 1 and self.regime == 1:
                reward -= 0.005
                reward_components["regime_penalty"] -= 0.005
            # Short puts in bear market = bad (puts gain value in bear markets)
            elif self.put_position == -1 and self.regime == -1:
                reward -= 0.005
                reward_components["regime_penalty"] -= 0.005
        
        # =====================================================================
        # REWARD SHAPING: Encourage sophisticated trading behavior
        # =====================================================================
        if self.use_reward_shaping:
            # 1. DELTA PENALTY: Encourage delta-neutral positions (hedging)
            # Portfolio delta = call_position * call_delta + put_position * put_delta
            # Example: Long call (delta=0.5) + Long put (delta=-0.5) = Delta-neutral straddle
            # Example: Long call (delta=0.5) + Short put (delta=-0.5, position=-1) = Synthetic long (delta~1)
            call_delta = self.call_greeks.get("delta", 0.5)
            put_delta = self.put_greeks.get("delta", -0.5)
            portfolio_delta = self.call_position * call_delta + self.put_position * put_delta
            delta_penalty = -self.delta_penalty_weight * abs(portfolio_delta)
            reward += delta_penalty
            reward_components["delta_penalty"] = delta_penalty
            
            # 2. IV PENALTY/BONUS: Buy cheap options, sell expensive ones
            # This encourages volatility arbitrage behavior
            current_iv = self.implied_volatility
            
            # Penalize buying when IV is high (options are expensive)
            if action in [self.ACTION_BUY_CALL, self.ACTION_BUY_PUT] and current_iv > self.iv_threshold:
                iv_penalty = -self.iv_penalty_weight
                reward += iv_penalty
                reward_components["iv_penalty"] = iv_penalty
            
            # Reward selling when IV is high (collect premium)
            elif action in [self.ACTION_SELL_CALL, self.ACTION_SELL_PUT] and current_iv > self.iv_threshold:
                iv_bonus = self.iv_bonus_weight
                reward += iv_bonus
                reward_components["iv_penalty"] = iv_bonus  # Stored as positive
        
        # 7. Check termination conditions
        terminated = False
        truncated = False
        
        # Episode ends if:
        # a) We've reached the episode length
        if self.step_count >= self.episode_length:
            truncated = True
            # Force close any open positions at episode end
            # NEW: Close both call and put positions
            if self.call_position != 0:
                close_cost = self.call_price * self.transaction_cost / 2 * 100
                if self.call_position == 1:
                    self.cash += (self.call_price - self.call_entry_price) * 100 - close_cost
                else:
                    self.cash += (self.call_entry_price - self.call_price) * 100 - close_cost
                self.call_position = 0
            
            if self.put_position != 0:
                close_cost = self.put_price * self.transaction_cost / 2 * 100
                if self.put_position == 1:
                    self.cash += (self.put_price - self.put_entry_price) * 100 - close_cost
                else:
                    self.cash += (self.put_entry_price - self.put_price) * 100 - close_cost
                self.put_position = 0
        
        # b) Options expired (time to expiry reached 0)
        if self.tte <= 0.001:
            terminated = True
            # At expiration: 
            # Call is worth max(S - K, 0)
            # Put is worth max(K - S, 0)
            call_intrinsic = max(0, self.spot - self.strike)
            put_intrinsic = max(0, self.strike - self.spot)
            
            # Settle call position
            if self.call_position == 1:
                self.cash += (call_intrinsic - self.call_entry_price) * 100
            elif self.call_position == -1:
                self.cash += (self.call_entry_price - call_intrinsic) * 100
            self.call_position = 0
            
            # Settle put position
            if self.put_position == 1:
                self.cash += (put_intrinsic - self.put_entry_price) * 100
            elif self.put_position == -1:
                self.cash += (self.put_entry_price - put_intrinsic) * 100
            self.put_position = 0
        
        # 8. Get info
        info = self._get_info()
        info["transaction_cost"] = transaction_cost
        info["reward"] = reward
        info["portfolio_value"] = value_after
        info["reward_components"] = reward_components  # For debugging reward shaping
        
        return obs, reward, terminated, truncated, info


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("OptionsEnv: REAL-LIFE WORKFLOW (Newton-Raphson IV Extraction)")
    print("=" * 80)
    
    # Create environment with regimes and stochastic vol enabled
    env = OptionsEnv(use_regime=True, use_stochastic_vol=True)
    
    # Reset and get initial state
    obs, info = env.reset(seed=42)
    
    print("\n" + "-" * 80)
    print("REAL-LIFE WORKFLOW")
    print("-" * 80)
    print("""
    1. Market Maker has TRUE volatility (hidden from agent)
       - Evolves via Ornstein-Uhlenbeck process
    
    2. Market Maker prices option using TRUE volatility
       - Agent sees this price on their screen
    
    3. Agent uses NEWTON-RAPHSON to extract IMPLIED volatility
       - Finds σ such that: BSM(spot, strike, T, r, σ) = Market_Price
    
    4. Agent makes decisions using EXTRACTED IV (not true vol!)
       - Greeks computed from extracted IV
    """)
    
    print("-" * 80)
    print("CONFIGURATION")
    print("-" * 80)
    print(f"Observation Space: Box{env.observation_space.shape}")
    print(f"Action Space: Discrete({env.action_space.n})")
    print(f"\nMarket Regimes: ENABLED")
    print(f"  Bull Drift: {env.bull_drift:.0%}")
    print(f"  Bear Drift: {env.bear_drift:.0%}")
    print(f"  Switch Probability: {env.regime_switch_prob:.0%} per day")
    print(f"\nStochastic Volatility: ENABLED (Real-Life Mode)")
    print(f"  Initial IV: {env.initial_volatility:.0%}")
    print(f"  Vol of Vol: {env.vol_of_vol:.0%}")
    print(f"  Mean Reversion Speed: {env.vol_mean_reversion}")
    print(f"  Long-term Mean: {env.vol_long_term_mean:.0%}")
    
    # Feature names for display
    feature_names = [
        "spot_norm", "tte", "delta", "gamma",
        "vega", "theta", "position", "pnl", "regime", "implied_iv"
    ]
    
    print("\n" + "-" * 80)
    print("EPISODE SIMULATION (Showing True Vol vs Extracted IV)")
    print("-" * 80)
    
    total_reward = 0
    action_names = {0: "BUY ", 1: "HOLD", 2: "SELL"}
    
    print(f"\n{'Step':>4} | {'Regime':>5} | {'TrueVol':>7} | {'ExtractIV':>9} | {'Error':>6} | {'Action':>4} | {'Spot':>7} | {'Reward':>8}")
    print("-" * 90)
    
    # Run a short episode
    for step in range(10):
        # Random action
        action = env.action_space.sample()
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        regime_str = info['regime_name']
        true_vol = info['true_vol_pct']
        impl_vol = info['implied_vol_pct']
        iv_err = f"{info['iv_error']*100:.2f}%"
        print(f"{step:>4} | {regime_str:>5} | {true_vol:>7} | {impl_vol:>9} | {iv_err:>6} | {action_names[action]:>4} | ${info['spot']:>6.2f} | {reward:>+8.4f}")
        
        if terminated or truncated:
            print("\n[Episode ended]")
            break
    
    print("\n" + "-" * 80)
    print("FINAL STATE")
    print("-" * 80)
    print(f"  Cash:            ${info['cash']:.2f}")
    print(f"  Position:        {info['position']}")
    print(f"  Portfolio Value: ${info['portfolio_value']:.2f}")
    print(f"  Total Reward:    {total_reward:.4f}")
    print(f"  Final Regime:    {info['regime_name']}")
    print(f"\n  TRUE Volatility:     {info['true_vol_pct']}  (Hidden from agent)")
    print(f"  EXTRACTED IV:        {info['implied_vol_pct']}  (Agent's estimate via Newton-Raphson)")
    print(f"  IV Extraction Error: {info['iv_error']*100:.4f}%")
    
    print("\n" + "-" * 60)
    print("OBSERVATION BREAKDOWN (Final)")
    print("-" * 60)
    for name, value in zip(feature_names, obs):
        print(f"  {name:12s}: {value:>8.4f}")
    
    # =========================================================================
    # Test a specific strategy: Buy and Hold
    # =========================================================================
    print("\n" + "=" * 60)
    print("STRATEGY TEST: Buy and Hold")
    print("=" * 60)
    
    obs, info = env.reset(seed=123)
    initial_value = info['cash']
    
    # Buy on day 0
    obs, reward, _, _, info = env.step(0)  # BUY
    print(f"Day 0: BUY  @ ${info['option_price']:.2f}")
    
    # Hold for 29 days
    for day in range(1, 30):
        obs, reward, terminated, truncated, info = env.step(1)  # HOLD
    
    final_value = info['portfolio_value']
    print(f"Day 30: Spot=${info['spot']:.2f}, Option=${info['option_price']:.2f}")
    print(f"\nReturn: ${final_value - initial_value:.2f} ({(final_value/initial_value - 1)*100:.2f}%)")
    
    print("\n✓ Environment working correctly!")
