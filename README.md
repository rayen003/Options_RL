# Options RL: Reinforcement Learning for Options Trading

A modular reinforcement learning system for options trading that integrates a differentiable Black-Scholes pricing engine with market regime detection.

## Project Structure

```
options_rl/
├── bsm.py          # Black-Scholes-Merton pricing engine with autograd Greeks
├── env.py          # Gymnasium environment for options trading
├── train.py        # PPO training script with experiment tracking
├── visualize.py    # Visualization and analysis tools
├── requirements.txt
└── README.md
```

## Features

### 1. Differentiable BSM Engine (`bsm.py`)
- Black-Scholes option pricing for calls/puts
- **Automatic Greek calculation** via PyTorch autograd (Delta, Gamma, Vega, Theta, Rho)
- **Newton-Raphson IV solver** for implied volatility calculation
- No hardcoded Greek formulas — all derivatives computed automatically

### 2. Trading Environment (`env.py`)
- Custom Gymnasium environment for options trading
- **Market regimes**: Bull/Bear phases with configurable drift
- GBM stock price simulation
- Configurable episode length (30-120+ trading days)
- Transaction costs and position management

### 3. Training (`train.py`)
- PPO algorithm via Stable-Baselines3
- Experiment tracking with datetime+UUID directories
- Configurable hyperparameters
- Automatic model saving and metadata logging

### 4. Visualization (`visualize.py`)
- Episode trajectory plots
- Action distribution analysis
- Reward comparison (trained vs random)
- Greeks vs actions scatter plots

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train an agent
```bash
# Default training (20k steps, 60-day episodes)
python train.py

# Custom training
python train.py --timesteps 50000 --episode-length 90
```

### Visualize results
```bash
python visualize.py --episodes 20
```

### Test the environment
```bash
python env.py
```

### Test BSM pricing
```bash
python bsm.py
```

## Key Concepts

### State Space (9 features)
1. `spot_normalized`: Stock price / initial price
2. `time_to_expiry`: Years remaining
3. `delta`: Option delta
4. `gamma`: Option gamma
5. `vega`: Option vega (scaled)
6. `theta`: Option theta (scaled)
7. `position`: Current position (-1, 0, +1)
8. `pnl_normalized`: Unrealized PnL / initial cash
9. `regime`: Market regime (-1 = bear, +1 = bull)

### Action Space
- `0`: BUY (go long)
- `1`: HOLD (do nothing)
- `2`: SELL (go short)

### Market Regimes
- **Bull market**: +30% annual drift
- **Bear market**: -30% annual drift
- 3% daily probability of regime switch

## Example Results

The trained agent learns regime-conditional behavior:
- **Bull market**: 100% BUY actions
- **Bear market**: 100% SELL actions

Typical improvement over random agent: +10-25% in average reward.

## Tech Stack

- Python 3.10+
- PyTorch (differentiable pricing)
- Gymnasium (RL environment)
- Stable-Baselines3 (PPO algorithm)
- NumPy, Matplotlib

## Future Enhancements

- [ ] Stochastic volatility (IV dynamics)
- [ ] Multi-leg strategies (spreads, straddles)
- [ ] Delta hedging with underlying
- [ ] Multi-asset portfolios
- [ ] Real market data integration

## License

MIT License

