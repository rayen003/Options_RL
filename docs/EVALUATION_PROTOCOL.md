# Paired simulated-policy evaluation protocol

Status: specified **before** running performance comparisons. This study measures behavior inside the current simulator, not returns in real options markets.

## Question

Does a PPO policy trained in this simulator produce better **terminal portfolio value** than simple policies on unseen simulator random seeds? Shaped RL reward is reported separately and is never called financial return.

## Policies

All policies use the same five actions and receive the same observation. No policy reads simulator-only `info` fields such as `true_vol`.

| Name | Rule |
| --- | --- |
| `cash` | Always HOLD. Cash accrues simulator's risk-free financing rate. |
| `random` | Uniformly sample one of five actions from a separately seeded random generator. |
| `regime_call` | If observed regime is bull (+1), BUY_CALL; if bear (−1), SELL_CALL. This can create a naked short call. |
| `ppo` | `model.predict(observation, deterministic=True)` from a newly trained model. |

`regime_call` is an intentionally simple diagnostic rule, not a recommended trading strategy. Current environment exposes regime directly and has no margin or collateral model. Comparing PPO against this rule tests whether PPO adds anything beyond a visible regime signal; it does not test alpha against real markets.

## Fairness and seeds

- Default environment parameters and 60-step episodes for every policy. No environment redesign during this comparison.
- Training seeds: **11, 23, 37**. Fresh PPO model and fresh environment for each run; default 20,000 training timesteps and existing PPO hyperparameters. No model trained before the ledger correction is valid.
- Held-out evaluation seeds: **10000–10019**. Every policy runs on each seed. Simulator random numbers are independent of actions, so same seed should produce same spot and regime path; tests must verify this.
- Random-action generator seeded independently for each episode. Repeating a run with same seed should yield same actions and metrics.
- No hyperparameter selection on held-out paths. If a pilot changes training budget or settings, record change before interpreting final results.

## Per-episode measurements

- `terminal_return_pct = 100 × (final_portfolio_value / initial_cash − 1)`; final value includes liquidation and modeled financing.
- `max_drawdown_pct = 100 × max_t(1 − value_t / max_{u≤t} value_u)`; positive loss magnitude.
- `spread_paid_dollars = sum(info["transaction_cost"])`, including forced final liquidation.
- `total_shaped_reward = sum(step_rewards)`; diagnostic only, never interchangeable with return.

Report policy distribution across the 20 paired paths and per-path difference from `random` and `cash`. PPO results reported by **training seed**, not only best run. Show counts, mean, median, minimum/maximum, and variability; charts should show distributions and paired differences, not only one trajectory. Small sample and simulator assumptions must appear next to figures.

## Execution gates

1. Tests prove each baseline action rule, matched market paths, deterministic random actions, terminal-value accounting, and metric calculations.
2. Run short baseline/policy pilot and measure seconds per environment step. Estimate full study time before multi-seed PPO training; discuss if excessive.
3. Train three PPO seeds only after timing gate. Save code version, seeds, training settings, and raw episode-level CSV alongside any summary.
4. Review results and limitations before changing README or portfolio claims. Weak/negative PPO result is valid finding, not reason to hide a policy or tune against held-out paths.

## Scope limits

No historical quotes, realistic margin, order-book fills, or independent volatility-pricing signal. PPO can learn simulator incentives rather than economic edge. Do not present simulated gains as investment performance. Pricing/Greeks/BSM study is separate Part C, after this evaluation.
