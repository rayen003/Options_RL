# Controlled bull and bear trajectory study

This is a diagnostic extension to the [paired mixed-regime policy study](EVALUATION_PROTOCOL.md). It answers a different question: how do the existing policies' marked portfolio P&L paths behave when the simulator remains in a bull regime or remains in a bear regime for a full episode? It is not a real-market backtest.

## Scenarios and policies

- `bull`: regime fixed at +1 for the full episode; simulator uses its configured bull drift.
- `bear`: regime fixed at -1 for the full episode; simulator uses its configured bear drift.
- Regime switching probability set to zero in both scenarios. All other environment parameters remain at defaults, including stochastic volatility and option transaction costs.
- Policies: hold cash, uniformly random actions, the observed-regime call rule, and all three PPO models trained with seeds 11, 23, and 37. No policy is retrained or retuned for this study.

## Pairing and measurements

- Evaluation seeds: **20000–20019**, separate from the original study's 10000–10019.
- Episode length: 60 steps. For each scenario and seed, every policy sees the same generated underlying and volatility path. Random actions use the existing independent episode-seeded generator.
- At each step, record marked portfolio value and cumulative P&L dollars relative to initial cash. Final point includes executable liquidation, consistent with the terminal metric in the original study. Shaped reward is not P&L.
- Plot one panel per fixed regime. Each policy line is the mean cumulative P&L across 20 paths; shaded band shows 10th–90th percentile across those same paths. Policies remain grouped by PPO training seed rather than selecting a best seed.

## Interpretation limits

These controlled paths test behavior under two simplified simulator conditions; they do not establish a strategy's performance in actual bull or bear markets. Regime drift is known directly to the policies through the observation. Options are priced by the simulator's BSM model, and the model omits realistic margin, collateral, liquidity, and market impact. Bear-scenario regime rule sells naked calls. The plot must be presented with these limitations.
