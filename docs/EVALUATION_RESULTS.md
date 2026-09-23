# Paired policy study: results

These are results inside a **synthetic options simulator**, not historical backtests, evidence of market alpha, or a deployable trading strategy. The [protocol](EVALUATION_PROTOCOL.md) was committed before performance comparisons. This page records all three PPO training seeds, including weak results.

## What was run

- Three fresh PPO runs: training seeds 11, 23, and 37; 20,000 requested timesteps each. Stable-Baselines3 completed 20,224 timesteps per run because rollout batches are indivisible. Default environment and PPO settings were unchanged.
- Four policy types: hold cash, seeded uniform random actions, a direct observed-regime call rule, and each PPO model. The simple rule buys a call in bull regime and sells a call in bear regime; the latter can be a naked short.
- Twenty held-out simulator seeds, 10000–10019, with 60 steps per episode. Each policy faced the same seeded market path. Neither models nor rule were chosen or tuned against these paths.
- Terminal portfolio value includes final liquidation and modeled financing. Return is `(final value / initial cash - 1) × 100`; positive maximum drawdown measures peak-to-trough loss. Shaped training reward is kept separate from financial return.

## Main results

Each row summarizes 20 paired simulated episodes. Return and drawdown columns are percentages; spread is dollars paid per episode. Lower drawdown and spread are preferable. Full distributions, including shaped reward, are in `experiments/study_20260923/evaluation/summary.csv` after running the commands below.

| Policy | Mean return | Median return | Worst return | Return SD | Mean max drawdown | Mean spread paid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hold cash | +1.20 | +1.20 | +1.20 | 0.00 | 0.00 | 0.00 |
| Uniform random | -0.65 | -0.93 | -36.77 | 12.38 | 10.11 | 276.16 |
| Observed-regime call rule | +6.60 | +6.61 | -10.57 | 8.28 | 5.34 | 34.47 |
| PPO, training seed 11 | +6.60 | +6.61 | -10.57 | 8.28 | 5.34 | 34.47 |
| PPO, training seed 23 | +7.25 | +10.33 | -19.06 | 11.52 | 7.57 | 51.13 |
| PPO, training seed 37 | +1.92 | +5.70 | -18.32 | 10.50 | 8.18 | 28.64 |

Because paths are paired, compare each PPO model with the rule on each same seed, not only their aggregate medians:

| PPO training seed | Mean return difference vs rule | Median difference vs rule | Paths better / tied / worse |
| --- | ---: | ---: | ---: |
| 11 | 0.00 percentage points | 0.00 | 0 / 20 / 0 |
| 23 | +0.65 percentage points | +3.90 | 12 / 0 / 8 |
| 37 | -4.68 percentage points | -0.77 | 9 / 1 / 10 |

Seed 11 produced exactly the same *reported episode metrics* as the rule on all 20 paths; this does not prove identical actions for every possible observation. Seed 23 had a higher median but a worse worst-case return and greater average drawdown than the rule. Seed 37 was weaker on mean return. **These runs do not establish a consistent PPO advantage over the simple rule.** Beating uniform random is a low bar because random trading pays much more spread.

## Limits and interpretation

The environment gives the bull/bear regime directly to every policy. Option quotes come from its BSM model, and extracted IV recovers the volatility used to generate those quotes; this is not an independent pricing edge. No historical quotes, liquidity model, margin/collateral requirement, stock hedge, or realistic broker financing is present. Naked short options are possible. Twenty synthetic paths and three training seeds are too few for a robust performance claim. No statistical significance or live-trading conclusion is claimed.

Useful technical finding: the accounting-corrected environment now supports reproducible, paired policy comparisons, and PPO behavior changes materially with training seed. Next research step would test stronger risk-aware baselines and more realistic market assumptions on a **new** evaluation set; retuning on these 20 seeds would invalidate their held-out status.

## Reproduce and inspect

From repository root, install dependencies, then run:

```bash
python train.py --timesteps 20000 --seed 11 --output-dir experiments/study_20260923/ppo_seed_11
python train.py --timesteps 20000 --seed 23 --output-dir experiments/study_20260923/ppo_seed_23
python train.py --timesteps 20000 --seed 37 --output-dir experiments/study_20260923/ppo_seed_37
python evaluate.py --episodes 20 --output experiments/study_20260923/evaluation \
  --model 11=experiments/study_20260923/ppo_seed_11/model.zip \
  --model 23=experiments/study_20260923/ppo_seed_23/model.zip \
  --model 37=experiments/study_20260923/ppo_seed_37/model.zip
```

`episodes.csv` has all 120 policy-path observations; `summary.csv` has distributions; `paired_differences.csv` has differences from cash and random; `comparison.png` shows return distributions and paired differences from random; `manifest.json` records code version and seeds. All live under ignored `experiments/study_20260923/evaluation/`, so repository does not ship model weights or raw generated outputs. Displayed values above come from code version `b171ff3` and can vary slightly across dependency versions or platforms.

Two earlier seed-11 attempts produced no usable model: one timed out in the execution tool, and another completed learning but could not save outside its filesystem sandbox. They were excluded. Final three saved runs were fresh training runs; no failed-run weights entered evaluation.
