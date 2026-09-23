"""Behavioral checks for paired simulated-policy comparisons."""

import csv
import json

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from evaluate import (
    choose_action,
    paired_differences,
    run_policy_episode,
    run_study,
    summarize_results,
    write_report,
)


def test_cash_and_regime_rule_use_only_observed_state():
    observation = np.zeros(16, dtype=np.float32)
    generator = np.random.default_rng(8)

    assert choose_action("cash", observation, generator) == 4
    observation[14] = 1
    assert choose_action("regime_call", observation, generator) == 0
    observation[14] = -1
    assert choose_action("regime_call", observation, generator) == 2


def test_regime_rule_rejects_observation_without_regime():
    with pytest.raises(ValueError, match="regime"):
        choose_action("regime_call", np.zeros(14), np.random.default_rng(1))


def test_ppo_action_is_deterministic_and_requires_model():
    class Model:
        def predict(self, observation, deterministic):
            assert deterministic is True
            return 1, None

    observation = np.zeros(16)
    generator = np.random.default_rng(1)
    assert choose_action("ppo", observation, generator, model=Model()) == 1
    with pytest.raises(ValueError, match="model"):
        choose_action("ppo", observation, generator)


def test_same_seed_gives_same_market_path_for_different_policies():
    cash = run_policy_episode("cash", seed=10_000, episode_length=4)
    random = run_policy_episode("random", seed=10_000, episode_length=4)
    rule = run_policy_episode("regime_call", seed=10_000, episode_length=4)

    assert cash["spots"] == pytest.approx(random["spots"])
    assert cash["spots"] == pytest.approx(rule["spots"])
    assert cash["regimes"] == random["regimes"] == rule["regimes"]


def test_random_actions_and_results_repeat_with_same_seed():
    first = run_policy_episode("random", seed=10_001, episode_length=5)
    second = run_policy_episode("random", seed=10_001, episode_length=5)

    assert first["actions"] == second["actions"]
    assert first["final_portfolio_value"] == pytest.approx(
        second["final_portfolio_value"]
    )


def test_return_drawdown_and_spread_come_from_ledger_not_shaped_reward():
    result = run_policy_episode("cash", seed=10_002, episode_length=3)
    values = np.asarray(result["portfolio_values"])

    assert result["terminal_return_pct"] == pytest.approx(
        100 * (values[-1] / values[0] - 1)
    )
    assert result["max_drawdown_pct"] == pytest.approx(
        100 * np.max(1 - values / np.maximum.accumulate(values))
    )
    assert result["spread_paid_dollars"] == 0
    assert result["final_portfolio_value"] == pytest.approx(values[-1])


def test_summary_keeps_ppo_training_seeds_separate():
    rows = [
        {"policy": "ppo", "training_seed": 11, "terminal_return_pct": 1.0},
        {"policy": "ppo", "training_seed": 11, "terminal_return_pct": 3.0},
        {"policy": "ppo", "training_seed": 23, "terminal_return_pct": -2.0},
    ]

    summaries = summarize_results(rows, metric="terminal_return_pct")

    assert len(summaries) == 2
    assert summaries[("ppo", 11)]["mean"] == pytest.approx(2.0)
    assert summaries[("ppo", 11)]["n"] == 2
    assert summaries[("ppo", 23)]["mean"] == pytest.approx(-2.0)


def test_study_runs_all_baselines_and_each_ppo_training_seed():
    class HoldModel:
        def predict(self, observation, deterministic=True):
            return 4, None

    rows = run_study(seeds=[10_000, 10_001], models={11: HoldModel()}, episode_length=2)

    assert len(rows) == 8
    assert {row["policy"] for row in rows} == {"cash", "random", "regime_call", "ppo"}
    assert {row["training_seed"] for row in rows if row["policy"] == "ppo"} == {11}
    for seed in (10_000, 10_001):
        paths = [row["spots"] for row in rows if row["seed"] == seed]
        assert all(path == pytest.approx(paths[0]) for path in paths)


def test_paired_differences_compare_each_path_to_same_seed_random():
    rows = [
        {"policy": "cash", "seed": 10_000, "training_seed": None, "terminal_return_pct": 1.0},
        {"policy": "random", "seed": 10_000, "training_seed": None, "terminal_return_pct": -2.0},
        {"policy": "regime_call", "seed": 10_000, "training_seed": None, "terminal_return_pct": 3.0},
        {"policy": "ppo", "seed": 10_000, "training_seed": 11, "terminal_return_pct": 0.0},
    ]

    paired = paired_differences(rows)

    assert len(paired) == 2
    assert paired[0]["policy"] == "regime_call"
    assert paired[0]["vs_random_pct_points"] == pytest.approx(5.0)
    assert paired[0]["vs_cash_pct_points"] == pytest.approx(2.0)
    assert paired[1]["vs_random_pct_points"] == pytest.approx(2.0)


def test_report_preserves_raw_rows_protocol_and_simulation_warning(tmp_path):
    rows = run_study(seeds=[10_000], models={}, episode_length=2)

    write_report(rows, tmp_path, episode_length=2, model_paths={})

    with (tmp_path / "episodes.csv").open(newline="") as file:
        saved_rows = list(csv.DictReader(file))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert len(saved_rows) == 3
    assert {int(row["seed"]) for row in saved_rows} == {10_000}
    assert manifest["evaluation_seeds"] == [10_000]
    assert manifest["episode_length"] == 2
    assert "simulated" in manifest["warning"].lower()
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "paired_differences.csv").exists()
    assert (tmp_path / "comparison.png").exists()
