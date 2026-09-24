"""Behavioral checks for paired simulated-policy comparisons."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from evaluate import (
    choose_action,
    paired_differences,
    run_policy_episode,
    run_regime_study,
    run_study,
    summarize_results,
    write_regime_report,
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


@pytest.mark.parametrize(("scenario", "expected_regime"), [("bull", 1), ("bear", -1)])
def test_controlled_regime_episode_keeps_regime_fixed(scenario, expected_regime):
    result = run_policy_episode(
        "cash", seed=20_000, episode_length=4, regime_scenario=scenario
    )

    assert result["regime_scenario"] == scenario
    assert result["regimes"] == [expected_regime] * 5
    assert result["portfolio_values"][0] == pytest.approx(10_000)
    assert len(result["portfolio_values"]) == 5


def test_controlled_regime_rejects_unknown_scenario():
    with pytest.raises(ValueError, match="regime scenario"):
        run_policy_episode("cash", seed=20_000, episode_length=2, regime_scenario="sideways")


def test_regime_study_pairs_paths_across_all_methods_and_scenarios():
    rows = run_regime_study(seeds=[20_000], episode_length=3)

    assert len(rows) == 6
    assert {row["regime_scenario"] for row in rows} == {"bull", "bear"}
    for scenario in ("bull", "bear"):
        scenario_rows = [row for row in rows if row["regime_scenario"] == scenario]
        assert {row["policy"] for row in scenario_rows} == {"cash", "random", "regime_call"}
        assert all(row["spots"] == pytest.approx(scenario_rows[0]["spots"]) for row in scenario_rows)
        assert all(row["regimes"] == scenario_rows[0]["regimes"] for row in scenario_rows)


def test_regime_report_writes_step_level_pnl_and_plot(tmp_path):
    rows = run_regime_study(seeds=[20_000], episode_length=2)

    write_regime_report(rows, tmp_path, episode_length=2)

    with (tmp_path / "regime_trajectories.csv").open(newline="") as file:
        trajectories = list(csv.DictReader(file))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert len(trajectories) == 18
    assert {row["regime_scenario"] for row in trajectories} == {"bull", "bear"}
    assert {row["step"] for row in trajectories} == {"0", "1", "2"}
    assert {row["cumulative_pnl_dollars"] for row in trajectories if row["step"] == "0"} == {"0.0"}
    assert manifest["regime_scenarios"] == {"bull": 1, "bear": -1}
    assert manifest["evaluation_seeds"] == [20_000]
    assert (tmp_path / "regime_pnl.png").exists()


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


def test_command_line_runs_baseline_pilot_and_saves_report(tmp_path):
    output = tmp_path / "pilot"
    result = subprocess.run(
        [
            sys.executable, "-B", "evaluate.py", "--episodes", "1",
            "--episode-length", "2", "--output", str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (output / "episodes.csv").exists()
    assert "simulated" in result.stdout.lower()


def test_command_line_runs_controlled_regime_pilot(tmp_path):
    output = tmp_path / "regime-pilot"
    result = subprocess.run(
        [
            sys.executable, "-B", "evaluate.py", "--regime-only", "--episodes", "1",
            "--episode-length", "2", "--seed-start", "20_000", "--output", str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (output / "regime_trajectories.csv").exists()
    assert (output / "regime_pnl.png").exists()
    assert "bull" in result.stdout.lower()
