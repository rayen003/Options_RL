"""Behavioral checks for paired simulated-policy comparisons."""

import numpy as np
import pytest

from evaluate import choose_action, run_policy_episode, summarize_results


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
