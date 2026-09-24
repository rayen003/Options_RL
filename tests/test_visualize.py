"""Checks for reader-facing episode diagnostics."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from env import OptionsEnv
from visualize import episode_diagnostics, plot_episode_summary, run_episode


def test_episode_diagnostics_reports_pnl_and_peak_to_trough_loss():
    values = [10_000, 10_100, 9_900, 10_200]

    daily_pnl, drawdown = episode_diagnostics(values)

    np.testing.assert_allclose(daily_pnl, [100, -200, 300])
    np.testing.assert_allclose(drawdown, [0, 0, (9_900 / 10_100 - 1) * 100, 0])


@pytest.mark.parametrize("values", [[], [10_000], [0, 10], [10_000, np.nan]])
def test_episode_diagnostics_rejects_invalid_series(values):
    with pytest.raises(ValueError):
        episode_diagnostics(values)


def test_episode_summary_has_value_pnl_and_drawdown_panels():
    fig = plot_episode_summary({"portfolio_values": [10_000, 9_900, 10_050]})

    assert [ax.get_ylabel() for ax in fig.axes] == [
        "Portfolio value ($)",
        "Daily P&L ($)",
        "Drawdown (%)",
    ]
    assert fig.axes[0].lines[0].get_ydata().tolist() == pytest.approx(
        [10_000, 9_900, 10_050]
    )
    assert "Simulated episode" in fig._suptitle.get_text()


def test_recorded_episode_uses_final_liquidated_portfolio_value():
    class BuyCallModel:
        def predict(self, observation, deterministic=True):
            return 0, None

    env = OptionsEnv(
        episode_length=2,
        use_regime=False,
        use_stochastic_vol=False,
        use_reward_shaping=False,
    )
    data = run_episode(env, BuyCallModel(), seed=7)

    assert len(data["portfolio_values"]) == 3
    assert data["portfolio_values"][-1] == pytest.approx(env.cash)
    assert data["portfolio_values"][-1] == pytest.approx(
        data["infos"][-1]["portfolio_value"]
    )
