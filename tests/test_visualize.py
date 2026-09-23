"""Checks for reader-facing episode diagnostics."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from visualize import episode_diagnostics, plot_episode_summary


def test_episode_diagnostics_reports_pnl_and_peak_to_trough_loss():
    values = [10_000, 10_100, 9_900, 10_200]

    daily_pnl, drawdown = episode_diagnostics(values)

    np.testing.assert_allclose(daily_pnl, [100, -200, 300])
    np.testing.assert_allclose(drawdown, [0, 0, (9_900 / 10_100 - 1) * 100, 0])


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
