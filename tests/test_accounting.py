"""Cash-flow and reward invariants for the simulated options ledger."""

import pytest

from env import OptionsEnv


def make_env(**kwargs):
    env = OptionsEnv(
        use_regime=False,
        use_stochastic_vol=False,
        use_reward_shaping=False,
        **kwargs,
    )
    env.reset(seed=7)
    return env


def test_buy_call_pays_ask_and_marks_position_at_mid():
    env = make_env()
    mid = env.call_price
    half_spread = mid * env.transaction_cost / 2

    reported_cost = env._execute_action(env.ACTION_BUY_CALL)

    assert env.call_position == 1
    assert env.cash == pytest.approx(env.initial_cash - (mid + half_spread) * 100)
    assert env._calculate_portfolio_value() == pytest.approx(
        env.initial_cash - half_spread * 100
    )
    assert reported_cost == pytest.approx(half_spread * 100)


def test_sell_put_receives_bid_and_marks_liability_at_mid():
    env = make_env()
    mid = env.put_price
    half_spread = mid * env.transaction_cost / 2

    env._execute_action(env.ACTION_SELL_PUT)

    assert env.put_position == -1
    assert env.cash == pytest.approx(env.initial_cash + (mid - half_spread) * 100)
    assert env._calculate_portfolio_value() == pytest.approx(
        env.initial_cash - half_spread * 100
    )


def test_reversing_short_call_uses_two_executable_fills():
    env = make_env()
    mid = env.call_price
    half_spread = mid * env.transaction_cost / 2
    env._execute_action(env.ACTION_SELL_CALL)

    reported_cost = env._execute_action(env.ACTION_BUY_CALL)

    assert env.call_position == 1
    assert env.cash == pytest.approx(
        env.initial_cash + (mid - half_spread) * 100 - 2 * (mid + half_spread) * 100
    )
    assert reported_cost == pytest.approx(2 * half_spread * 100)


def test_hold_changes_no_cash_or_position():
    env = make_env()
    original_cash = env.cash

    reported_cost = env._execute_action(env.ACTION_HOLD)

    assert env.cash == original_cash
    assert env.call_position == 0
    assert env.put_position == 0
    assert reported_cost == 0


def test_final_reward_uses_value_after_liquidation():
    env = make_env(episode_length=1)
    initial_value = env._calculate_portfolio_value()

    obs, reward, terminated, truncated, info = env.step(env.ACTION_BUY_CALL)

    assert not terminated
    assert truncated
    assert env.call_position == 0
    assert obs[10] == 0
    assert info["portfolio_value"] == pytest.approx(env.cash)
    assert reward == pytest.approx((env.cash - initial_value) / env.initial_cash)


def test_short_put_terminal_close_pays_ask():
    env = make_env(episode_length=1, risk_free_rate=0)
    opening_bid = env.put_price * (1 - env.transaction_cost / 2)
    env.step(env.ACTION_SELL_PUT)
    final_ask = env.put_price * (1 + env.transaction_cost / 2)

    assert env.cash == pytest.approx(env.initial_cash + 100 * (opening_bid - final_ask))
    assert env.put_position == 0


def test_expiry_settles_call_at_intrinsic_value_without_spread():
    env = make_env(episode_length=4, risk_free_rate=0)
    env.initial_tte = 1 / 252
    env.reset(seed=7)
    opening_ask = env.call_price * (1 + env.transaction_cost / 2)

    _, _, terminated, truncated, info = env.step(env.ACTION_BUY_CALL)
    intrinsic = max(env.spot - env.strike, 0)

    assert terminated
    assert not truncated
    assert env.call_position == 0
    assert info["portfolio_value"] == pytest.approx(
        env.initial_cash - 100 * opening_ask + 100 * intrinsic
    )


def test_each_unshaped_reward_reconciles_with_account_value():
    env = make_env(episode_length=4)
    previous_value = env._calculate_portfolio_value()

    for action in (
        env.ACTION_BUY_CALL,
        env.ACTION_HOLD,
        env.ACTION_SELL_CALL,
        env.ACTION_BUY_PUT,
    ):
        _, reward, _, _, info = env.step(action)
        current_value = info["portfolio_value"]
        assert reward == pytest.approx(
            (current_value - previous_value) / env.initial_cash
        )
        previous_value = current_value
