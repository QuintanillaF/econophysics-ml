"""Test suite.

The tests are written against *properties that must hold mathematically*, not
against numbers copied from a previous run. That distinction matters: a snapshot
test tells you the code still does what it did yesterday, while a property test
tells you it does what it is supposed to do. Where an analytic result exists —
the normal-VaR quantile, the convergence of Monte Carlo to the parametric
answer, the square-root-of-time rule — it is checked directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from varengine import (
    GarchParams,
    Portfolio,
    basel_traffic_light,
    christoffersen_independence,
    compare_methods,
    conditional_coverage,
    expected_shortfall,
    historical_var,
    kupiec_pof,
    monte_carlo_var,
    parametric_var,
    rolling_var_backtest,
    run_full_backtest,
    scale_horizon,
    simulate_market,
)

# ---------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return simulate_market(["AAA", "BBB", "CCC"], n_days=1500, seed=7)


@pytest.fixture(scope="module")
def book(prices: pd.DataFrame) -> Portfolio:
    return Portfolio(prices, value=1_000_000.0)


@pytest.fixture(scope="module")
def normal_returns() -> pd.Series:
    """Exactly normal returns, so analytic VaR is known in closed form."""
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0.0, 0.01, 200_000))


# ---------------------------------------------------------------- data

class TestData:
    def test_shape_and_positivity(self, prices):
        assert prices.shape == (1500, 3)
        assert (prices > 0).all().all()

    def test_deterministic_given_seed(self):
        a = simulate_market(["X"], n_days=200, seed=99)
        b = simulate_market(["X"], n_days=200, seed=99)
        pd.testing.assert_frame_equal(a, b)

    def test_garch_produces_fat_tails(self, prices):
        """The whole point of the generator: excess kurtosis above normal."""
        r = np.log(prices / prices.shift(1)).dropna()
        assert r.kurtosis().min() > 1.0, "returns should be leptokurtic"

    def test_garch_produces_volatility_clustering(self, prices):
        """Squared returns should be autocorrelated even though returns are not."""
        r = np.log(prices["AAA"] / prices["AAA"].shift(1)).dropna()
        acf_sq = r.pow(2).autocorr(lag=1)
        assert acf_sq > 0.05, f"expected clustering, got acf(r^2)={acf_sq:.3f}"

    def test_nonstationary_params_rejected(self):
        with pytest.raises(ValueError, match="not stationary"):
            GarchParams(alpha=0.5, beta=0.6)

    def test_nu_below_two_rejected(self):
        with pytest.raises(ValueError, match="nu must exceed 2"):
            GarchParams(nu=1.5)


# ---------------------------------------------------------------- portfolio

class TestPortfolio:
    def test_default_weights_are_equal(self, book):
        assert np.allclose(book.w, 1 / 3)

    def test_weights_normalised(self, prices):
        p = Portfolio(prices, weights=[2.0, 2.0, 2.0])
        assert np.isclose(p.w.sum(), 1.0)

    def test_dict_weights_follow_column_order(self, prices):
        p = Portfolio(prices, weights={"CCC": 0.5, "AAA": 0.3, "BBB": 0.2})
        assert np.allclose(p.w, [0.3, 0.2, 0.5])

    def test_missing_weight_rejected(self, prices):
        with pytest.raises(ValueError, match="no weight given"):
            Portfolio(prices, weights={"AAA": 1.0})

    def test_analytic_vol_matches_realised(self, book):
        """sqrt(w' Sigma w) must agree with the realised series std."""
        assert book.volatility == pytest.approx(book.returns.std(ddof=1), rel=0.02)

    def test_risk_contributions_sum_to_one(self, book):
        assert book.marginal_var_contribution().sum() == pytest.approx(1.0)

    def test_diversification_reduces_volatility(self, prices):
        """A blend cannot be riskier than the average of its parts."""
        blend = Portfolio(prices).volatility
        singles = [
            Portfolio(prices[[c]].copy()).volatility for c in prices.columns
        ]
        assert blend < np.mean(singles)


# ---------------------------------------------------------------- VaR core

class TestVaRProperties:
    def test_var_is_positive_loss(self, book):
        assert historical_var(book.returns).var > 0

    def test_es_never_below_var(self, book):
        """ES is the mean beyond the threshold, so it must dominate VaR."""
        for conf in (0.90, 0.95, 0.99):
            res = historical_var(book.returns, conf)
            assert res.expected_shortfall >= res.var

    def test_var_increases_with_confidence(self, book):
        vars_ = [historical_var(book.returns, c).var for c in (0.90, 0.95, 0.99)]
        assert vars_ == sorted(vars_), "VaR must be monotone in confidence"

    def test_too_few_observations_rejected(self):
        with pytest.raises(ValueError, match="at least 30"):
            historical_var(pd.Series([0.01, -0.02, 0.005]))

    def test_confidence_outside_range_rejected(self, book):
        with pytest.raises(ValueError, match="confidence must lie"):
            historical_var(book.returns, confidence=1.5)


class TestParametricAgainstTheory:
    def test_normal_var_matches_closed_form(self, normal_returns):
        """For N(0, 0.01) at 99%, VaR = -z_0.01 * sigma = 2.3263 * 0.01."""
        res = parametric_var(normal_returns, 0.99, distribution="normal")
        expected = -stats.norm.ppf(0.01) * 0.01
        assert res.var == pytest.approx(expected, rel=0.02)

    def test_normal_es_matches_closed_form(self, normal_returns):
        """ES = sigma * phi(z_a) / a for a zero-mean normal."""
        res = parametric_var(normal_returns, 0.99, distribution="normal")
        z = stats.norm.ppf(0.01)
        expected = 0.01 * stats.norm.pdf(z) / 0.01
        assert res.expected_shortfall == pytest.approx(expected, rel=0.02)

    def test_historical_recovers_normal_quantile(self, normal_returns):
        """With 200k draws the empirical quantile must find the true one."""
        emp = historical_var(normal_returns, 0.99).var
        theory = -stats.norm.ppf(0.01) * 0.01
        assert emp == pytest.approx(theory, rel=0.03)

    def test_t_gives_higher_var_on_fat_tails(self, book):
        """The core finding this project is built to demonstrate."""
        normal = parametric_var(book.returns, 0.99, distribution="normal").var
        student = parametric_var(book.returns, 0.99, distribution="t").var
        assert student > normal, "t must charge more for fat tails than normal"

    def test_normal_understates_versus_historical(self, book):
        """On leptokurtic data, normal VaR sits below the empirical quantile."""
        normal = parametric_var(book.returns, 0.99, distribution="normal").var
        hist = historical_var(book.returns, 0.99).var
        assert normal < hist


class TestMonteCarlo:
    def test_converges_to_parametric_under_normality(self, book):
        """A free correctness check: same assumptions must give the same answer."""
        mc = monte_carlo_var(
            book.asset_returns, book.w, 0.99,
            n_simulations=200_000, distribution="normal", seed=1,
        ).var
        param = parametric_var(book.returns, 0.99, distribution="normal").var
        assert mc == pytest.approx(param, rel=0.03)

    def test_reproducible_with_seed(self, book):
        kw = {"confidence": 0.99, "n_simulations": 20_000, "seed": 123}
        a = monte_carlo_var(book.asset_returns, book.w, **kw).var
        b = monte_carlo_var(book.asset_returns, book.w, **kw).var
        assert a == b

    def test_t_innovations_raise_var(self, book):
        kw = {"confidence": 0.99, "n_simulations": 100_000, "seed": 5}
        norm = monte_carlo_var(book.asset_returns, book.w, distribution="normal", **kw).var
        tdist = monte_carlo_var(book.asset_returns, book.w, distribution="t", nu=4.0, **kw).var
        assert tdist > norm

    def test_weight_length_mismatch_rejected(self, book):
        with pytest.raises(ValueError, match="weights has"):
            monte_carlo_var(book.asset_returns, np.array([0.5, 0.5]))

    def test_too_few_simulations_rejected(self, book):
        with pytest.raises(ValueError, match="at least 1000"):
            monte_carlo_var(book.asset_returns, book.w, n_simulations=100)


class TestHorizonScaling:
    def test_square_root_of_time(self):
        assert scale_horizon(0.02, 4) == pytest.approx(0.04)
        assert scale_horizon(0.01, 10) == pytest.approx(0.01 * np.sqrt(10))

    def test_applied_consistently(self, book):
        one = historical_var(book.returns, 0.99, horizon_days=1).var
        ten = historical_var(book.returns, 0.99, horizon_days=10).var
        assert ten == pytest.approx(one * np.sqrt(10))


# ---------------------------------------------------------------- backtesting

class TestKupiec:
    def test_perfectly_calibrated_model_passes(self):
        """Exactly 1% exceptions over 1000 days must not be rejected."""
        exc = np.zeros(1000, dtype=bool)
        exc[::100] = True  # 10 exceptions = 1.0%
        assert not kupiec_pof(exc, 0.99).reject

    def test_badly_undercalibrated_model_rejected(self):
        """8% exceptions against a 99% model is indefensible."""
        exc = np.zeros(1000, dtype=bool)
        exc[:80] = True
        assert kupiec_pof(exc, 0.99).reject

    def test_zero_exceptions_handled(self):
        res = kupiec_pof(np.zeros(500, dtype=bool), 0.99)
        assert np.isfinite(res.statistic) and res.statistic >= 0

    def test_all_exceptions_handled(self):
        res = kupiec_pof(np.ones(500, dtype=bool), 0.99)
        assert np.isfinite(res.statistic) and res.reject

    def test_statistic_is_non_negative(self):
        rng = np.random.default_rng(3)
        for _ in range(25):
            exc = rng.random(500) < 0.01
            assert kupiec_pof(exc, 0.99).statistic >= 0


class TestChristoffersen:
    def test_clustered_exceptions_rejected(self):
        """Ten breaches on ten consecutive days: right count, wrong pattern."""
        exc = np.zeros(1000, dtype=bool)
        exc[500:510] = True
        assert christoffersen_independence(exc).reject

    def test_spread_exceptions_pass(self):
        exc = np.zeros(1000, dtype=bool)
        exc[::100] = True
        assert not christoffersen_independence(exc).reject

    def test_kupiec_blind_to_clustering(self):
        """Motivates the independence test: same count, opposite verdict."""
        clustered = np.zeros(1000, dtype=bool)
        clustered[500:510] = True
        spread = np.zeros(1000, dtype=bool)
        spread[::100] = True

        assert kupiec_pof(clustered, 0.99).reject == kupiec_pof(spread, 0.99).reject
        assert christoffersen_independence(clustered).reject
        assert not christoffersen_independence(spread).reject

    def test_degenerate_sequence_does_not_crash(self):
        res = christoffersen_independence(np.zeros(100, dtype=bool))
        assert not res.reject and "too few" in res.detail


class TestConditionalCoverage:
    def test_statistic_is_sum_of_components(self):
        rng = np.random.default_rng(11)
        exc = rng.random(800) < 0.015
        cc = conditional_coverage(exc, 0.99)
        expected = kupiec_pof(exc, 0.99).statistic + christoffersen_independence(exc).statistic
        assert cc.statistic == pytest.approx(expected)

    def test_uses_two_degrees_of_freedom(self):
        exc = np.zeros(1000, dtype=bool)
        exc[::100] = True
        cc = conditional_coverage(exc, 0.99)
        assert cc.critical_value == pytest.approx(stats.chi2.ppf(0.95, df=2))


class TestBaselTrafficLight:
    @pytest.mark.parametrize(
        "n,zone", [(0, "GREEN"), (4, "GREEN"), (5, "YELLOW"), (9, "YELLOW"), (10, "RED"), (25, "RED")]
    )
    def test_zone_boundaries(self, n, zone):
        assert basel_traffic_light(n, 250)["zone"] == zone

    def test_multiplier_increases_with_zone(self):
        g = basel_traffic_light(2, 250)["capital_multiplier"]
        y = basel_traffic_light(7, 250)["capital_multiplier"]
        r = basel_traffic_light(15, 250)["capital_multiplier"]
        assert g < y < r

    def test_negative_exceptions_rejected(self):
        with pytest.raises(ValueError):
            basel_traffic_light(-1)


class TestWalkForward:
    def test_no_lookahead_in_window(self, book):
        """Forecast count must equal sample length minus the training window."""
        bt = rolling_var_backtest(book.returns, window=250, confidence=0.99)
        assert len(bt) == len(book.returns) - 250

    def test_forecasts_are_positive(self, book):
        bt = rolling_var_backtest(book.returns, window=250)
        assert (bt["var_forecast"] > 0).all()

    def test_exception_flag_matches_definition(self, book):
        bt = rolling_var_backtest(book.returns, window=250)
        manual = bt["realised_return"] < -bt["var_forecast"]
        pd.testing.assert_series_equal(bt["exception"], manual, check_names=False)

    def test_insufficient_data_rejected(self, book):
        with pytest.raises(ValueError, match="need more than"):
            rolling_var_backtest(book.returns.iloc[:100], window=250)

    def test_full_backtest_bundle_is_complete(self, book):
        out = run_full_backtest(book.returns, window=250, confidence=0.99)
        for key in ("kupiec", "christoffersen", "conditional_coverage", "basel"):
            assert key in out
        assert out["n_exceptions"] == out["frame"]["exception"].sum()


# ---------------------------------------------------------------- integration

class TestComparison:
    def test_all_methods_present(self, book):
        df = compare_methods(book.asset_returns, book.w, 0.99, portfolio_value=1e6)
        assert len(df) == 5
        assert (df["VaR"] > 0).all()

    def test_currency_amounts_consistent(self, book):
        value = 2_500_000.0
        df = compare_methods(book.asset_returns, book.w, 0.99, portfolio_value=value)
        assert np.allclose(df["VaR_amount"], df["VaR"] * value)

    def test_expected_shortfall_helper_agrees(self, book):
        standalone = expected_shortfall(book.returns, 0.99)
        embedded = historical_var(book.returns, 0.99).expected_shortfall
        assert standalone == pytest.approx(embedded)


class TestDegreesOfFreedomFloor:
    """Regression tests for the nu-clipping fix.

    Maximum-likelihood fitting of Student-t degrees of freedom is unstable on
    short windows and can return nu < 2, where the variance does not exist. That
    used to abort an entire rolling backtest; it is now clipped and flagged.
    """

    def test_extreme_tails_do_not_crash(self):
        rng = np.random.default_rng(17)
        heavy = pd.Series(rng.standard_t(1.4, size=250) * 0.01)  # nu below 2
        res = parametric_var(heavy, 0.99, distribution="t")
        assert np.isfinite(res.var) and res.var > 0

    def test_clip_is_disclosed_not_hidden(self):
        rng = np.random.default_rng(17)
        heavy = pd.Series(rng.standard_t(1.4, size=250) * 0.01)
        res = parametric_var(heavy, 0.99, distribution="t")
        assert "clipped" in res.note

    def test_normal_sample_is_not_clipped(self, normal_returns):
        res = parametric_var(normal_returns[:2000], 0.99, distribution="t")
        assert "clipped" not in res.note

    def test_invalid_floor_rejected(self, book):
        with pytest.raises(ValueError, match="min_nu must exceed 2"):
            parametric_var(book.returns, 0.99, distribution="t", min_nu=1.5)

    def test_rolling_t_backtest_completes(self, book):
        """The scenario that originally broke: 250-day windows, t distribution."""
        bt = rolling_var_backtest(
            book.returns, window=250, confidence=0.99,
            method="parametric", distribution="t",
        )
        assert len(bt) == len(book.returns) - 250
        assert (bt["var_forecast"] > 0).all()
