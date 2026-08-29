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
    PurgedKFold,
    basel_traffic_light,
    christoffersen_independence,
    compare_methods,
    conditional_coverage,
    deflated_sharpe_ratio,
    es_backtest_acerbi_szekely,
    evt_expected_shortfall,
    evt_var,
    ewma_volatility,
    expected_max_sharpe,
    expected_shortfall,
    factor_betas,
    filtered_historical_var,
    fit_factor_model,
    fit_gpd,
    garch11_fit,
    historical_stress,
    historical_var,
    kupiec_pof,
    ledoit_wolf_cov,
    monte_carlo_var,
    parametric_stress,
    parametric_var,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    reverse_stress,
    rolling_var_backtest,
    run_full_backtest,
    scale_horizon,
    simulate_market,
)
from varengine.stress import HISTORICAL_SCENARIOS, SHOCK_LIBRARY

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
        # historical, parametric normal/t, monte carlo normal/t, filtered-historical
        assert len(df) == 6
        assert (df["VaR"] > 0).all()
        assert any("filtered-historical" in m for m in df.index)

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


# ---------------------------------------------------------------- conditional volatility

class TestConditionalVolatility:
    def test_ewma_recursion_matches_definition(self):
        rng = np.random.default_rng(1)
        r = pd.Series(rng.normal(0, 0.01, 500))
        lam = 0.94
        sigma = ewma_volatility(r, lam)
        # Reproduce a couple of steps of the recursion by hand.
        seed = float(np.var(r.iloc[:20], ddof=1))
        v = seed
        for i in range(3):
            assert sigma.iloc[i] == pytest.approx(np.sqrt(v), rel=1e-9)
            v = lam * v + (1 - lam) * r.iloc[i] ** 2

    def test_ewma_reacts_faster_than_rolling_window(self):
        """A vol spike should move EWMA more than a 250-day equal-weight sigma."""
        calm = np.full(400, 0.005)
        spike = np.concatenate([calm, np.full(10, 0.05)])
        r = pd.Series(np.random.default_rng(2).normal(0, 1, spike.size) * spike)
        ewma = ewma_volatility(r, 0.94).iloc[-1]
        rolling = r.iloc[-250:].std()
        assert ewma > rolling

    def test_garch_fit_recovers_persistence(self):
        """GARCH on simulate_market data should find a high, stationary persistence."""
        px = simulate_market(["X"], n_days=3000, seed=3)
        r = np.log(px["X"] / px["X"].shift(1)).dropna()
        model = garch11_fit(r)
        assert 0.80 < model.persistence < 1.0
        assert model.params["omega"] > 0

    def test_garch_multiday_forecast_mean_reverts(self):
        px = simulate_market(["X"], n_days=2000, seed=4)
        r = np.log(px["X"] / px["X"].shift(1)).dropna()
        model = garch11_fit(r)
        # Horizon forecasts are monotone and, because persistence < 1, the
        # per-day variance implied by forecast(h) converges to the long-run level.
        fs = [model.forecast(h) for h in (1, 5, 10, 60, 250)]
        assert fs == sorted(fs)
        long_run_daily_vol = np.sqrt(model._long_run_var)
        assert model.forecast(250) / np.sqrt(250) == pytest.approx(
            long_run_daily_vol, rel=0.25
        )


# ---------------------------------------------------------------- filtered historical simulation

class TestFilteredHistorical:
    def test_var_properties_hold(self, book):
        res = filtered_historical_var(book.returns, 0.99, vol="ewma")
        assert res.var > 0
        assert res.expected_shortfall >= res.var

    def test_monotone_in_confidence(self, book):
        vars_ = [filtered_historical_var(book.returns, c, vol="ewma").var
                 for c in (0.90, 0.95, 0.99)]
        assert vars_ == sorted(vars_)

    def test_reacts_to_current_volatility(self):
        """FHS scales by today's vol: the same history, ended calm vs ended wild,
        must give different VaR."""
        rng = np.random.default_rng(5)
        base = rng.standard_t(6, 600) * 0.01
        calm_tail = rng.standard_t(6, 60) * 0.003
        wild_tail = rng.standard_t(6, 60) * 0.03
        calm = pd.Series(np.concatenate([base, calm_tail]))
        wild = pd.Series(np.concatenate([base, wild_tail]))
        assert (filtered_historical_var(wild, 0.99, vol="ewma").var
                > filtered_historical_var(calm, 0.99, vol="ewma").var)

    def test_included_in_compare_methods(self, book):
        df = compare_methods(book.asset_returns, book.w, 0.99)
        assert any("filtered-historical" in m for m in df.index)

    def test_rolling_fhs_backtest_completes(self, book):
        bt = rolling_var_backtest(book.returns, window=250, method="fhs", distribution="ewma")
        assert len(bt) == len(book.returns) - 250
        assert (bt["var_forecast"] > 0).all()
        assert (bt["es_forecast"] >= bt["var_forecast"]).all()


# ---------------------------------------------------------------- extreme value theory

class TestEVT:
    def test_gpd_finds_heavy_tail_on_t_data(self):
        rng = np.random.default_rng(6)
        r = pd.Series(rng.standard_t(4, 4000) * 0.01)
        fit = fit_gpd(r, threshold_q=0.95)
        assert fit.xi > 0            # heavy tail
        assert fit.n_exceed >= 10

    def test_evt_var_exceeds_historical_in_far_tail(self, book):
        """Where historical simulation runs out of data, EVT should be at least
        as conservative."""
        evt = evt_var(book.returns, 0.995).var
        hist = historical_var(book.returns, 0.995).var
        assert evt >= hist * 0.95   # allow a little slack; EVT smooths the tail

    def test_evt_es_dominates_var(self, book):
        res = evt_var(book.returns, 0.99)
        assert res.expected_shortfall >= res.var

    def test_confidence_inside_threshold_rejected(self, book):
        with pytest.raises(ValueError, match="inside the threshold"):
            evt_var(book.returns, 0.90, threshold_q=0.95)


# ---------------------------------------------------------------- covariance shrinkage + component VaR

class TestShrinkageAndComponentVaR:
    def test_ledoit_wolf_is_psd(self, book):
        cov = ledoit_wolf_cov(book.asset_returns)
        assert np.all(np.linalg.eigvalsh(cov) > -1e-12)

    def test_shrinkage_beats_sample_on_short_window(self):
        """On a short sample the shrunk estimate should be closer to the truth."""
        rng = np.random.default_rng(7)
        n = 8
        true_cov = 0.0001 * (0.3 * np.ones((n, n)) + 0.7 * np.eye(n))
        errs_sample, errs_lw = [], []
        for s in range(20):
            r = pd.DataFrame(
                np.random.default_rng(s).multivariate_normal(np.zeros(n), true_cov, 40)
            )
            sample = r.cov().to_numpy()
            lw = ledoit_wolf_cov(r)
            errs_sample.append(np.linalg.norm(sample - true_cov))
            errs_lw.append(np.linalg.norm(lw - true_cov))
        assert np.mean(errs_lw) < np.mean(errs_sample)

    def test_component_var_sums_to_total(self, book):
        cv = book.component_var(0.99, distribution="normal")
        z = stats.norm.ppf(0.01)
        total = -z * book.volatility          # zero-mean parametric VaR
        assert cv["component"].sum() == pytest.approx(total, rel=1e-6)
        assert cv["component_pct"].sum() == pytest.approx(1.0)

    def test_incremental_var_positive_for_a_risk_adding_asset(self):
        """A big, largely independent, high-vol sleeve must raise portfolio VaR."""
        rng = np.random.default_rng(8)
        calm = np.cumprod(1 + rng.normal(0.0003, 0.006, (800, 2)), axis=0) * 100
        wild = np.cumprod(1 + rng.normal(0.0, 0.03, (800, 1)), axis=0) * 100
        px = pd.DataFrame(np.hstack([calm, wild]), columns=["calm1", "calm2", "wild"])
        book = Portfolio(px, weights=[0.35, 0.35, 0.30])
        inc = book.component_var(0.99)["incremental"]
        assert inc["wild"] > 0


# ---------------------------------------------------------------- ES backtesting

class TestESBacktest:
    def test_well_calibrated_es_not_rejected(self):
        """Normal returns, normal VaR/ES forecasts: the ES test should pass."""
        rng = np.random.default_rng(9)
        r = rng.normal(0, 0.01, 4000)
        z = stats.norm.ppf(0.01)
        var = np.full_like(r, -z * 0.01)
        es = np.full_like(r, 0.01 * stats.norm.pdf(z) / 0.01)
        res = es_backtest_acerbi_szekely(r, var, es, 0.99, n_boot=500, seed=1)
        assert not res.reject

    def test_understated_es_is_rejected(self):
        """Fat-tailed losses against an ES that is far too small."""
        rng = np.random.default_rng(10)
        r = rng.standard_t(3, 4000) * 0.01
        z = stats.norm.ppf(0.01)
        var = np.full_like(r, -z * 0.01)
        es = np.full_like(r, 0.011)            # deliberately too low for t(3) tails
        res = es_backtest_acerbi_szekely(r, var, es, 0.99, n_boot=500, seed=2)
        assert res.reject
        assert res.statistic < 0

    def test_in_full_backtest_bundle(self, book):
        out = run_full_backtest(book.returns, window=250, confidence=0.99)
        assert "es_test" in out
        assert np.isfinite(out["es_test"].statistic)


# ---------------------------------------------------------------- stress testing

@pytest.fixture(scope="module")
def factor_returns() -> pd.DataFrame:
    """Synthetic risk-factor return panel, so stress tests run offline."""
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2007-01-01", periods=4500)
    cols = ["equity", "rates", "credit", "usd", "gold", "oil", "crypto"]
    data = rng.multivariate_normal(
        np.zeros(len(cols)),
        0.0001 * (0.2 * np.ones((len(cols), len(cols))) + 0.8 * np.eye(len(cols))),
        len(idx),
    )
    return pd.DataFrame(data, index=idx, columns=cols)


@pytest.fixture(scope="module")
def stress_book(factor_returns) -> Portfolio:
    """A portfolio whose returns are a known linear combination of the factors."""
    betas = {"equity": 1.1, "credit": 0.4, "crypto": 0.2}
    port_ret = sum(b * factor_returns[f] for f, b in betas.items())
    port_ret += np.random.default_rng(12).normal(0, 0.001, len(port_ret))
    prices = 100 * (1 + port_ret).cumprod().to_frame("BOOK")
    return Portfolio(prices, value=1_000_000.0)


class TestStress:
    def test_factor_betas_recover_the_exposure(self, stress_book, factor_returns):
        betas = factor_betas(stress_book, factor_returns)
        assert betas["equity"] == pytest.approx(1.1, abs=0.15)
        assert betas["crypto"] == pytest.approx(0.2, abs=0.15)

    def test_historical_replay_is_a_loss_in_a_crash(self, stress_book, factor_returns):
        # Force a crash into the COVID window of the synthetic factor panel.
        fr = factor_returns.copy()
        win = fr.loc["2020-02-19":"2020-03-23"].index
        fr.loc[win, "equity"] = -0.03
        fr.loc[win, "credit"] = -0.02
        res = historical_stress(stress_book, "COVID crash Mar 2020", fr)
        assert res.pnl_pct < 0

    def test_parametric_equity_crash_loses_money(self, stress_book, factor_returns):
        res = parametric_stress(stress_book, "equity_crash", factor_returns)
        assert res.pnl_pct < 0
        assert res.pnl_amount == pytest.approx(res.pnl_pct * stress_book.value)

    def test_reverse_stress_hits_the_target(self, stress_book, factor_returns):
        res = reverse_stress(stress_book, target_loss_pct=0.10,
                             direction="risk_off", factor_returns=factor_returns)
        assert res.pnl_pct == pytest.approx(-0.10)
        # The scaled moves, fed back through the betas, must reproduce -10%.
        betas = factor_betas(stress_book, factor_returns)
        pnl = sum(betas.get(f, 0.0) * m for f, m in res.factor_moves.items())
        assert pnl == pytest.approx(-0.10, abs=1e-6)

    def test_crypto_scenarios_present(self):
        for name in ("Terra / UST May 2022", "FTX collapse Nov 2022"):
            assert name in HISTORICAL_SCENARIOS
        for name in ("stablecoin_depeg", "exchange_insolvency"):
            assert name in SHOCK_LIBRARY

    def test_ftx_scenario_hurts_a_crypto_book(self, factor_returns):
        fr = factor_returns.copy()
        win = fr.loc["2022-11-05":"2022-11-14"].index
        fr.loc[win, "crypto"] = -0.08
        # A book that is mostly crypto.
        port_ret = 0.9 * fr["crypto"] + 0.1 * fr["equity"]
        px = 100 * (1 + port_ret).cumprod().to_frame("CRYPTOBOOK")
        book = Portfolio(px, value=1_000_000.0)
        res = historical_stress(book, "FTX collapse Nov 2022", fr)
        assert res.pnl_pct < 0


# ---------------------------------------------------------------- factor model

@pytest.fixture(scope="module")
def ff_factors() -> pd.DataFrame:
    """A synthetic Fama-French-style panel, so the tests run offline."""
    rng = np.random.default_rng(20)
    idx = pd.bdate_range("2018-01-01", periods=1200)
    cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
    data = rng.normal(0, 0.008, (len(idx), len(cols)))
    df = pd.DataFrame(data, index=idx, columns=cols)
    df["RF"] = 0.00008
    return df


class TestFactorModel:
    def test_recovers_known_betas(self, ff_factors):
        rng = np.random.default_rng(21)
        true = {"Mkt-RF": 1.15, "SMB": 0.40, "HML": -0.30, "Mom": 0.20}
        port = ff_factors["RF"].copy()
        for f, b in true.items():
            port = port + b * ff_factors[f]
        port = port + rng.normal(0, 0.002, len(port))
        port.name = "port"

        fm = fit_factor_model(port, factors=ff_factors)
        for f, b in true.items():
            assert fm.betas[f] == pytest.approx(b, abs=0.08)
        assert fm.r_squared > 0.8

    def test_risk_contributions_sum_to_one(self, ff_factors):
        rng = np.random.default_rng(22)
        port = (1.0 * ff_factors["Mkt-RF"] + 0.3 * ff_factors["HML"]
                + ff_factors["RF"] + rng.normal(0, 0.003, len(ff_factors)))
        port.name = "port"
        fm = fit_factor_model(port, factors=ff_factors)
        assert fm.risk_contribution.sum() == pytest.approx(1.0, abs=1e-6)
        assert 0.0 <= fm.idiosyncratic_share <= 1.0

    def test_pure_noise_has_low_r_squared_and_high_idio(self, ff_factors):
        rng = np.random.default_rng(23)
        port = pd.Series(rng.normal(0, 0.01, len(ff_factors)), index=ff_factors.index, name="port")
        fm = fit_factor_model(port, factors=ff_factors)
        assert fm.r_squared < 0.1
        assert fm.idiosyncratic_share > 0.9


# ---------------------------------------------------------------- backtest diagnostics

class TestDiagnostics:
    def test_psr_high_for_a_clearly_profitable_series(self):
        rng = np.random.default_rng(30)
        r = rng.normal(0.001, 0.008, 2000)   # annual Sharpe ~2
        assert probabilistic_sharpe_ratio(r, sr_benchmark=0.0) > 0.99

    def test_psr_about_half_at_own_sharpe(self):
        rng = np.random.default_rng(31)
        r = rng.normal(0.0005, 0.01, 3000)
        sr = r.mean() / r.std(ddof=1) * np.sqrt(252)
        assert probabilistic_sharpe_ratio(r, sr_benchmark=sr) == pytest.approx(0.5, abs=0.05)

    def test_deflation_threshold_rises_with_trials(self):
        lo = expected_max_sharpe(5, trials_sr_std=0.5 / np.sqrt(252))
        hi = expected_max_sharpe(500, trials_sr_std=0.5 / np.sqrt(252))
        assert hi > lo > 0

    def test_noise_strategy_fails_dsr_after_many_trials(self):
        rng = np.random.default_rng(32)
        r = pd.Series(rng.normal(0.0, 0.01, 1500))  # zero edge
        res = deflated_sharpe_ratio(r, n_trials=1000)
        assert res["dsr"] < 0.5
        assert res["verdict"] == "likely overfit"

    def test_pbo_is_high_for_pure_noise_and_low_for_a_real_winner(self):
        """The discriminating property: selecting on noise does not generalise,
        selecting a genuinely-better trial does. PBO on 20 noise trials is high
        (well above 0); adding one trial that dominates everywhere collapses it."""
        rng = np.random.default_rng(33)
        M = rng.normal(0, 0.01, (2000, 20))
        pbo_noise = probability_of_backtest_overfitting(M, n_splits=10)["pbo"]

        M2 = M.copy()
        M2[:, 0] += 0.0015                    # trial 0 dominates in every subset
        pbo_winner = probability_of_backtest_overfitting(M2, n_splits=10)["pbo"]

        assert pbo_noise > 0.30
        assert pbo_winner < 0.15
        assert pbo_winner < pbo_noise

    def test_purged_kfold_leaves_a_gap(self):
        pk = PurgedKFold(n_splits=5, horizon=10, embargo=5)
        n = 200
        for train, test in pk.split(np.arange(n)):
            lo, hi = test.min() - 10, test.max() + 5
            assert not any(lo <= i <= hi for i in train)
        # every observation is tested exactly once
        tested = np.concatenate([te for _, te in pk.split(np.arange(n))])
        assert sorted(tested) == list(range(n))
