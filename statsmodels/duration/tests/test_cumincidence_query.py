"""
Tests for interval queries on CumIncidenceRight and for the
post-event (right-continuous) convention exposed by PHReg.

These tests cover:
  * empirical Aalen-Johansen queries and interval splicing,
  * event-boundary semantics of PHReg baseline cumulative hazards,
  * validation and error paths,
  * preservation of the pre-existing CumIncidenceRight interface.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import (
    CumIncidenceRight, FAILURE_FREE_LABEL)

FF = FAILURE_FREE_LABEL

# Data used by the existing R-validated cumulative incidence test.
FTIME = np.r_[1, 1, 2, 4, 4, 4, 6, 6, 7, 8, 9, 9, 9, 1, 2, 2, 4, 4]
FSTAT = np.r_[1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def _empirical_ci():
    return CumIncidenceRight(FTIME, FSTAT)


# ----------------------------------------------------------------------
# Empirical mode
# ----------------------------------------------------------------------

def test_empirical_query_matches_cinc_at_origin_zero():
    ci = _empirical_ci()
    event_times = np.unique(FTIME)
    q = ci.predict(0.0, event_times)
    # At origin 0 everyone is alive, so conditional equals absolute.
    for j, label in enumerate([1, 2, 3]):
        expected = []
        for t in event_times:
            ix = np.searchsorted(ci.times, t, side="right") - 1
            expected.append(ci.cinc[j][ix] if ix >= 0 else 0.0)
        assert_allclose(q[label].values, expected)
    assert_allclose(q[FF].values, 1.0 - sum(ci_c for ci_c in
                                            [q[1], q[2], q[3]]),
                    atol=1e-12)


def test_empirical_closure_and_monotonicity():
    ci = _empirical_ci()
    q = ci.predict(0.0, np.array([0., 1., 2., 4., 6., 7., 9.]))
    assert_allclose(q.sum(axis=1), 1.0, atol=1e-12)
    for label in [1, 2, 3]:
        assert np.all(np.diff(q[label].values) >= -1e-12)
    assert np.all(np.diff(q[FF].values) <= 1e-12)


def test_empirical_hand_km_values():
    time = np.array([1., 1., 2., 2., 3., 3.])
    status = np.array([1, 1, 2, 0, 0, 0])
    ci = CumIncidenceRight(time, status)
    assert_allclose(ci._emp_surv, [2. / 3, 1. / 2])
    q = ci.predict(0.0, np.array([1., 2., 3.]))
    assert_allclose(q[1].values, [1. / 3, 1. / 3, 1. / 3])
    assert_allclose(q[2].values, [0., 1. / 6, 1. / 6])
    assert_allclose(q[FF].values, [2. / 3, 1. / 2, 1. / 2])


def test_empirical_conditional_from_survived_origin():
    time = np.array([1., 1., 2., 2., 3., 3.])
    status = np.array([1, 1, 2, 0, 0, 0])
    ci = CumIncidenceRight(time, status)
    q = ci.predict(1.0, np.array([1., 2., 3.]))
    assert_allclose(q[FF].values, [1.0, 0.75, 0.75])
    assert_allclose(q[2].values, [0.0, 0.25, 0.25])
    assert_allclose(q[1].values, 0.0)
    assert_allclose(q.sum(axis=1), 1.0)


def test_empirical_interval_splicing():
    ci = _empirical_ci()
    o, tm, t2 = 0.0, 4.0, 9.0
    full = ci.predict(o, np.array([t2]))
    head = ci.predict(o, np.array([tm]))
    tail = ci.predict(tm, np.array([t2]))
    for label in [1, 2, 3]:
        lhs = full[label].iloc[0]
        rhs = head[label].iloc[0] + \
            head[FF].iloc[0] * tail[label].iloc[0]
        assert_allclose(lhs, rhs, atol=1e-12)
    assert_allclose(full[FF].iloc[0],
                    head[FF].iloc[0] * tail[FF].iloc[0], atol=1e-12)


def test_empirical_repeated_times_and_origin_equals_time():
    ci = _empirical_ci()
    q = ci.predict(0.0, np.array([2., 2., 4.]))
    assert q.index.tolist() == [2., 2., 4.]
    assert_allclose(q.iloc[0].values, q.iloc[1].values)
    q0 = ci.predict(3.0, np.array([3.0]))
    assert_allclose(q0[[1, 2, 3]].values, 0.0)
    assert_allclose(q0[FF].iloc[0], 1.0)


@pytest.mark.parametrize("origin,times", [
    (-1.0, [1.]),
    (np.nan, [1.]),
    (np.inf, [1.]),
    (1.0, [0.5]),
    (1.0, [2., 1.]),
    (1.0, [np.inf]),
    (1.0, [np.nan]),
])
def test_empirical_invalid_arguments(origin, times):
    ci = _empirical_ci()
    with pytest.raises(ValueError):
        ci.predict(origin, np.array(times))


def test_empirical_times_must_be_1d_nonempty():
    ci = _empirical_ci()
    with pytest.raises(ValueError):
        ci.predict(0.0, np.array([[1., 2.]]))
    with pytest.raises(ValueError):
        ci.predict(0.0, np.array([]))


def test_empirical_weights_and_exog_rejected_for_query():
    ciw = CumIncidenceRight(FTIME, FSTAT,
                            freq_weights=np.ones(len(FTIME)))
    with pytest.raises(ValueError):
        ciw.predict(0.0, np.array([1.]))
    cix = CumIncidenceRight(FTIME, FSTAT,
                            exog=np.random.normal(size=(len(FTIME), 2)))
    with pytest.raises(ValueError):
        cix.predict(0.0, np.array([1.]))
    # Fitted-mode arguments are not accepted by empirical curves.
    with pytest.raises(ValueError):
        _empirical_ci().predict(0.0, np.array([1.]),
                                exog=np.ones((1, 1)),
                                strata=np.zeros(1, dtype=int))


def test_empirical_no_survivable_mass_at_origin():
    ci = CumIncidenceRight(np.array([2., 2., 2.]),
                           np.array([1, 1, 2]))
    with pytest.raises(ValueError):
        ci.predict(2.0, np.array([3.]))
    # But querying up to that final time is still well defined.
    q = ci.predict(0.0, np.array([2.]))
    assert_allclose(q.sum(axis=1), 1.0)


def test_empirical_preserves_legacy_attributes():
    ci = _empirical_ci()
    # Pre-existing attributes and default behaviour unchanged.
    assert len(ci.cinc) == 3
    assert ci.cinc_se is not None
    assert hasattr(ci, "times")
    assert hasattr(ci, "title")
    # Legacy point estimates still match the R reference values.
    assert_allclose(ci.cinc[0],
                    [0.11111111, 0.17037037, 0.17037037, 0.17037037,
                     0.17037037, 0.17037037, 0.17037037])


# ----------------------------------------------------------------------
# PHReg post-event semantics
# ----------------------------------------------------------------------

@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_phreg_post_event_baseline_steps(ties):
    rng = np.random.default_rng(0)
    time = np.array([1., 2., 2., 4., 5., 6.])
    status = np.array([1, 1, 0, 1, 0, 1])
    exog = rng.normal(size=(6, 2))
    mod = PHReg(time, exog, status, ties=ties)
    res = mod.fit()
    pre = mod.baseline_cumulative_hazard(res.params, post_event=False)
    post = mod.baseline_cumulative_hazard(res.params, post_event=True)
    h_pre, h_post = pre[0][1], post[0][1]
    # Default behaviour is the pre-event (left-continuous) convention.
    default = mod.baseline_cumulative_hazard(res.params)
    assert_allclose(default[0][1], h_pre)
    # Post-event includes each jump at the event time.
    jumps = np.diff(np.r_[0.0, h_post])
    assert_allclose(h_pre, np.r_[0.0, h_post[:-1]])
    assert_allclose(np.cumsum(jumps), h_post)
    assert_allclose(post[0][2], np.exp(-h_post))


def test_phreg_post_event_function_boundary_values():
    rng = np.random.default_rng(0)
    time = np.array([1., 2., 2., 4., 5., 6.])
    status = np.array([1, 1, 0, 1, 0, 1])
    exog = rng.normal(size=(6, 2))
    mod = PHReg(time, exog, status)
    res = mod.fit()
    f_post = mod.baseline_cumulative_hazard_function(
        res.params, post_event=True)
    f_pre = mod.baseline_cumulative_hazard_function(
        res.params, post_event=False)
    q = np.array([0., 0.9, 1., 1.5, 2., 6., 100.])
    post_vals = f_post[0](q)
    pre_vals = f_pre[0](q)
    assert post_vals[0] == 0.0 and post_vals[1] == 0.0
    # Final jump carried forward.
    h_post = mod.baseline_cumulative_hazard(res.params, True)[0][1]
    assert_allclose(post_vals[-1], h_post[-1])
    # Scalar call returns a scalar.
    assert np.isscalar(f_post[0](2.0))
    # Pre-event at the first event is still zero.
    assert pre_vals[2] == 0.0


def test_phreg_predict_post_event_consistency():
    rng = np.random.default_rng(0)
    time = np.array([1., 2., 2., 4., 5., 6.])
    status = np.array([1, 1, 0, 1, 0, 1])
    exog = rng.normal(size=(6, 2))
    mod = PHReg(time, exog, status)
    res = mod.fit()
    xq = rng.normal(size=(1, 2))
    sv = res.predict(endog=np.array([2.]), exog=xq,
                     pred_type="surv", post_event=True).predicted_values
    ch = res.predict(endog=np.array([2.]), exog=xq,
                     pred_type="cumhaz", post_event=True).predicted_values
    assert_allclose(sv, np.exp(-ch))
    sv_pre = res.predict(endog=np.array([2.]), exog=xq,
                         pred_type="surv",
                         post_event=False).predicted_values
    h_pre = mod.baseline_cumulative_hazard(res.params, False)[0][1][1]
    hr = np.exp(np.dot(xq, res.params))[0]
    assert_allclose(sv_pre, np.exp(-h_pre * hr))
    # Default predict keeps the old (pre-event) behaviour.
    sv_default = res.predict(endog=np.array([2.]), exog=xq,
                             pred_type="surv").predicted_values
    assert_allclose(sv_default, sv_pre)


def test_phreg_post_event_unknown_stratum_errors():
    rng = np.random.default_rng(0)
    time = np.array([1., 2., 2., 4., 5., 6.])
    status = np.array([1, 1, 0, 1, 0, 1])
    exog = rng.normal(size=(6, 2))
    strata = np.array(["a", "a", "a", "b", "b", "b"])
    res = PHReg(time, exog, status, strata=strata).fit()
    with pytest.raises(KeyError):
        res.predict(endog=np.array([2.]), exog=exog[:1],
                    strata=np.array(["c"]), pred_type="cumhaz",
                    post_event=True)
