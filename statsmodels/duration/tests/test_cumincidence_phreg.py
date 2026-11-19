"""
Tests for CumIncidenceRight.from_phreg fitted competing-risk queries.

Covers Breslow/Efron fits, same-day competing events, strata with a
missing cause, delayed entry, prediction offsets, single-outcome
degeneration, cohort compatibility checks and label/order invariance.
"""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import (
    CumIncidenceRight, FAILURE_FREE_LABEL)

FF = FAILURE_FREE_LABEL
TOL = 1e-10


def _competing_data(rng, n=120, two_causes=True):
    x = rng.normal(size=(n, 2))
    t1 = -np.log(rng.uniform(size=n)) / \
        np.exp(x @ np.array([0.5, -0.3]))
    t2 = -np.log(rng.uniform(size=n)) / \
        np.exp(x @ np.array([-0.2, 0.4])) * 2
    tc = rng.uniform(0.5, 6, size=n)
    time = np.minimum(np.minimum(t1, t2), tc)
    if two_causes:
        status = np.where(t1 <= np.minimum(t2, tc), 1,
                          np.where(t2 <= tc, 2, 0))
    else:
        status = np.where(np.minimum(t1, t2) <= tc, 1, 0)
    return time, x, status


def _two_fits(time, x, status, **kw):
    f1 = PHReg(time, x, (status == 1).astype(float), **kw).fit()
    f2 = PHReg(time, x, (status == 2).astype(float), **kw).fit()
    return f1, f2


# ----------------------------------------------------------------------
# Basic fitted-mode behaviour
# ----------------------------------------------------------------------

@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_fitted_closure_monotonicity_and_structure(ties):
    rng = np.random.default_rng(42)
    time, x, status = _competing_data(rng)
    f1, f2 = _two_fits(time, x, status, ties=ties)
    ci = CumIncidenceRight.from_phreg({"A": f1, "B": f2})
    qtimes = np.array([0., 1., 1., 2., 4.])
    q = ci.predict(0.0, qtimes, exog=x[:5],
                   strata=np.zeros(5, dtype=int))
    assert list(q.columns) == ["A", "B", FF]
    assert q.index.names == ["row", "time"]
    assert q.loc[0].index.tolist() == [0., 1., 1., 2., 4.]
    assert_allclose(q.sum(axis=1), 1.0, atol=TOL)
    for r in range(5):
        sub = q.loc[r]
        for c in ("A", "B"):
            assert np.all(np.diff(sub[c].values) >= -TOL)
        assert np.all(np.diff(sub[FF].values) <= TOL)


def test_fitted_origin_equal_time_is_no_failure():
    rng = np.random.default_rng(42)
    time, x, status = _competing_data(rng)
    f1, f2 = _two_fits(time, x, status)
    ci = CumIncidenceRight.from_phreg({"A": f1, "B": f2})
    q = ci.predict(2.0, np.array([2.0]), exog=x[:2],
                   strata=np.zeros(2, int))
    assert_allclose(q[["A", "B"]].values, 0.0)
    assert_allclose(q[FF].values, 1.0)


def test_fitted_interval_splicing():
    rng = np.random.default_rng(42)
    time, x, status = _competing_data(rng)
    f1, f2 = _two_fits(time, x, status)
    ci = CumIncidenceRight.from_phreg({"A": f1, "B": f2})
    o, m, t = 0.5, 2.0, 4.0
    full = ci.predict(o, np.array([t]), exog=x[:4],
                      strata=np.zeros(4, int))
    head = ci.predict(o, np.array([m]), exog=x[:4],
                      strata=np.zeros(4, int))
    tail = ci.predict(m, np.array([t]), exog=x[:4],
                      strata=np.zeros(4, int))
    for r in range(4):
        for c in ("A", "B"):
            lhs = full.loc[r, c].iloc[0]
            rhs = head.loc[r, c].iloc[0] + \
                head.loc[r, FF].iloc[0] * tail.loc[r, c].iloc[0]
            assert_allclose(lhs, rhs, atol=TOL)
        assert_allclose(
            full.loc[r, FF].iloc[0],
            head.loc[r, FF].iloc[0] * tail.loc[r, FF].iloc[0],
            atol=TOL)


def test_fitted_order_and_repeat_invariance():
    rng = np.random.default_rng(42)
    time, x, status = _competing_data(rng)
    f1, f2 = _two_fits(time, x, status)
    ci = CumIncidenceRight.from_phreg({"A": f1, "B": f2})
    qtimes = np.array([0., 1., 1., 2., 4.])
    q = ci.predict(0.0, qtimes, exog=x[:5],
                   strata=np.zeros(5, int))
    # Insertion order of the fits mapping.
    ci_swap = CumIncidenceRight.from_phreg({"B": f2, "A": f1})
    q_swap = ci_swap.predict(0.0, qtimes, exog=x[:5],
                             strata=np.zeros(5, int))
    assert_allclose(q["A"].values, q_swap["A"].values)
    assert_allclose(q["B"].values, q_swap["B"].values)
    # Swapping two independent prediction rows.
    ex3, ex3b = x[:3].copy(), x[:3][[1, 0, 2]]
    q3 = ci.predict(0.0, np.array([1., 2., 3.]), exog=ex3,
                    strata=np.zeros(3, int))
    q3b = ci.predict(0.0, np.array([1., 2., 3.]), exog=ex3b,
                     strata=np.zeros(3, int))
    assert_allclose(q3.loc[0].values, q3b.loc[1].values)
    assert_allclose(q3.loc[1].values, q3b.loc[0].values)
    # Repeated queries are deterministic and inputs are reusable.
    q_re = ci.predict(0.0, qtimes, exog=x[:5],
                      strata=np.zeros(5, int))
    pd.testing.assert_frame_equal(q, q_re)


def test_fitted_label_rename_follows_probabilities():
    rng = np.random.default_rng(3)
    n = 40
    tt = np.array([3.] * 10 + [5.] * 10 + [8.] * 20, dtype=float)
    xx = rng.normal(size=(n, 1))
    sa = np.zeros(n)
    sa[:10] = 1
    sb = np.zeros(n)
    sb[20:25] = 1
    g1 = PHReg(tt, xx, sa).fit()
    g2 = PHReg(tt, xx, sb).fit()
    ci = CumIncidenceRight.from_phreg({"a": g1, "b": g2})
    ci_r = CumIncidenceRight.from_phreg({"motor": g2, "sensor": g1})
    xq = rng.normal(size=(1, 1))
    q = ci.predict(0.0, np.array([8.]), exog=xq,
                   strata=np.zeros(1, int))
    qr = ci_r.predict(0.0, np.array([8.]), exog=xq,
                      strata=np.zeros(1, int))
    assert_allclose(qr["motor"].iloc[0], q["b"].iloc[0])
    assert_allclose(qr["sensor"].iloc[0], q["a"].iloc[0])


# ----------------------------------------------------------------------
# Single-outcome degeneration
# ----------------------------------------------------------------------

@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_single_outcome_matches_phreg_survival(ties):
    rng = np.random.default_rng(7)
    n = 80
    x = rng.normal(size=(n, 2))
    time = rng.uniform(0.5, 8, n)
    status = rng.integers(0, 2, n).astype(float)
    fit = PHReg(time, x, status, ties=ties).fit()
    ci = CumIncidenceRight.from_phreg({"only": fit})
    o = 1.5
    qtimes = np.array([1.5, 2.5, 4.0, 7.0])
    q = ci.predict(o, qtimes, exog=x[:4],
                   strata=np.zeros(4, int))
    for r in range(4):
        st = fit.predict(endog=qtimes, exog=np.tile(x[r], (4, 1)),
                         pred_type="surv",
                         post_event=True).predicted_values
        so = fit.predict(endog=np.array([o]), exog=x[r:r + 1],
                         pred_type="surv",
                         post_event=True).predicted_values[0]
        assert_allclose(q.loc[r][FF].values, st / so, atol=TOL)
        assert_allclose(q.loc[r]["only"].values, 1 - st / so,
                        atol=TOL)
        assert_allclose(q.loc[r].sum(axis=1), 1.0)


# ----------------------------------------------------------------------
# Same-day tie apportionment
# ----------------------------------------------------------------------

def test_tie_split_proportional_to_scaled_jumps():
    rng = np.random.default_rng(3)
    tt = np.array([3., 3., 3., 3., 5., 6.])
    ss = np.array([1, 1, 2, 2, 0, 0])
    xx = rng.normal(size=(6, 2))
    f1 = PHReg(tt, xx, (ss == 1).astype(float)).fit()
    f2 = PHReg(tt, xx, (ss == 2).astype(float)).fit()
    ci = CumIncidenceRight.from_phreg({"a": f1, "b": f2})
    xq = rng.normal(size=(1, 2))
    q = ci.predict(0.0, np.array([3.]), exog=xq,
                   strata=np.zeros(1, int))
    w = np.exp(ci._params @ xq[0])
    scaled_jumps = ci._jumps[0][:, 0] * w
    assert_allclose(q["a"].iloc[0] / q["b"].iloc[0],
                    scaled_jumps[0] / scaled_jumps[1], rtol=1e-10)
    s3 = np.exp(-scaled_jumps.sum())
    assert_allclose(q[FF].iloc[0], s3)
    assert_allclose(q["a"].iloc[0] + q["b"].iloc[0], 1 - s3)


def test_zero_jump_cause_gets_no_mass():
    rng = np.random.default_rng(3)
    n = 40
    tt = np.array([3.] * 10 + [5.] * 10 + [8.] * 20, dtype=float)
    xx = rng.normal(size=(n, 1))
    sa = np.zeros(n)
    sa[:10] = 1
    sb = np.zeros(n)
    sb[20:25] = 1
    g1 = PHReg(tt, xx, sa).fit()
    g2 = PHReg(tt, xx, sb).fit()
    ci = CumIncidenceRight.from_phreg({"a": g1, "b": g2})
    xq = rng.normal(size=(1, 1))
    q = ci.predict(0.0, np.array([3., 5.]), exog=xq,
                   strata=np.zeros(1, int))
    assert_allclose(q["b"].values, 0.0)
    assert np.all(q["a"].values > 0)
    assert_allclose(q.sum(axis=1), 1.0)


# ----------------------------------------------------------------------
# Strata, missing cause in one stratum, delayed entry, offset
# ----------------------------------------------------------------------

def _stratified_data():
    rng = np.random.default_rng(11)
    n = 160
    x = rng.normal(size=(n, 2))
    strata = np.where(rng.uniform(size=n) < 0.5, "L", "H")
    entry = rng.uniform(0, 1.0, n)
    u1 = -np.log(rng.uniform(size=n)) / \
        np.exp(x @ np.array([0.4, 0.1]))
    u2 = -np.log(rng.uniform(size=n)) / \
        np.exp(x @ np.array([-0.3, 0.5])) * 1.5
    u2 = np.where(strata == "H", np.inf, u2)  # no cause-2 events in H
    tc = rng.uniform(2.0, 8.0, n)
    raw = np.column_stack([u1, u2, tc])
    time = np.maximum(np.min(raw, axis=1), entry + 0.05)
    which = np.argmin(raw, axis=1)
    status = np.where(which == 2, 0, which + 1)
    return time, x, strata, entry, status


def test_stratified_missing_cause_entry_and_offset():
    time, x, strata, entry, status = _stratified_data()
    f1 = PHReg(time, x, (status == 1).astype(float),
               strata=strata, entry=entry).fit()
    f2 = PHReg(time, x, (status == 2).astype(float),
               strata=strata, entry=entry).fit()
    ci = CumIncidenceRight.from_phreg({"c1": f1, "c2": f2})
    # Stratum H has no cause-2 events: aligned hazard must be zero.
    assert ci._cumhaz["H"][1].max() == 0.0
    assert ci._cumhaz["H"][0].max() > 0.0
    ex = x[:6].copy()
    st = np.array(["L", "H", "L", "H", "L", "H"])
    q = ci.predict(1.0, np.array([1.0, 3.0, 5.0, 8.0]),
                   exog=ex, strata=st)
    assert_allclose(q.sum(axis=1), 1.0, atol=TOL)
    h_rows = q.loc[np.flatnonzero(st == "H")]
    assert_allclose(h_rows["c2"].values, 0.0)
    for r in range(6):
        sub = q.loc[r]
        assert np.all(np.diff(sub["c1"].values) >= -TOL)
        assert np.all(np.diff(sub[FF].values) <= TOL)
    # Prediction offset: internal recomputation must match the output.
    off = np.array([0.5, -0.5, 0.2, 0.0, 1.0, -0.2])
    qo = ci.predict(1.0, np.array([3.0, 5.0]), exog=ex,
                    strata=st, offset=off)
    qd = ci.predict(1.0, np.array([3.0, 5.0]), exog=ex, strata=st,
                    offset=np.zeros(6))
    for r in range(6):
        grid = ci._grid[st[r]]
        w = np.exp(ci._params @ ex[r] + off[r])
        total_jumps = (ci._jumps[st[r]] * w[:, None]).sum(0)
        surv = np.exp(-np.cumsum(total_jumps))

        def lk(qv):
            ix = np.searchsorted(grid, qv, "right") - 1
            return 1.0 if ix < 0 else surv[ix]

        expect = np.array([lk(3.0), lk(5.0)]) / lk(1.0)
        assert_allclose(qo.loc[r][FF].values, expect, atol=TOL)
    # Zero offset equals the default; a nonzero offset changes results.
    pd.testing.assert_frame_equal(
        qd, ci.predict(1.0, np.array([3.0, 5.0]), exog=ex, strata=st))
    assert not np.allclose(qo.loc[0][FF].values,
                           qd.loc[0][FF].values)


def test_unknown_stratum_and_bad_shapes():
    time, x, strata, entry, status = _stratified_data()
    f1 = PHReg(time, x, (status == 1).astype(float),
               strata=strata, entry=entry).fit()
    f2 = PHReg(time, x, (status == 2).astype(float),
               strata=strata, entry=entry).fit()
    ci = CumIncidenceRight.from_phreg({"c1": f1, "c2": f2})
    with pytest.raises(ValueError):
        ci.predict(1.0, np.array([2.]), exog=x[:1],
                   strata=np.array(["Z"]))
    with pytest.raises(ValueError):
        ci.predict(1.0, np.array([2.]), exog=x[:2, :1],  # wrong width
                   strata=np.array(["L", "H"]))
    with pytest.raises(ValueError):
        ci.predict(1.0, np.array([2.]), exog=x[:2],
                   strata=np.array(["L"]))  # length mismatch
    with pytest.raises(ValueError):
        ci.predict(1.0, np.array([2.]), exog=x[:2],
                   strata=np.array(["L", "H"]),
                   offset=np.array([0.1]))  # offset length mismatch
    with pytest.raises(ValueError):
        ci.predict(1.0, np.array([2.]))  # missing exog/strata


# ----------------------------------------------------------------------
# Cohort compatibility
# ----------------------------------------------------------------------

def test_cohort_conflicts_are_rejected():
    rng = np.random.default_rng(5)
    n = 60
    x = rng.normal(size=(n, 2))
    time = rng.uniform(1, 8, n)
    s1 = rng.integers(0, 2, n).astype(float)
    s2 = np.where(s1 == 1, 0.0,
                  rng.integers(0, 2, n).astype(float))
    f1 = PHReg(time, x, s1).fit()

    def check(**kw):
        g = PHReg(kw.get("time", time), kw.get("exog", x),
                  kw.get("status", s2), entry=kw.get("entry"),
                  strata=kw.get("strata"),
                  offset=kw.get("offset")).fit()
        with pytest.raises(ValueError):
            CumIncidenceRight.from_phreg({"a": f1, "b": g})

    check(time=time + 0.01)
    check(exog=x + 1.0)
    check(time=time[:-1], exog=x[:-1], status=s2[:-1])
    check(strata=rng.integers(0, 2, n))
    check(offset=rng.normal(size=n))
    check(entry=rng.uniform(0, 0.5, n))
    # A row that is an event in both models.
    check(status=s1.copy())


def test_cohort_uses_row_position_not_index():
    rng = np.random.default_rng(5)
    n = 60
    x = rng.normal(size=(n, 2))
    time = rng.uniform(1, 8, n)
    s1 = rng.integers(0, 2, n).astype(float)
    s2 = np.where(s1 == 1, 0.0,
                  rng.integers(0, 2, n).astype(float))
    df = pd.DataFrame({"t": time, "s1": s1, "s2": s2,
                       "x0": x[:, 0], "x1": x[:, 1]})
    shuffled = df.sample(frac=1.0, random_state=0)
    fa = PHReg(df["t"].values, df[["x0", "x1"]].values,
               df["s1"].values).fit()
    fb = PHReg(shuffled["t"].values,
               shuffled[["x0", "x1"]].values,
               shuffled["s2"].values).fit()
    with pytest.raises(ValueError):
        CumIncidenceRight.from_phreg({"a": fa, "b": fb})


def test_regularized_and_wrong_type_rejected():
    rng = np.random.default_rng(8)
    n = 50
    x = rng.normal(size=(n, 2))
    time = rng.uniform(1, 8, n)
    s1 = rng.integers(0, 2, n).astype(float)
    s2 = np.where(s1 == 1, 0.0,
                  rng.integers(0, 2, n).astype(float))
    f2 = PHReg(time, x, s2).fit()
    reg = PHReg(time, x, s1).fit_regularized(alpha=0.05)
    with pytest.raises(ValueError):
        CumIncidenceRight.from_phreg({"a": reg, "b": f2})
    with pytest.raises(ValueError):
        CumIncidenceRight.from_phreg({"a": "not a fit"})
    for bad in ({}, ["a", "b"], "x", None):
        with pytest.raises(ValueError):
            CumIncidenceRight.from_phreg(bad)


def test_wrapped_results_accepted():
    import statsmodels.base.wrapper as wrap
    rng = np.random.default_rng(8)
    n = 50
    x = rng.normal(size=(n, 2))
    time = rng.uniform(1, 8, n)
    s1 = rng.integers(0, 2, n).astype(float)
    s2 = np.where(s1 == 1, 0.0,
                  rng.integers(0, 2, n).astype(float))
    f1 = PHReg(time, x, s1).fit()
    f2 = PHReg(time, x, s2).fit()

    class DummyWrap(wrap.ResultsWrapper):
        _attrs = {}
        _wrap_attrs = {}
        _methods = {}

    wrapped = DummyWrap(f1)
    ci = CumIncidenceRight.from_phreg({"a": wrapped, "b": f2})
    q = ci.predict(0.0, np.array([2., 4.]), exog=x[:3],
                   strata=np.zeros(3, int))
    assert_allclose(q.sum(axis=1), 1.0)
