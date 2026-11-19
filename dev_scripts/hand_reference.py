"""
Independent hand-computed reference checks for CumIncidenceRight.

All reference numbers here are derived by hand from Nelson-Aalen /
Kaplan-Meier definitions on tiny deterministic samples, not by the
library code under test.
"""
import numpy as np
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import (
    CumIncidenceRight, FAILURE_FREE_LABEL)

FF = FAILURE_FREE_LABEL


def test_empirical_hand_values():
    # 6 subjects: events of cause 1 at t=1 (x2), cause 2 at t=2 (x1),
    # censor at t=2 (x1), censor at t=3 (x2).
    time = np.array([1., 1., 2., 2., 3., 3.])
    status = np.array([1, 1, 2, 0, 0, 0])
    ci = CumIncidenceRight(time, status)

    # Hand KM: n(t=1)=6, d=2 -> S(1)=4/6=2/3
    #         n(t=2)=4, d=1 -> S(2)=2/3 * 3/4 = 1/2
    assert np.isclose(ci._emp_surv[0], 2. / 3)
    assert np.isclose(ci._emp_surv[1], 1. / 2)

    # Hand Aalen-Johansen:
    # I_1(1) = 1 * 2/6 = 1/3
    # I_2(2) = S(1) * 1/4 = (2/3)/4 = 1/6
    q = ci.predict(0.0, np.array([1., 2., 3.]))
    assert np.allclose(q[1].values, [1. / 3, 1. / 3, 1. / 3])
    assert np.allclose(q[2].values, [0., 1. / 6, 1. / 6])
    assert np.allclose(q[FF].values, [2. / 3, 1. / 2, 1. / 2])
    assert np.allclose(q.sum(axis=1), 1.0)

    # Conditional on surviving t=1: S(2)/S(1) = (1/2)/(2/3) = 3/4
    qc = ci.predict(1.0, np.array([1., 2., 3.]))
    assert np.isclose(qc[FF].iloc[0], 1.0)
    assert np.isclose(qc[FF].iloc[1], 0.75)
    # conditional cause-2 mass in (1,2]: (1/6) / (2/3) = 1/4
    assert np.isclose(qc[2].iloc[1], 0.25)
    assert np.isclose(qc[1].iloc[1], 0.0)
    assert np.allclose(qc.sum(axis=1), 1.0)


def test_fitted_hand_values_single_cause():
    # Exponential, single covariate column with coefficient fitted.
    # Construct data so the baseline Nelson-Aalen jump at the only
    # event time t=2 is d / sum(exp(x*beta)).  We verify the query
    # against an independently evaluated Nelson-Aalen expression.
    rng = np.random.default_rng(0)
    time = np.array([2., 2., 2., 5., 5.])
    status = np.array([1., 1., 1., 0., 0.])
    exog = rng.normal(size=(5, 1))
    fit = PHReg(time, exog, status, ties="breslow").fit()
    beta = fit.params[0]
    base = fit.model.baseline_cumulative_hazard(fit.params,
                                                 post_event=True)[0]
    jump = base[1][0]  # 3 / sum(exp(x*beta))
    assert np.isclose(jump, 3.0 / np.exp(exog @ fit.params).sum())

    ci = CumIncidenceRight.from_phreg({"c": fit})
    xq = np.array([[0.7]])
    q = ci.predict(0.0, np.array([1., 2., 5.]), exog=xq,
                   strata=np.zeros(1, int))
    h = jump * np.exp(beta * 0.7)
    assert np.isclose(q[FF].iloc[0], 1.0)           # before event
    assert np.isclose(q[FF].iloc[1], np.exp(-h))    # at event day
    assert np.isclose(q[FF].iloc[2], np.exp(-h))    # carried forward
    assert np.isclose(q["c"].iloc[1], 1 - np.exp(-h))
    assert np.allclose(q.sum(axis=1), 1.0)

    # Conditional from origin 2 (already through the event): no further
    # risk, so failure-free stays 1 and cause mass stays 0.
    q2 = ci.predict(2.0, np.array([2., 5.]), exog=xq,
                    strata=np.zeros(1, int))
    assert np.allclose(q2[FF].values, 1.0)
    assert np.allclose(q2["c"].values, 0.0)


def test_fitted_two_cause_hand_tie_split():
    # One shared event time t=4, two causes, one event each, and a
    # covariate design that gives equal fitted risk weights for the two
    # failing rows (use identical exog rows for the two events).
    rng = np.random.default_rng(1)
    time = np.array([4., 4., 6., 6., 7., 7., 8., 8., 9., 9., 10., 10.])
    status1 = np.array([1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0.])
    status2 = np.array([0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1.])
    exog = rng.normal(size=(12, 2))
    exog[1] = exog[0]  # equal covariate for the tied events at t=4
    f1 = PHReg(time, exog, status1).fit()
    f2 = PHReg(time, exog, status2).fit()
    ci = CumIncidenceRight.from_phreg({"a": f1, "b": f2})
    xq = np.zeros((1, 2))  # unit hazard ratio for both
    q = ci.predict(0.0, np.array([4., 8.]), exog=xq,
                   strata=np.zeros(1, int))
    j1 = f1.model.baseline_cumulative_hazard(f1.params, True)[0][1][0]
    j2 = f2.model.baseline_cumulative_hazard(f2.params, True)[0][1][0]
    # total event mass at t=4 is 1 - exp(-(j1+j2)), split in proportion
    # to the two (unit-scaled, xq=0) baseline jumps.
    mass = 1 - np.exp(-(j1 + j2))
    assert np.isclose(q["a"].iloc[0], mass * j1 / (j1 + j2))
    assert np.isclose(q["b"].iloc[0], mass * j2 / (j1 + j2))
    assert np.isclose(q[FF].iloc[0], np.exp(-(j1 + j2)))
    assert np.allclose(q.sum(axis=1), 1.0)


def test_interval_splice_identity():
    # P(fail by t2) = P(fail by t1) + S(t1|o) * P(fail in (t1,t2]|t1)
    rng = np.random.default_rng(12)
    n = 80
    x = rng.normal(size=(n, 2))
    t1 = -np.log(rng.uniform(size=n)) / np.exp(x @ [0.3, -0.2])
    t2 = -np.log(rng.uniform(size=n)) / np.exp(x @ [-0.4, 0.1]) * 2
    tc = rng.uniform(1, 7, n)
    time = np.minimum(np.minimum(t1, t2), tc)
    status = np.where(t1 <= np.minimum(t2, tc), 1,
                      np.where(t2 <= tc, 2, 0))
    f1 = PHReg(time, x, (status == 1).astype(float)).fit()
    f2 = PHReg(time, x, (status == 2).astype(float)).fit()
    ci = CumIncidenceRight.from_phreg({"a": f1, "b": f2})
    o, m, t = 0.8, 2.3, 5.0
    full = ci.predict(o, np.array([t]), exog=x[:4],
                      strata=np.zeros(4, int))
    head = ci.predict(o, np.array([m]), exog=x[:4],
                      strata=np.zeros(4, int))
    tail = ci.predict(m, np.array([t]), exog=x[:4],
                      strata=np.zeros(4, int))
    for r in range(4):
        for c in ("a", "b"):
            lhs = full.loc[r, c].iloc[0]
            rhs = head.loc[r, c].iloc[0] + \
                head.loc[r, FF].iloc[0] * tail.loc[r, c].iloc[0]
            assert np.isclose(lhs, rhs, atol=1e-12)
        assert np.isclose(
            full.loc[r, FF].iloc[0],
            head.loc[r, FF].iloc[0] * tail.loc[r, FF].iloc[0],
            atol=1e-12)


if __name__ == "__main__":
    test_empirical_hand_values()
    test_fitted_hand_values_single_cause()
    test_fitted_two_cause_hand_tie_split()
    test_interval_splice_identity()
    print("all hand-reference checks passed")
