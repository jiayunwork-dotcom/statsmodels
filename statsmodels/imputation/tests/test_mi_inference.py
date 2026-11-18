"""
Tests for finite-sample multiple-imputation inference.

These tests cover the shared pooling/inference machinery used by both
the MICE and BayesGaussMI/MI analysis chains: Rubin combining rules,
Barnard--Rubin scalar degrees of freedom, Li--Raghunathan--Rubin
joint F tests, constraint handling, validation and the two entry
points.  Independent reference values are recomputed directly with
NumPy/SciPy below, never with the internal functions under test.
"""
import numpy as np
import pytest
from scipy import stats

import statsmodels.api as sm
from statsmodels.imputation.bayes_mi import BayesGaussMI, MI
from statsmodels.imputation._mi_inference import (
    MIPooledInference,
    MI_RANK_TOL,
    _extract_fit,
    _validate_inference_kwds,
)

RTOL = 1e-7
ATOL = 1e-10


def scalar(x):
    """Collapse a 0-d/1-element array to a Python float without the
    deprecated implicit ndarray-to-scalar conversion."""
    return np.asarray(x).squeeze().item()


def make_completed(rng, m=5, n=50, differing=(12, 1)):
    """Hand-completed data sets with direction-specific missing info.

    The first non-intercept column varies strongly across imputations,
    the second only weakly, producing clearly different fractions of
    missing information in different parameter directions.
    """
    beta = np.array([0.5, 1.5, -0.8])
    out = []
    for j in range(m):
        X = rng.normal(size=(n, 3))
        X[:, 0] = 1.0
        X[2:12, 1] += differing[0] * (j - (m - 1) / 2.0)
        X[20:25, 2] += differing[1] * 0.1 * (j - (m - 1) / 2.0)
        y = X @ beta + rng.normal(size=n) * 0.8
        out.append((y, X))
    return out


def pool_from(completed, df_complete=None):
    fits = [_extract_fit(sm.OLS(y, X).fit(), j + 1)
            for j, (y, X) in enumerate(completed)]
    return MIPooledInference(fits, df_complete), fits


# ---------------------------------------------------------------------------
# Rubin pooling against an independent NumPy calculation
# ---------------------------------------------------------------------------
def test_rubin_pooling_independent():
    rng = np.random.RandomState(99)
    completed = make_completed(rng, m=5)
    pool, raw_fits = pool_from(completed)

    P = np.array([f["params"] for f in raw_fits])
    C = np.array([f["cov"] for f in raw_fits])
    m = len(raw_fits)
    qbar = P.mean(0)
    U = C.mean(0)
    B = np.cov(P.T, ddof=1)
    T = U + (1 + 1 / m) * B

    assert np.allclose(qbar, pool.params, RTOL, ATOL)
    assert np.allclose(U, pool.cov_within, RTOL, ATOL)
    assert np.allclose(B, pool.cov_between, RTOL, ATOL)
    assert np.allclose(T, pool.cov_total, RTOL, ATOL)
    fmi = (1 + 1 / m) * np.diag(B) / np.diag(T)
    assert np.allclose(fmi, pool.frac_miss_info, RTOL, ATOL)
    # Different directions carry different missing information.
    assert not np.allclose(pool.frac_miss_info,
                           pool.frac_miss_info[0])


def test_barnard_rubin_scalar_independent():
    rng = np.random.RandomState(99)
    completed = make_completed(rng, m=5)
    pool, raw_fits = pool_from(completed)
    m = pool.m
    dfcom = pool.df_complete
    qbar, U, B = pool.params, pool.cov_within, pool.cov_between

    def reference(row, q):
        u = row @ U @ row
        b = row @ B @ row
        vt = u + (1 + 1 / m) * b
        r = (1 + 1 / m) * b / u
        df_old = (m - 1) * (1 + 1 / r) ** 2
        df_obs = dfcom * (dfcom + 1) / (dfcom + 3)
        df = df_old * df_obs / (df_old + df_obs)
        t = (row @ qbar - q) / np.sqrt(vt)
        p = 2 * stats.t.sf(abs(t), df)
        ci = (row @ qbar - q) + np.array([-1, 1]) * \
            stats.t.ppf(0.975, df) * np.sqrt(vt)
        return t, p, df, ci

    rows = [np.eye(3)[i] for i in range(3)]
    rows += [np.array([0., 1., -1.]), np.array([1., 0.5, 0.])]
    qs = [0, 0, 0, 0.3, 0.0]
    for row, q in zip(rows, qs):
        t, p, df, ci = reference(row, q)
        cr = pool.t_test(row.reshape(1, -1), np.array([q]))
        assert np.isclose(cr.statistic.item(), t, RTOL, ATOL)
        assert np.isclose(cr.pvalue.item(), p, RTOL, ATOL)
        assert np.isclose(cr.df_denom.item(), df, RTOL, ATOL)
        assert np.allclose(cr.conf_int().ravel(), ci, RTOL, ATOL)
        assert np.isclose(cr.effect.item(), row @ qbar - q, RTOL, ATOL)


def test_lrr_joint_independent():
    rng = np.random.RandomState(99)
    completed = make_completed(rng, m=5)
    pool, _ = pool_from(completed)
    m, dfcom = pool.m, pool.df_complete
    qbar, U, B = pool.params, pool.cov_within, pool.cov_between

    def reference(R, qv):
        k = np.linalg.matrix_rank(R)
        Uc, Bc = R @ U @ R.T, R @ B @ R.T
        Tc = Uc + (1 + 1 / m) * Bc
        rbar = np.trace((1 + 1 / m) * Bc @ np.linalg.inv(Uc)) / k
        d = R @ qbar - qv
        wald = d @ np.linalg.solve(Tc, d)
        F = wald / k
        a = k * (m - 1)
        if a > 4:
            df_lrr = 4 + (a - 4) * (1 + (1 - 2 / a) / rbar) ** 2
        else:
            # Scalar / small-k branch: the Barnard--Rubin form.
            df_lrr = (m - 1) * (1 + 1 / rbar) ** 2
        df_obs = dfcom * (dfcom + 1) / (dfcom + 3)
        dfden = df_lrr * df_obs / (df_lrr + df_obs)
        return F, stats.f.sf(F, k, dfden), k, dfden, rbar, wald

    R = np.array([[0., 1., 0.], [0., 0., 1.]])
    F, p, k, dfden, rbar, wald = reference(R, np.zeros(2))
    cr = pool.wald_f_test(R)
    assert np.isclose(scalar(cr.statistic), F, RTOL, ATOL)
    assert np.isclose(scalar(cr.pvalue), p, RTOL, ATOL)
    assert cr.df_num == k
    assert np.isclose(cr.df_denom, dfden, RTOL, ATOL)
    assert np.isclose(cr.relative_increase_variance, rbar, RTOL, ATOL)
    assert np.isclose(cr.mi_wald, wald, RTOL, ATOL)

    # Nonzero q.
    R1 = np.array([[0., 1., 0.]])
    Fq, pq, _, _, _, _ = reference(R1, np.array([0.7]))
    crq = pool.wald_f_test((R1, np.array([0.7])))
    assert np.isclose(scalar(crq.statistic), Fq, RTOL, ATOL)
    assert np.isclose(scalar(crq.pvalue), pq, RTOL, ATOL)


def test_rank_one_equals_squared_t():
    rng = np.random.RandomState(99)
    pool, _ = pool_from(make_completed(rng, m=5))
    for row in [np.array([0., 1., 0.]), np.array([0., 1., -1.]),
                np.array([1., 0.3, -2.])]:
        t = pool.t_test(row)
        f = pool.wald_f_test(row.reshape(1, -1))
        assert np.isclose(scalar(f.statistic),
                          scalar(t.statistic) ** 2, RTOL, ATOL)
        assert np.isclose(scalar(f.pvalue), scalar(t.pvalue), RTOL, ATOL)
        assert np.isclose(f.df_denom, scalar(t.df_denom),
                          RTOL, ATOL)


# ---------------------------------------------------------------------------
# Zero between-imputation variance and infinite complete-data df
# ---------------------------------------------------------------------------
def test_zero_between_variance_no_nan():
    rng = np.random.RandomState(7)
    completed = make_completed(rng, m=4)
    _, fits = pool_from(completed)
    repeated = [dict(fits[0]) for _ in range(3)]
    pool = MIPooledInference(repeated, None)
    # np.cov on identical rows leaves only floating-point noise.
    assert np.abs(pool.cov_between).max() < 1e-30
    t = pool.t_test(np.eye(3))
    assert np.all(np.isfinite(t.pvalue))
    assert np.allclose(t.df_denom, pool.df_complete)
    f = pool.wald_f_test(np.array([[0, 1, 0], [0, 0, 1]]))
    assert np.isfinite(scalar(f.pvalue))
    assert f.df_denom == pool.df_complete


def test_infinite_df_complete_limits():
    rng = np.random.RandomState(7)
    pool, _ = pool_from(make_completed(rng, m=4), df_complete=np.inf)
    t = pool.t_test(np.array([0., 1., 0.]))
    assert np.isfinite(t.df_denom)  # still finite because B > 0
    # Zero B + infinite dfcom -> normal / chi-square limit.
    _, fits = pool_from(make_completed(rng, m=4))
    pool0 = MIPooledInference([dict(fits[0]) for _ in range(3)], np.inf)
    assert np.isinf(pool0.t_test(np.eye(3)).df_denom).all()
    f = pool0.wald_f_test(np.array([[0, 1, 0], [0, 0, 1]]))
    assert np.isinf(f.df_denom)
    assert np.isclose(scalar(f.pvalue),
                      stats.chi2.sf(f.mi_wald, 2), RTOL, ATOL)


# ---------------------------------------------------------------------------
# Constraint validation and equivalence properties
# ---------------------------------------------------------------------------
def test_constraint_equivalence_under_transforms():
    rng = np.random.RandomState(5)
    pool, _ = pool_from(make_completed(rng, m=4))
    R = np.array([[0., 1., 0.], [0., 0., 1.]])
    base = pool.wald_f_test(R)
    A = np.array([[2.0, 0.5], [0.0, 3.0]])  # invertible
    transformed = pool.wald_f_test(A @ R)
    reordered = pool.wald_f_test(R[::-1])
    scaled = pool.wald_f_test(R * np.array([2., -1.])[:, None])
    for other in (transformed, reordered, scaled):
        assert np.isclose(scalar(other.statistic), scalar(base.statistic),
                          RTOL, ATOL)
        assert np.isclose(other.df_denom, base.df_denom, RTOL, ATOL)
        assert np.isclose(scalar(other.pvalue), scalar(base.pvalue),
                          RTOL, ATOL)


def test_redundant_compatible_uses_effective_rank():
    rng = np.random.RandomState(5)
    pool, _ = pool_from(make_completed(rng, m=4))
    R = np.array([[0., 1., 0.], [0., 2., 0.]])
    f = pool.wald_f_test(R)
    assert f.df_num == 1
    one = pool.wald_f_test(np.array([[0., 1., 0.]]))
    assert np.isclose(scalar(f.statistic), scalar(one.statistic), RTOL, ATOL)


def test_contradictory_zero_rank_and_width_errors():
    rng = np.random.RandomState(5)
    pool, _ = pool_from(make_completed(rng, m=4))
    R = np.array([[0., 1., 0.], [0., 1., 0.]])
    with pytest.raises(ValueError, match="contradictory"):
        pool.wald_f_test((R, np.array([0.0, 1.0])))
    with pytest.raises(ValueError, match="zero-rank"):
        pool.wald_f_test(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="width"):
        pool.wald_f_test(np.eye(4))


def test_rank_tol_public_and_scale_dependence():
    assert MI_RANK_TOL == 1e-10
    rng = np.random.RandomState(5)
    pool, _ = pool_from(make_completed(rng, m=4))
    R = np.array([[0., 1., 0.], [0., 0., 1.], [0., 1., 1e-12]])
    assert pool.wald_f_test(R, rank_tol=1e-13).df_num == 2
    assert pool.wald_f_test(R, rank_tol=1e-9).df_num == 2
    # The pool's default tolerance is restored after a scoped override.
    assert pool.rank_tol == MI_RANK_TOL


# ---------------------------------------------------------------------------
# Name alignment and degrees of freedom validation
# ---------------------------------------------------------------------------
def test_parameter_name_reordering_aligns():
    rng = np.random.RandomState(5)
    _, fits = pool_from(make_completed(rng, m=4))
    reordered = []
    for j, f in enumerate(fits):
        order = [2, 0, 1] if j % 2 else [0, 1, 2]
        reordered.append({
            "names": [f["names"][i] for i in order],
            "params": f["params"][order],
            "cov": f["cov"][np.ix_(order, order)],
            "df_resid": f["df_resid"],
            "model_class": f["model_class"],
        })
    a = MIPooledInference(fits, None)
    b = MIPooledInference(reordered, None)
    assert a.param_names == b.param_names
    assert np.allclose(a.params, b.params, RTOL, ATOL)
    assert np.allclose(a.cov_total, b.cov_total, RTOL, ATOL)


def test_name_identity_and_duplicate_errors():
    rng = np.random.RandomState(5)
    _, fits = pool_from(make_completed(rng, m=4))
    bad = dict(fits[1])
    bad["names"] = fits[1]["names"][:2] + ["zzz"]
    with pytest.raises(ValueError, match="imputation 2"):
        MIPooledInference([fits[0], bad], None)
    dup = dict(fits[0])
    dup["names"] = ["a", "a", "b"]
    with pytest.raises(ValueError, match="duplicated"):
        MIPooledInference([dup, fits[1]], None)


def test_residual_df_consistency_and_override():
    rng = np.random.RandomState(5)
    _, fits = pool_from(make_completed(rng, m=4))
    d2 = dict(fits[1])
    d2["df_resid"] = fits[1]["df_resid"] - 5
    with pytest.raises(ValueError, match="differ across"):
        MIPooledInference([fits[0], d2], None)
    pool = MIPooledInference([fits[0], d2], 33.0)
    assert pool.df_complete == 33.0


@pytest.mark.parametrize("bad", [-1.0, 0.0, np.nan, "x", True])
def test_invalid_df_complete(bad):
    rng = np.random.RandomState(5)
    _, fits = pool_from(make_completed(rng, m=4))
    with pytest.raises((ValueError, TypeError)):
        MIPooledInference(fits, bad)


def test_inference_kwd_validation_before_running():
    with pytest.raises(ValueError, match="inference"):
        _validate_inference_kwds("bogus", None, 5)
    with pytest.raises(ValueError, match="at least two"):
        _validate_inference_kwds("small_sample", None, 1)
    with pytest.raises(ValueError, match="bool"):
        _validate_inference_kwds("small_sample", True, 5)
    with pytest.raises(ValueError, match="positive"):
        _validate_inference_kwds("small_sample", -2.0, 5)
    # Legacy mode does not constrain the count.
    assert _validate_inference_kwds("legacy", None, 1) is None


def test_two_imputations_supported():
    rng = np.random.RandomState(5)
    pool, _ = pool_from(make_completed(rng, m=2))
    assert pool.m == 2
    t = pool.t_test(np.eye(3))
    f = pool.wald_f_test(np.array([[0, 1, 0], [0, 0, 1]]))
    assert np.all(np.isfinite(t.pvalue)) and np.isfinite(scalar(f.pvalue))


def test_parameter_rescaling_invariance():
    rng = np.random.RandomState(5)
    completed = make_completed(rng, m=4)
    fits_a = [_extract_fit(sm.OLS(y, X).fit(), j + 1)
              for j, (y, X) in enumerate(completed)]
    c = 10.0
    fits_b = []
    for j, (y, X) in enumerate(completed):
        Xs = X.copy()
        Xs[:, 2] = Xs[:, 2] * c
        fits_b.append(_extract_fit(sm.OLS(y, Xs).fit(), j + 1))
    a = MIPooledInference(fits_a, None)
    b = MIPooledInference(fits_b, None)
    ta, tb = (a.t_test(np.array([0., 0., 1.])),
              b.t_test(np.array([0., 0., 1.])))
    assert np.isclose(scalar(ta.statistic), scalar(tb.statistic), RTOL, ATOL)
    assert np.isclose(scalar(ta.pvalue), scalar(tb.pvalue), RTOL, ATOL)
    assert np.isclose(ta.df_denom, tb.df_denom, RTOL, ATOL)
    assert np.isclose(a.params[2], c * b.params[2], RTOL, ATOL)


def test_singular_within_covariance_raises():
    rng = np.random.RandomState(5)
    pool, _ = pool_from(make_completed(rng, m=4))
    saved = pool.cov_within.copy()
    pool.cov_within = np.zeros_like(saved)
    with pytest.raises(ValueError, match="singular"):
        pool.wald_f_test(np.eye(3))
    pool.cov_within = saved


# ---------------------------------------------------------------------------
# Result-object level interface (the mixin)
# ---------------------------------------------------------------------------
def _mi_result(nrep=5, seed=0, inference="small_sample", **kwds):
    rng = np.random.RandomState(seed)
    x = rng.normal(size=(160, 4))
    x[[1, 3, 9], 0] = np.nan
    x[[1, 4, 3], 1] = np.nan
    x[[2, 11, 21], 2] = np.nan
    x[[11, 22, 99], 3] = np.nan

    def args_fn(d):
        return d[:, 0], sm.add_constant(d[:, 1:])

    mi = MI(BayesGaussMI(x.copy()), sm.OLS, args_fn,
            burn=10, nrep=nrep, skip=3)
    return mi.fit(inference=inference, **kwds)


def test_mixin_rejects_unsupported_overrides():
    r = _mi_result()
    k = len(r.params)
    R = np.eye(k)[:2]
    with pytest.raises(ValueError, match="use_t"):
        r.t_test(np.eye(k), use_t=False)
    with pytest.raises(ValueError, match="cov_p"):
        r.t_test(np.eye(k), cov_p=np.eye(k))
    with pytest.raises(ValueError, match="invcov"):
        r.f_test(R, invcov=np.eye(2))
    with pytest.raises(ValueError, match="degrees of freedom"):
        r.wald_test(R, df_constraints=2)
    with pytest.raises(ValueError, match="chi-square"):
        r.wald_test(R, use_f=False)


def test_mixin_scalar_entrypoints_aligned():
    r = _mi_result()
    k = len(r.params)
    tt = r.t_test(np.eye(k))
    assert np.allclose(r.bse, tt.sd, RTOL, ATOL)
    assert np.allclose(np.atleast_1d(r.pvalues),
                       np.atleast_1d(tt.pvalue), RTOL, ATOL)
    assert np.allclose(r.tvalues, tt.statistic, RTOL, ATOL)
    for alpha in (0.01, 0.05, 0.1):
        assert np.allclose(r.conf_int(alpha), tt.conf_int(alpha),
                           RTOL, ATOL)
    assert "df_denom" in tt.summary_frame().columns
    with pytest.raises(ValueError):
        r.conf_int(0.0)
    with pytest.raises(ValueError):
        r.t_test(np.eye(k), alpha=1.0)


def test_constraint_input_forms():
    r = _mi_result()
    R = np.array([0., 1., 0., 0.])
    numeric = r.t_test(R)
    string = r.t_test("x1 = 0")
    tuple_form = r.t_test((R.reshape(1, -1), np.array([0.0])))
    assert np.allclose(numeric.statistic, string.statistic, RTOL, ATOL)
    assert np.allclose(numeric.statistic, tuple_form.statistic, RTOL, ATOL)
    # String with nonzero q and named arithmetic.
    s2 = r.t_test("x1 / 2 = 0.25")
    t2 = r.t_test((np.array([[0., 0.5, 0., 0.]]), np.array([0.25])))
    assert np.isclose(scalar(s2.statistic), scalar(t2.statistic),
                      RTOL, ATOL)


def test_diagnostics_expose_method_and_reference():
    r = _mi_result()
    t = r.t_test(np.eye(len(r.params)))
    assert "Barnard" in t._mi_diagnostics["method"]
    assert "Barnard" in t._mi_diagnostics["references"]
    f = r.f_test(np.eye(len(r.params))[:2])
    assert "LRR" in f._mi_diagnostics["method"]
    assert "Li" in f._mi_diagnostics["references"]
    assert r.mi_inference.references["pooling"].startswith("Rubin")
