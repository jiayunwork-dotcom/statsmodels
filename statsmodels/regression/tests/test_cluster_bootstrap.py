"""
Tests for cluster bootstrap inference (pairs cluster bootstrap) for OLS.

The reference values in this file are recomputed with an independent
implementation of the documented algorithm (per-resample SeedSequence
children, bincount multiplicities, Cholesky refit, one-pass variance,
order-statistic critical values, (1 + #) / (B + 1) tail probabilities).
No internal sampling or interval functions of the new implementation are
used to generate the reference values.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import linalg

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression._cluster_bootstrap import (
    MAX_ATTEMPTS,
    ClusterBootstrapCalibration,
)
import statsmodels.regression._cluster_bootstrap as cb_module


# ----------------------------------------------------------------------
# independent reference implementation of the documented algorithm
# ----------------------------------------------------------------------
def reference_boot_params(x, y, cluster_id, n_clusters, n_resamples,
                          random_state, max_attempts=10):
    """Resample whole clusters and refit OLS, following the documented
    algorithm: one SeedSequence child per resample, G uniform cluster
    indices per draw, bincount multiplicities, Cholesky refit."""
    seed_seq = np.random.SeedSequence(random_state)
    children = seed_seq.spawn(n_resamples)
    out = []
    for b in range(n_resamples):
        gen = np.random.default_rng(children[b])
        for _ in range(max_attempts):
            idx = gen.integers(0, n_clusters, size=n_clusters)
            counts = np.bincount(idx, minlength=n_clusters)
            w = counts[cluster_id].astype(np.float64)
            xw = x * w[:, None]
            try:
                cho = linalg.cho_factor(x.T @ xw)
                beta = linalg.cho_solve(cho, xw.T @ y)
            except np.linalg.LinAlgError:
                continue
            if np.all(np.isfinite(beta)):
                out.append(beta)
                break
    return np.array(out)


def reference_linear_dev(r_matrix, boot_centered):
    """D[i, b] = sum_k R[i, k] * Bc[b, k], accumulated in parameter order."""
    out = np.zeros((r_matrix.shape[0], boot_centered.shape[0]))
    for k in range(r_matrix.shape[1]):
        out += r_matrix[:, k:k + 1] * boot_centered[:, k:k + 1].T
    return out


def reference_critical_rank(alpha, n_boot):
    k = int(np.ceil((n_boot + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n_boot)
    return k - 1


def reference_cluster_ids(labels):
    """Cluster ids in order of first appearance (documented rule)."""
    _, first_pos, inverse = np.unique(labels, return_index=True,
                                      return_inverse=True)
    rank = np.empty(len(first_pos), dtype=np.intp)
    rank[np.argsort(first_pos)] = np.arange(len(first_pos))
    return rank[inverse]


def reference_inference(boot_params, params, r_matrix, q, alpha,
                        simultaneous):
    """Documented one-pass variance, order-statistic critical values and
    (1 + #) / (B + 1) p-values."""
    n_boot = boot_params.shape[0]
    boot_centered = boot_params - params
    dev = reference_linear_dev(r_matrix, boot_centered)
    effect = r_matrix.dot(params)
    sum_d = dev.sum(axis=1)
    sumsq_d = (dev * dev).sum(axis=1)
    var = (sumsq_d - sum_d * sum_d / n_boot) / (n_boot - 1.0)
    sd = np.sqrt(var)
    t_abs = np.abs(dev) / sd[:, None]
    t_obs = (effect - q) / sd
    abs_t_obs = np.abs(t_obs)
    rank = reference_critical_rank(alpha, n_boot)
    if simultaneous:
        max_t = t_abs.max(axis=0)
        crit = np.full(r_matrix.shape[0], np.partition(max_t, rank)[rank])
        counts = (max_t[None, :] >= abs_t_obs[:, None]).sum(axis=1)
    else:
        crit = np.partition(t_abs, rank, axis=1)[:, rank]
        counts = (t_abs >= abs_t_obs[:, None]).sum(axis=1)
    pvalue = (1.0 + counts) / (n_boot + 1.0)
    ci = np.column_stack((effect - crit * sd, effect + crit * sd))
    return effect, sd, t_obs, pvalue, ci


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def unequal_cluster_data():
    """Small synthetic table with unequal cluster sizes."""
    rng = np.random.default_rng(20260401)
    sizes = [3, 5, 2, 8, 4, 6, 3, 7, 1, 5, 4, 6, 2, 9]
    rows, groups = [], []
    beta = np.array([1.0, 0.5, -0.8])
    for g, s in enumerate(sizes):
        x1 = rng.normal(size=s)
        x2 = rng.normal(size=s)
        u = rng.normal()
        y = (beta[0] + beta[1] * x1 + beta[2] * x2 + u
             + rng.normal(scale=0.5, size=s))
        rows.append(np.column_stack([np.ones(s), x1, x2, y]))
        groups += [f"school{g}"] * s
    data = np.vstack(rows)
    x, y = data[:, :3], data[:, 3]
    cluster_id = reference_cluster_ids(np.array(groups))
    return {
        "x": x, "y": y, "groups": groups, "cluster_id": cluster_id,
        "sizes": sizes, "beta": beta,
    }


@pytest.fixture(scope="module")
def fitted_unequal(unequal_cluster_data):
    d = unequal_cluster_data
    res = sm.OLS(d["y"], d["x"]).fit()
    cal = res.get_cluster_bootstrap(d["groups"], n_resamples=999,
                                    random_state=20260402, batch_size=32)
    return res, cal, d


# ----------------------------------------------------------------------
# calibration basics
# ----------------------------------------------------------------------
class TestCalibrationBasics:
    def test_independent_recomputation_boot_params(self, fitted_unequal):
        res, cal, d = fitted_unequal
        ref = reference_boot_params(
            d["x"], d["y"], d["cluster_id"], len(d["sizes"]), 999, 20260402)
        assert cal.n_valid == 999
        assert cal.n_failed == 0
        assert cal.failure_reasons == {}
        assert_array_equal_bitwise(cal.boot_params, ref)
        assert_array_equal_bitwise(
            cal.params, np.asarray(res.params, dtype=float))

    def test_independent_recomputation_contrast(self, fitted_unequal):
        res, cal, d = fitted_unequal
        r_matrix = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                             [0., 1., -1.]])
        q = np.array([0.9, 0.0, 0.0, 1.0])
        alpha = 0.1
        for simultaneous in (False, True):
            effect, sd, t_obs, pvalue, ci = reference_inference(
                cal.boot_params, cal.params, r_matrix, q, alpha,
                simultaneous)
            tt = res.t_test((r_matrix, q), inference=cal,
                            simultaneous=simultaneous)
            assert_array_equal_bitwise(tt.effect, effect)
            assert_array_equal_bitwise(tt.sd, sd)
            assert_array_equal_bitwise(tt.tvalue, t_obs)
            assert_array_equal_bitwise(np.atleast_1d(tt.pvalue), pvalue)
            assert_array_equal_bitwise(tt.conf_int(alpha), ci)
            frame = tt.summary_frame(alpha=alpha)
            assert_array_equal_bitwise(frame["coef"].to_numpy(), effect)
            assert_array_equal_bitwise(frame["std err"].to_numpy(), sd)
            assert_array_equal_bitwise(
                frame["Conf. Int. Low"].to_numpy(), ci[:, 0])
            assert_array_equal_bitwise(
                frame["Conf. Int. Upp."].to_numpy(), ci[:, 1])

    def test_independent_recomputation_prediction(self, fitted_unequal):
        res, cal, d = fitted_unequal
        rng = np.random.default_rng(7)
        a_new = np.column_stack(
            [np.ones(6), rng.normal(size=6), rng.normal(size=6)])
        alpha = 0.05
        for simultaneous in (False, True):
            effect, sd, _, _, ci = reference_inference(
                cal.boot_params, cal.params, a_new, np.zeros(6), alpha,
                simultaneous)
            pred = res.get_prediction(a_new, inference=cal)
            assert_array_equal_bitwise(pred.predicted_mean, effect)
            assert_array_equal_bitwise(pred.se_mean, sd)
            assert_array_equal_bitwise(
                pred.conf_int(alpha=alpha, simultaneous=simultaneous), ci)
            frame = pred.summary_frame(alpha=alpha,
                                       simultaneous=simultaneous)
            assert_array_equal_bitwise(frame["mean"].to_numpy(), effect)
            assert_array_equal_bitwise(frame["mean_se"].to_numpy(), sd)
            assert_array_equal_bitwise(
                frame["mean_ci_lower"].to_numpy(), ci[:, 0])
            assert_array_equal_bitwise(
                frame["mean_ci_upper"].to_numpy(), ci[:, 1])
            assert frame["obs_ci_lower"].isna().all()
            assert frame["obs_ci_upper"].isna().all()

    def test_calibration_attributes(self, fitted_unequal):
        res, cal, d = fitted_unequal
        assert cal.n_resamples == 999
        assert cal.n_valid == 999
        assert cal.n_failed == 0
        assert cal.n_clusters == len(d["sizes"])
        assert_array_equal_bitwise(
            np.sort(cal.cluster_sizes), np.sort(np.array(d["sizes"])))
        assert cal.nobs == len(d["y"])
        assert cal.df_resid == res.df_resid
        assert cal.random_state == 20260402
        assert cal.batch_size == 32
        assert cal.max_attempts == MAX_ATTEMPTS
        assert "ClusterBootstrapCalibration" in repr(cal)

    def test_same_seed_reproducible(self, unequal_cluster_data):
        d = unequal_cluster_data
        res = sm.OLS(d["y"], d["x"]).fit()
        cal1 = res.get_cluster_bootstrap(d["groups"], random_state=5)
        cal2 = res.get_cluster_bootstrap(d["groups"], random_state=5)
        assert_array_equal_bitwise(cal1.boot_params, cal2.boot_params)
        cal3 = res.get_cluster_bootstrap(d["groups"], random_state=6)
        assert not np.array_equal(cal1.boot_params, cal3.boot_params)

    def test_batch_size_does_not_change_results(self, unequal_cluster_data):
        d = unequal_cluster_data
        res = sm.OLS(d["y"], d["x"]).fit()
        cals = [res.get_cluster_bootstrap(d["groups"], n_resamples=199,
                                          random_state=9, batch_size=bs)
                for bs in (1, 2, 8, 32, 1000)]
        for cal in cals[1:]:
            assert_array_equal_bitwise(cals[0].boot_params, cal.boot_params)
        r_matrix = np.array([[1., 0., 0.], [0., 1., 0.], [0., 1., -1.]])
        a_new = d["x"][:11]
        for cal in cals[1:]:
            for simultaneous in (False, True):
                t0 = res.t_test((r_matrix, [0.1, 0., 0.]), inference=cals[0],
                                simultaneous=simultaneous)
                t1 = res.t_test((r_matrix, [0.1, 0., 0.]), inference=cal,
                                simultaneous=simultaneous)
                assert_array_equal_bitwise(t0.sd, t1.sd)
                assert_array_equal_bitwise(np.atleast_1d(t0.pvalue),
                                           np.atleast_1d(t1.pvalue))
                assert_array_equal_bitwise(t0.conf_int(0.1),
                                           t1.conf_int(0.1))
                p0 = cals[0]._infer(a_new, None, 0.1, simultaneous, False)
                p1 = cal._infer(a_new, None, 0.1, simultaneous, False)
                assert_array_equal_bitwise(p0["crit"], p1["crit"])

    def test_global_random_state_untouched(self, unequal_cluster_data):
        d = unequal_cluster_data
        res = sm.OLS(d["y"], d["x"]).fit()
        np.random.seed(12345)
        before = np.random.get_state()
        cal = res.get_cluster_bootstrap(d["groups"], n_resamples=199,
                                        random_state=3)
        res.t_test([[0., 1., 0.]], inference=cal)
        res.get_prediction(d["x"][:4], inference=cal).summary_frame()
        after = np.random.get_state()
        for b, a in zip(before, after):
            if isinstance(b, np.ndarray):
                assert_array_equal_bitwise(b, a)
            else:
                assert b == a
        np.random.seed(12345)
        expected = np.random.normal(size=3)
        np.random.seed(12345)
        _ = res.get_cluster_bootstrap(d["groups"], random_state=3)
        got = np.random.normal(size=3)
        assert_array_equal_bitwise(expected, got)

    def test_groups_copied_and_inputs_unmodified(self, unequal_cluster_data):
        d = unequal_cluster_data
        x_before = d["x"].copy()
        y_before = d["y"].copy()
        groups = list(d["groups"])
        res = sm.OLS(d["y"], d["x"]).fit()
        params_before = np.asarray(res.params).copy()
        cal = res.get_cluster_bootstrap(groups, n_resamples=199,
                                        random_state=1)
        boot_before = cal.boot_params.copy()
        # mutate the caller's containers after calibration
        groups[0] = "mutated"
        groups[-1] = "mutated"
        assert_array_equal_bitwise(cal.boot_params, boot_before)
        assert_array_equal_bitwise(d["x"], x_before)
        assert_array_equal_bitwise(d["y"], y_before)
        assert_array_equal_bitwise(np.asarray(res.params), params_before)
        # ndarray groups are copied as well
        g_arr = np.array(d["groups"])
        cal2 = res.get_cluster_bootstrap(g_arr, n_resamples=199,
                                         random_state=1)
        boot2 = cal2.boot_params.copy()
        g_arr[:] = "mutated"
        assert_array_equal_bitwise(cal2.boot_params, boot2)
        assert_array_equal_bitwise(cal.boot_params, cal2.boot_params)

    def test_repeated_queries_bitwise_identical(self, fitted_unequal):
        res, cal, d = fitted_unequal
        r_matrix = np.array([[0., 1., 0.], [0., 0., 1.], [1., 1., 1.]])
        a_new = d["x"][:7]
        # prediction first, then contrasts
        p1 = res.get_prediction(a_new, inference=cal).summary_frame(
            alpha=0.1, simultaneous=True)
        t1 = res.t_test((r_matrix, [0., 0., 1.]), inference=cal,
                        simultaneous=True)
        p1v, t1v = p1.values.copy(), np.atleast_1d(t1.pvalue).copy()
        c1v = t1.conf_int(0.1).copy()
        # contrasts first, then prediction
        t2 = res.t_test((r_matrix, [0., 0., 1.]), inference=cal,
                        simultaneous=True)
        p2 = res.get_prediction(a_new, inference=cal).summary_frame(
            alpha=0.1, simultaneous=True)
        assert_array_equal_bitwise(np.atleast_1d(t2.pvalue), t1v)
        assert_array_equal_bitwise(t2.conf_int(0.1), c1v)
        assert_array_equal_bitwise(p2.values, p1v)

    def test_unwrapped_results_work(self, unequal_cluster_data):
        d = unequal_cluster_data
        res = sm.OLS(d["y"], d["x"]).fit()
        raw = res._results  # unwrapped OLSResults
        cal = raw.get_cluster_bootstrap(d["groups"], n_resamples=199,
                                        random_state=2)
        tt = raw.t_test([[0., 1., 0.]], inference=cal)
        assert tt.pvalue is not None
        # wrapped and unwrapped share the same underlying fit
        tt2 = res.t_test([[0., 1., 0.]], inference=cal)
        assert_array_equal_bitwise(np.atleast_1d(tt.pvalue),
                                   np.atleast_1d(tt2.pvalue))
        pred = res.get_prediction(d["x"][:3], inference=cal)
        assert pred.predicted_mean is not None


def assert_array_equal_bitwise(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    assert a.shape == b.shape, f"shape mismatch {a.shape} vs {b.shape}"
    assert np.array_equal(a, b, equal_nan=True), \
        "arrays are not bitwise equal"


# ----------------------------------------------------------------------
# groups validation
# ----------------------------------------------------------------------
class TestGroupsValidation:
    def test_string_and_int_groups(self, unequal_cluster_data):
        d = unequal_cluster_data
        res = sm.OLS(d["y"], d["x"]).fit()
        # integer labels of the same contiguous partition as the strings
        g_int = [g for g, s in enumerate(d["sizes"]) for _ in range(s)]
        cal_int = res.get_cluster_bootstrap(g_int, n_resamples=99,
                                            random_state=0)
        assert cal_int.n_clusters == 14
        g_series = pd.Series(d["groups"], index=range(len(d["y"])))
        cal_ser = res.get_cluster_bootstrap(g_series, n_resamples=99,
                                            random_state=0)
        assert cal_ser.n_clusters == 14
        g_tuple = tuple(d["groups"])
        cal_tuple = res.get_cluster_bootstrap(g_tuple, n_resamples=99,
                                              random_state=0)
        assert cal_tuple.n_clusters == 14
        # integer and string labels of the same partition coincide
        assert_array_equal_bitwise(cal_int.boot_params,
                                   cal_tuple.boot_params)

    @pytest.mark.parametrize("bad_groups,exc", [
        ("not-a-container", TypeError),
        (12345, TypeError),
        ({"a": 1}, TypeError),
        (pd.DataFrame({"g": [1, 2, 3]}), TypeError),
        (np.array([[1, 2], [3, 4]]), ValueError),       # 2-D
        (np.array([True, False, True]), TypeError),     # bool
        (np.array([1.5, 2.5, 3.5]), TypeError),         # float
        (np.array([1.0, np.nan, 3.0]), ValueError),     # NaN -> missing
        ([1, None, 3], ValueError),                     # None -> missing
        (["a", "", "c"], ValueError),                   # empty string
        (["a", 1, "b"], TypeError),                     # mixed str/int
        ([1, "a", 2], TypeError),                       # mixed int/str
        ([1, True, 3], TypeError),                      # bool among ints
        (np.array([b"a", b"b"], dtype="S1"), TypeError)  # bytes
    ])
    def test_bad_groups(self, unequal_cluster_data, bad_groups, exc):
        d = unequal_cluster_data
        res = sm.OLS(d["y"], d["x"]).fit()
        with pytest.raises(exc):
            res.get_cluster_bootstrap(bad_groups)

    def test_groups_length_mismatch(self, unequal_cluster_data):
        d = unequal_cluster_data
        res = sm.OLS(d["y"], d["x"]).fit()
        with pytest.raises(ValueError, match="length"):
            res.get_cluster_bootstrap(d["groups"][:-1])
        with pytest.raises(ValueError, match="length"):
            res.get_cluster_bootstrap(d["groups"] + ["extra"])

    def test_cluster_count_boundary(self, unequal_cluster_data):
        d = unequal_cluster_data
        res = sm.OLS(d["y"], d["x"]).fit()
        n = len(d["y"])
        g6 = [f"c{i % 6}" for i in range(n)]
        cal = res.get_cluster_bootstrap(g6, n_resamples=99, random_state=0)
        assert cal.n_clusters == 6
        g5 = [f"c{i % 5}" for i in range(n)]
        with pytest.raises(ValueError, match="at least 6"):
            res.get_cluster_bootstrap(g5, n_resamples=99, random_state=0)

    def test_single_row_clusters(self):
        rng = np.random.default_rng(0)
        sizes = [1, 1, 1, 4, 5, 6, 1, 3]
        rows, groups = [], []
        for g, s in enumerate(sizes):
            x = rng.normal(size=s)
            y = 1 + 0.3 * x + rng.normal(size=s)
            rows.append(np.column_stack([np.ones(s), x, y]))
            groups += [g] * s
        data = np.vstack(rows)
        res = sm.OLS(data[:, 2], data[:, :2]).fit()
        cal = res.get_cluster_bootstrap(groups, n_resamples=199,
                                        random_state=0)
        assert cal.n_valid == 199
        assert 1 in cal.cluster_sizes

    @pytest.mark.parametrize("kwargs,exc", [
        ({"n_resamples": 98}, ValueError),
        ({"n_resamples": 99.0}, TypeError),
        ({"n_resamples": True}, TypeError),
        ({"n_resamples": None}, TypeError),
        ({"n_resamples": "999"}, TypeError),
        ({"batch_size": 0}, ValueError),
        ({"batch_size": -3}, ValueError),
        ({"batch_size": 2.5}, TypeError),
        ({"batch_size": True}, TypeError),
        ({"random_state": -1}, ValueError),
        ({"random_state": 2 ** 32}, ValueError),
        ({"random_state": 2 ** 32 - 1}, None),
        ({"random_state": 1.5}, TypeError),
        ({"random_state": False}, TypeError),
    ])
    def test_bad_parameters(self, unequal_cluster_data, kwargs, exc):
        d = unequal_cluster_data
        res = sm.OLS(d["y"], d["x"]).fit()
        n = len(d["y"])
        groups = [f"c{i % 8}" for i in range(n)]
        if exc is None:
            cal = res.get_cluster_bootstrap(groups, n_resamples=99,
                                            **kwargs)
            assert cal.n_valid == 99
        else:
            with pytest.raises(exc):
                res.get_cluster_bootstrap(groups, **kwargs)


# ----------------------------------------------------------------------
# unsupported models
# ----------------------------------------------------------------------
class TestUnsupportedModels:
    def test_wls_gls_rejected(self, unequal_cluster_data):
        d = unequal_cluster_data
        n = len(d["y"])
        groups = [f"c{i % 8}" for i in range(n)]
        res_w = sm.WLS(d["y"], d["x"], weights=np.arange(1., n + 1)).fit()
        with pytest.raises(ValueError, match="OLS"):
            res_w.get_cluster_bootstrap(groups)
        res_g = sm.GLS(d["y"], d["x"], sigma=np.eye(n)).fit()
        with pytest.raises(ValueError, match="OLS"):
            res_g.get_cluster_bootstrap(groups)

    def test_two_dimensional_endog_rejected(self, unequal_cluster_data):
        d = unequal_cluster_data
        n = len(d["y"])
        groups = [f"c{i % 8}" for i in range(n)]
        y2 = np.column_stack([d["y"], d["y"]])
        res = sm.OLS(y2, d["x"]).fit()
        with pytest.raises(ValueError, match="one-dimensional"):
            res.get_cluster_bootstrap(groups)

    def test_offset_rejected(self, unequal_cluster_data):
        d = unequal_cluster_data
        n = len(d["y"])
        groups = [f"c{i % 8}" for i in range(n)]
        res = sm.OLS(d["y"], d["x"], offset=d["y"] * 0.1).fit()
        with pytest.raises(ValueError, match="offset"):
            res.get_cluster_bootstrap(groups)

    def test_rank_deficient_rejected(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(30, 2))
        x = np.column_stack([np.ones(30), x, x[:, 0] + x[:, 1]])
        y = rng.normal(size=30)
        res = sm.OLS(y, x).fit()
        groups = [f"c{i % 7}" for i in range(30)]
        with pytest.raises(ValueError, match="full column rank"):
            res.get_cluster_bootstrap(groups)

    def test_non_ols_results_type(self, unequal_cluster_data):
        d = unequal_cluster_data
        with pytest.raises(TypeError):
            cb_module.get_cluster_bootstrap(
                object(), d["groups"], n_resamples=99, random_state=0,
                batch_size=32)


# ----------------------------------------------------------------------
# missing values, duplicate index, formula, categorical
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def missing_data():
    rng = np.random.default_rng(20260403)
    n = 80
    df = pd.DataFrame({
        "y": rng.normal(size=n),
        "x": rng.normal(size=n),
        "c": rng.choice(["a", "b", "c"], size=n),
    })
    df.index = [i % 23 for i in range(n)]  # duplicated index
    groups = np.array([f"store{i % 11}" for i in range(n)])
    # missing at physical positions (set by iloc on purpose)
    df.iloc[5, df.columns.get_loc("y")] = np.nan
    df.iloc[17, df.columns.get_loc("x")] = np.nan
    df.iloc[29, df.columns.get_loc("c")] = None
    df.iloc[63, df.columns.get_loc("y")] = np.nan
    return df, groups


class TestMissingAndFormula:

    def test_physical_row_deletion_alignment(self, missing_data):
        df, groups = missing_data
        res = smf.ols("y ~ x + C(c)", data=df).fit()
        cal = res.get_cluster_bootstrap(groups, n_resamples=199,
                                        random_state=11)
        # manual deletion at the same physical positions
        keep = np.ones(len(df), dtype=bool)
        keep[res.model.data.missing_row_idx] = False
        df_manual = df.iloc[keep].reset_index(drop=True)
        groups_manual = groups[keep]
        res_manual = smf.ols("y ~ x + C(c)", data=df_manual).fit()
        cal_manual = res_manual.get_cluster_bootstrap(
            groups_manual, n_resamples=199, random_state=11)
        # cluster identity and all calibration content coincide
        assert_array_equal_bitwise(cal.boot_params, cal_manual.boot_params)
        assert_array_equal_bitwise(cal.params, cal_manual.params)
        assert_array_equal_bitwise(cal.cluster_sizes,
                                   cal_manual.cluster_sizes)
        assert cal.n_clusters == cal_manual.n_clusters
        # and every downstream output coincides
        r_matrix = np.eye(4)
        t1 = res.t_test(r_matrix, inference=cal, simultaneous=True)
        t2 = res_manual.t_test(r_matrix, inference=cal_manual,
                               simultaneous=True)
        assert_array_equal_bitwise(np.atleast_1d(t1.pvalue),
                                   np.atleast_1d(t2.pvalue))
        assert_array_equal_bitwise(t1.conf_int(0.1), t2.conf_int(0.1))

    def test_missing_drop_ndarray(self):
        rng = np.random.default_rng(4)
        n = 50
        x = rng.normal(size=(n, 2))
        y = rng.normal(size=n)
        x[3, 0] = np.nan
        y[10] = np.nan
        x[41, 1] = np.nan
        groups = np.array([i % 9 for i in range(n)])
        res = sm.OLS(y, sm.add_constant(x), missing="drop").fit()
        cal = res.get_cluster_bootstrap(groups, n_resamples=99,
                                        random_state=0)
        keep = np.ones(n, dtype=bool)
        keep[[3, 10, 41]] = False
        res2 = sm.OLS(y[keep], sm.add_constant(x)[keep]).fit()
        cal2 = res2.get_cluster_bootstrap(groups[keep], n_resamples=99,
                                          random_state=0)
        assert_array_equal_bitwise(cal.boot_params, cal2.boot_params)

    def test_groups_length_uses_pre_deletion_rows(self, missing_data):
        df, groups = missing_data
        res = smf.ols("y ~ x + C(c)", data=df).fit()
        # post-deletion length must be rejected
        keep = np.ones(len(df), dtype=bool)
        keep[res.model.data.missing_row_idx] = False
        with pytest.raises(ValueError, match="length"):
            res.get_cluster_bootstrap(groups[keep])

    def test_formula_prediction_transform_and_labels(self, missing_data):
        df, groups = missing_data
        res = smf.ols("y ~ x + C(c)", data=df).fit()
        cal = res.get_cluster_bootstrap(groups, n_resamples=199,
                                        random_state=3)
        new = pd.DataFrame({"x": [-1.0, 0.0, 1.0, 0.5],
                            "c": ["a", "b", "c", "a"]},
                           index=["r", "r", "s", "t"])  # duplicated labels
        pred = res.get_prediction(new, inference=cal)
        frame = pred.summary_frame(alpha=0.1)
        assert list(frame.index) == ["r", "r", "s", "t"]
        assert len(frame) == 4
        # transform=False with the design matrix
        from patsy import dmatrix
        design = np.asarray(dmatrix(res.model.data.design_info, new))
        pred2 = res.get_prediction(design, transform=False, inference=cal)
        assert_array_equal_bitwise(pred.predicted_mean, pred2.predicted_mean)
        assert_array_equal_bitwise(
            pred.conf_int(alpha=0.1, simultaneous=True),
            pred2.conf_int(alpha=0.1, simultaneous=True))

    def test_new_data_missing_or_unknown_rejected(self, missing_data):
        df, groups = missing_data
        res = smf.ols("y ~ x + C(c)", data=df).fit()
        cal = res.get_cluster_bootstrap(groups, n_resamples=99,
                                        random_state=3)
        with pytest.raises(ValueError, match="missing|unknown"):
            res.get_prediction(pd.DataFrame({"x": [1.0, np.nan],
                                             "c": ["a", "b"]}),
                               inference=cal)
        with pytest.raises(ValueError, match="missing|unknown"):
            res.get_prediction(pd.DataFrame({"x": [1.0], "c": ["newlevel"]}),
                               inference=cal)
        with pytest.raises(ValueError, match="missing|unknown"):
            res.get_prediction(pd.DataFrame({"x": [1.0], "c": [None]}),
                               inference=cal)
        # raw array with NaN
        with pytest.raises(ValueError, match="non-finite"):
            res.get_prediction(np.array([[1.0, np.nan, 0.0, 0.0]]),
                               transform=False, inference=cal)
        # without inference the historical behavior is unchanged
        pred = res.get_prediction(pd.DataFrame({"x": [1.0, np.nan],
                                                "c": ["a", "b"]}))
        assert len(pred.predicted_mean) == 1

    def test_insample_prediction(self, missing_data):
        df, groups = missing_data
        res = smf.ols("y ~ x + C(c)", data=df).fit()
        cal = res.get_cluster_bootstrap(groups, n_resamples=99,
                                        random_state=3)
        pred = res.get_prediction(inference=cal)
        assert len(pred.predicted_mean) == int(res.nobs)
        frame = pred.summary_frame()
        assert len(frame) == int(res.nobs)


# ----------------------------------------------------------------------
# t_test behavior with the calibration
# ----------------------------------------------------------------------
class TestTTestInference:
    def test_constraint_input_forms(self, fitted_unequal):
        res, cal, d = fitted_unequal
        # 1-D numeric
        t1 = res.t_test([0., 1., 0.], inference=cal)
        assert np.ndim(t1.effect) == 1 and len(t1.effect) == 1
        # 2-D numeric
        t2 = res.t_test([[0., 1., 0.], [0., 0., 1.]], inference=cal)
        assert len(t2.effect) == 2
        # (R, q) tuple with scalar q (single row, as upstream t_test)
        t3 = res.t_test(([[0., 1., 0.]], 0.5), inference=cal)
        assert len(t3.effect) == 1
        # (R, q) tuple with vector q
        t3b = res.t_test(([[0., 1., 0.], [0., 0., 1.]], [0.5, -0.25]),
                         inference=cal)
        assert len(t3b.effect) == 2
        # string constraints
        df = pd.DataFrame({"y": d["y"], "x1": d["x"][:, 1],
                           "x2": d["x"][:, 2]})
        res_f = smf.ols("y ~ x1 + x2", data=df).fit()
        cal_f = res_f.get_cluster_bootstrap(d["groups"], n_resamples=999,
                                            random_state=20260402)
        t4 = res_f.t_test("x1 = 0.5, x2 = -0.3", inference=cal_f)
        assert len(t4.effect) == 2
        t5 = res_f.t_test("x1 = x2", inference=cal_f)
        assert len(t5.effect) == 1
        # numeric and string forms agree on the same model
        t6 = res_f.t_test(([[0., 1., 0.], [0., 0., 1.]], [0.5, -0.3]),
                          inference=cal_f)
        assert_array_equal_bitwise(np.atleast_1d(t4.pvalue),
                                   np.atleast_1d(t6.pvalue))
        assert_array_equal_bitwise(t4.conf_int(0.1), t6.conf_int(0.1))

    def test_nonzero_q(self, fitted_unequal):
        res, cal, d = fitted_unequal
        r_matrix = [[0., 1., 0.]]
        t0 = res.t_test((r_matrix, [0.0]), inference=cal)
        t1 = res.t_test((r_matrix, [1.5]), inference=cal)
        # q only changes the tested value, not the reference
        assert_array_equal_bitwise(t0.sd, t1.sd)
        assert_array_equal_bitwise(t0.effect, t1.effect)
        assert_array_equal_bitwise(t0.conf_int(0.1), t1.conf_int(0.1))
        assert not np.array_equal(np.atleast_1d(t0.pvalue),
                                  np.atleast_1d(t1.pvalue))
        # statistic shifts by q / sd
        assert_allclose = np.testing.assert_allclose
        assert_allclose(t1.tvalue, t0.tvalue - 1.5 / t0.sd, rtol=1e-12)

    def test_simultaneous_family_properties(self, fitted_unequal):
        res, cal, d = fitted_unequal
        alpha = 0.1
        r2 = np.array([[0., 1., 0.], [0., 0., 1.]])
        r_dup = np.vstack([r2, r2[1:]])
        r_big = np.vstack([r2, [1., 0., 0.]])
        ci2 = res.t_test(r2, inference=cal, simultaneous=True).conf_int(alpha)
        ci_dup = res.t_test(r_dup, inference=cal,
                            simultaneous=True).conf_int(alpha)
        ci_big = res.t_test(r_big, inference=cal,
                            simultaneous=True).conf_int(alpha)
        # duplicating a direction does not change the joint critical value
        assert_array_equal_bitwise(ci2, ci_dup[:2])
        assert_array_equal_bitwise(ci2[1], ci_dup[2])
        # enlarging the family cannot narrow the joint intervals
        width2 = ci2[:, 1] - ci2[:, 0]
        width_big = ci_big[:, 1] - ci_big[:, 0]
        assert np.all(width_big[:2] >= width2 - 1e-12)
        # reordering only reorders the output
        r_rev = r2[::-1]
        t_fwd = res.t_test(r2, inference=cal, simultaneous=True)
        t_rev = res.t_test(r_rev, inference=cal, simultaneous=True)
        assert_array_equal_bitwise(np.atleast_1d(t_rev.pvalue)[::-1],
                                   np.atleast_1d(t_fwd.pvalue))
        assert_array_equal_bitwise(t_rev.conf_int(alpha)[::-1],
                                   t_fwd.conf_int(alpha))
        # adjusted p-values dominate pointwise p-values
        t_pw = res.t_test(r2, inference=cal, simultaneous=False)
        assert np.all(np.atleast_1d(t_fwd.pvalue)
                      >= np.atleast_1d(t_pw.pvalue) - 1e-15)
        # joint intervals contain pointwise intervals
        ci_pw = t_pw.conf_int(alpha)
        assert np.all(ci2[:, 0] <= ci_pw[:, 0] + 1e-12)
        assert np.all(ci2[:, 1] >= ci_pw[:, 1] - 1e-12)

    def test_single_direction_simultaneous_equals_pointwise(
            self, fitted_unequal):
        res, cal, d = fitted_unequal
        r1 = [[0., 1., 0.]]
        t_pw = res.t_test(r1, inference=cal, simultaneous=False)
        t_si = res.t_test(r1, inference=cal, simultaneous=True)
        assert_array_equal_bitwise(np.atleast_1d(t_pw.pvalue),
                                   np.atleast_1d(t_si.pvalue))
        assert_array_equal_bitwise(t_pw.conf_int(0.1), t_si.conf_int(0.1))

    def test_pvalue_never_zero(self, fitted_unequal):
        res, cal, d = fitted_unequal
        # a tested value far in the tail still gives a nonzero p-value
        t_far = res.t_test(([[0., 1., 0.]], [1e6]), inference=cal)
        assert np.atleast_1d(t_far.pvalue)[0] == 1.0 / (cal.n_valid + 1)
        assert np.atleast_1d(t_far.pvalue)[0] > 0

    def test_pvalue_interval_inversion(self, fitted_unequal):
        res, cal, d = fitted_unequal
        alpha = 0.1
        r_matrix = np.array([[0., 1., 0.], [0., 0., 1.], [1., 1., 1.]])
        for simultaneous in (False, True):
            tt = res.t_test((r_matrix, np.zeros(3)), inference=cal,
                            simultaneous=simultaneous)
            ci = tt.conf_int(alpha)
            for j in range(3):
                for side in (0, 1):
                    q = np.zeros(3)
                    q[j] = ci[j, side]
                    pv = res.t_test((r_matrix, q), inference=cal,
                                    simultaneous=simultaneous).pvalue
                    assert np.atleast_1d(pv)[j] >= alpha - 1e-12
                # the midpoint is always inside
                q = np.zeros(3)
                q[j] = ci[j].mean()
                pv = res.t_test((r_matrix, q), inference=cal,
                                simultaneous=simultaneous).pvalue
                assert np.atleast_1d(pv)[j] >= alpha

    def test_bad_constraints_rejected(self, fitted_unequal):
        res, cal, d = fitted_unequal
        with pytest.raises(ValueError, match="all-zero"):
            res.t_test(np.zeros((1, 3)), inference=cal)
        with pytest.raises(ValueError, match="all-zero"):
            res.t_test([[0., 1., 0.], [0., 0., 0.]], inference=cal)
        with pytest.raises(ValueError, match="non-finite"):
            res.t_test([[0., np.inf, 0.]], inference=cal)
        with pytest.raises(ValueError, match="non-finite"):
            res.t_test([[0., np.nan, 0.]], inference=cal)
        with pytest.raises(ValueError, match="non-finite"):
            res.t_test(([[0., 1., 0.]], [np.nan]), inference=cal)
        with pytest.raises(ValueError, match="at least one"):
            res.t_test(np.empty((0, 3)), inference=cal)
        # wrong width is rejected (by patsy's constraint parser)
        with pytest.raises(ValueError):
            res.t_test([[0., 1.]], inference=cal)
        # mismatched q length is rejected (by patsy's constraint parser)
        with pytest.raises(ValueError):
            res.t_test(([[0., 1., 0.], [0., 0., 1.]], [0., 0., 0.]),
                       inference=cal)

    def test_zero_variance_direction_rejected(self, fitted_unequal):
        # a direction whose bootstrap coefficients never vary is not
        # estimable from the empirical reference
        res, cal, d = fitted_unequal
        params = cal.params
        boot_const = np.tile(params, (200, 1))
        cal_const = ClusterBootstrapCalibration(
            res._results, params, boot_const, n_resamples=200,
            failure_reasons={}, cluster_sizes=cal.cluster_sizes,
            nobs=cal.nobs, df_resid=cal.df_resid, random_state=0,
            batch_size=32, max_attempts=10)
        with pytest.raises(ValueError, match="zero bootstrap variance"):
            res.t_test([[0., 1., 0.]], inference=cal_const)
        with pytest.raises(ValueError, match="zero bootstrap variance"):
            res.get_prediction(d["x"][:3], inference=cal_const)

    def test_inference_combinations_rejected(self, fitted_unequal):
        res, cal, d = fitted_unequal
        r_matrix = [[0., 1., 0.]]
        with pytest.raises(ValueError, match="cov_p"):
            res.t_test(r_matrix, cov_p=np.eye(3), inference=cal)
        with pytest.raises(ValueError, match="use_t"):
            res.t_test(r_matrix, use_t=True, inference=cal)
        with pytest.raises(ValueError, match="use_t"):
            res.t_test(r_matrix, use_t=False, inference=cal)
        with pytest.raises(ValueError, match="simultaneous"):
            res.t_test(r_matrix, simultaneous=True)
        with pytest.raises(TypeError, match="ClusterBootstrapCalibration"):
            res.t_test(r_matrix, inference=object())
        # existing behavior without inference is untouched
        t_classic = res.t_test(r_matrix)
        assert t_classic.pvalue is not None
        t_cov = res.t_test(r_matrix, cov_p=np.eye(3))
        assert t_cov.pvalue is not None

    def test_wrong_owner_rejected(self, unequal_cluster_data):
        d = unequal_cluster_data
        res1 = sm.OLS(d["y"], d["x"]).fit()
        res2 = sm.OLS(d["y"].copy(), d["x"].copy()).fit()
        cal = res1.get_cluster_bootstrap(d["groups"], n_resamples=99,
                                         random_state=0)
        with pytest.raises(ValueError, match="different fitted results"):
            res2.t_test([[0., 1., 0.]], inference=cal)
        with pytest.raises(ValueError, match="different fitted results"):
            res2.get_prediction(d["x"][:3], inference=cal)
        # same shapes, even same data, still a different fit
        res3 = sm.OLS(d["y"], d["x"]).fit()
        with pytest.raises(ValueError, match="different fitted results"):
            res3.t_test([[0., 1., 0.]], inference=cal)


# ----------------------------------------------------------------------
# prediction with the calibration
# ----------------------------------------------------------------------
class TestPredictionInference:
    def test_contrast_prediction_consistency(self, fitted_unequal):
        res, cal, d = fitted_unequal
        rng = np.random.default_rng(13)
        a_new = np.column_stack(
            [np.ones(9), rng.normal(size=9), rng.normal(size=9)])
        pred = res.get_prediction(a_new, inference=cal)
        for simultaneous in (False, True):
            tt_s = res.t_test(a_new, inference=cal,
                              simultaneous=simultaneous)
            np.testing.assert_allclose(tt_s.effect, pred.predicted_mean,
                                       rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(tt_s.sd, pred.se_mean,
                                       rtol=1e-10, atol=1e-12)
            for alpha in (0.01, 0.05, 0.1, 0.25):
                np.testing.assert_allclose(
                    tt_s.conf_int(alpha),
                    pred.conf_int(alpha=alpha, simultaneous=simultaneous),
                    rtol=1e-10, atol=1e-12)

    def test_obs_rejected_and_obs_ci_nan(self, fitted_unequal):
        res, cal, d = fitted_unequal
        pred = res.get_prediction(d["x"][:4], inference=cal)
        with pytest.raises(NotImplementedError, match="obs"):
            pred.conf_int(obs=True)
        frame = pred.summary_frame()
        assert frame["obs_ci_lower"].isna().all()
        assert frame["obs_ci_upper"].isna().all()
        assert list(frame.columns) == ["mean", "mean_se", "mean_ci_lower",
                                       "mean_ci_upper", "obs_ci_lower",
                                       "obs_ci_upper"]

    def test_summary_frame_matches_attributes(self, fitted_unequal):
        res, cal, d = fitted_unequal
        pred = res.get_prediction(d["x"][:6], inference=cal)
        for simultaneous in (False, True):
            frame = pred.summary_frame(alpha=0.1, simultaneous=simultaneous)
            ci = pred.conf_int(alpha=0.1, simultaneous=simultaneous)
            assert_array_equal_bitwise(frame["mean"].to_numpy(),
                                       pred.predicted_mean)
            assert_array_equal_bitwise(frame["mean_se"].to_numpy(),
                                       pred.se_mean)
            assert_array_equal_bitwise(frame["mean_ci_lower"].to_numpy(),
                                       ci[:, 0])
            assert_array_equal_bitwise(frame["mean_ci_upper"].to_numpy(),
                                       ci[:, 1])

    def test_predicted_mean_equals_conditional_mean(self, fitted_unequal):
        res, cal, d = fitted_unequal
        a_new = d["x"][:8]
        pred_boot = res.get_prediction(a_new, inference=cal)
        pred_plain = res.get_prediction(a_new)
        assert_array_equal_bitwise(pred_boot.predicted_mean,
                                   pred_plain.predicted_mean)

    def test_weights_rejected_with_inference(self, fitted_unequal):
        res, cal, d = fitted_unequal
        with pytest.raises(ValueError, match="weights"):
            res.get_prediction(d["x"][:3], weights=np.ones(3),
                               inference=cal)

    def test_prediction_interface_defaults(self, fitted_unequal):
        res, cal, d = fitted_unequal
        pred = res.get_prediction(d["x"][:5], inference=cal)
        # PredictionResults-compatible interface
        assert pred.predicted_mean.shape == (5,)
        assert pred.var_pred_mean.shape == (5,)
        assert pred.se_mean.shape == (5,)
        assert np.all(np.isnan(pred.se_obs))
        ci = pred.conf_int()
        assert ci.shape == (5, 2)
        # alpha validation
        with pytest.raises(ValueError, match="alpha"):
            pred.conf_int(alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            pred.conf_int(alpha=1.0)
        with pytest.raises(ValueError, match="alpha"):
            pred.conf_int(alpha=-0.1)


# ----------------------------------------------------------------------
# invalid resamples
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def fragile_design():
    # dummy column present in only 3 of 8 clusters: resamples that
    # miss all three clusters are rank deficient
    rng = np.random.default_rng(20260404)
    rows, groups = [], []
    for g in range(8):
        s = 9
        x1 = rng.normal(size=s)
        x2 = np.ones(s) if g < 3 else np.zeros(s)
        y = 1 + 0.5 * x1 + 1.5 * x2 + rng.normal(size=s)
        rows.append(np.column_stack([np.ones(s), x1, x2, y]))
        groups += [f"c{g}"] * s
    data = np.vstack(rows)
    return data[:, :3], data[:, 3], groups


class TestInvalidResamples:
    def test_retries_recover_by_default(self, fragile_design):
        x, y, groups = fragile_design
        res = sm.OLS(y, x).fit()
        cal = res.get_cluster_bootstrap(groups, n_resamples=99,
                                        random_state=0)
        assert cal.n_valid == 99
        assert cal.n_failed == 0

    def test_failures_reported_with_warning(self, fragile_design,
                                            monkeypatch):
        x, y, groups = fragile_design
        res = sm.OLS(y, x).fit()
        monkeypatch.setattr(cb_module, "MAX_ATTEMPTS", 1)
        with pytest.warns(UserWarning, match="invalid"):
            cal = res.get_cluster_bootstrap(groups, n_resamples=99,
                                            random_state=0)
        assert cal.n_valid + cal.n_failed == 99
        assert cal.n_failed > 0
        assert cal.failure_reasons == {
            "rank-deficient resampled design matrix": cal.n_failed}
        # inference uses the valid resamples only
        tt = res.t_test([[0., 1., 0.]], inference=cal)
        assert 0 < np.atleast_1d(tt.pvalue)[0] <= 1
        assert cal.boot_params.shape == (cal.n_valid, 3)

    def test_too_many_failures_raise(self, fragile_design, monkeypatch):
        x, y, groups = fragile_design
        # dummy in a single cluster: failure probability per draw is high
        x2 = x.copy()
        x2[:, 2] = 0.0
        x2[:9, 2] = 1.0  # only cluster c0 carries the dummy
        res = sm.OLS(y, x2).fit()
        monkeypatch.setattr(cb_module, "MAX_ATTEMPTS", 1)
        with pytest.raises(ValueError, match="only .* of .* resamples"):
            res.get_cluster_bootstrap(groups, n_resamples=99,
                                      random_state=0)
        # the results instance is not corrupted: a new call works
        monkeypatch.setattr(cb_module, "MAX_ATTEMPTS", MAX_ATTEMPTS)
        cal = res.get_cluster_bootstrap(groups, n_resamples=99,
                                        random_state=0)
        assert cal.n_valid == 99
        assert res.t_test([[0., 1., 0.]]).pvalue is not None

    def test_coupon_collector_design_raises(self):
        # 6 clusters with one dummy column each and no intercept: a
        # resample is estimable only if every cluster is drawn, so almost
        # every resample is invalid and calibration must fail
        rng = np.random.default_rng(20260405)
        rows, groups = [], []
        for g in range(6):
            s = 5
            x1 = rng.normal(size=s)
            dummies = np.zeros((s, 6))
            dummies[:, g] = 1.0
            y = 1 + 0.5 * x1 + g + rng.normal(size=s)
            rows.append(np.column_stack([x1, dummies, y]))
            groups += [g] * s
        data = np.vstack(rows)
        res = sm.OLS(data[:, 7], data[:, :7]).fit()
        assert res.model.exog.shape[1] == 7  # x1 plus six cluster dummies
        with pytest.raises(ValueError, match="only .* of .* resamples"):
            res.get_cluster_bootstrap(groups, n_resamples=99,
                                      random_state=0)


# ----------------------------------------------------------------------
# existing behavior is preserved
# ----------------------------------------------------------------------
class TestExistingBehaviorUnchanged:
    def test_t_test_default_unchanged(self, fitted_unequal):
        res, cal, d = fitted_unequal
        t = res.t_test([[0., 1., 0.], [0., 0., 1.]])
        assert t.distribution == "t"
        ci = t.conf_int()
        assert ci.shape == (2, 2)
        t_str = None
        df = pd.DataFrame({"y": d["y"], "x1": d["x"][:, 1],
                           "x2": d["x"][:, 2]})
        res_f = smf.ols("y ~ x1 + x2", data=df).fit()
        t_str = res_f.t_test("x1 = 0, x2 = 0")
        assert len(t_str.effect) == 2

    def test_get_prediction_default_unchanged(self, fitted_unequal):
        res, cal, d = fitted_unequal
        pred = res.get_prediction(d["x"][:4])
        ci = pred.conf_int(obs=True)
        frame = pred.summary_frame()
        assert not frame["obs_ci_lower"].isna().any()
        assert ci.shape == (4, 2)

    def test_cluster_cov_still_works(self, fitted_unequal):
        res, cal, d = fitted_unequal
        res_c = sm.OLS(d["y"], d["x"]).fit(
            cov_type="cluster", cov_kwds={"groups": d["groups"]})
        assert res_c.cov_type == "cluster"
        t = res_c.t_test([[0., 1., 0.]])
        assert t.pvalue is not None
