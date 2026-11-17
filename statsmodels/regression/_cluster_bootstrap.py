"""
Cluster bootstrap inference for OLS with related observations.

This module implements a pairs (cluster resampling) bootstrap calibration
for ordinary least squares results and routes it into linear contrasts
(:meth:`statsmodels.regression.linear_model.RegressionResults.t_test`) and
into conditional mean predictions
(:meth:`statsmodels.regression.linear_model.RegressionResults.get_prediction`).

The calibration is created by
:meth:`statsmodels.regression.linear_model.RegressionResults.get_cluster_bootstrap`
and is bound to the fitted results instance that produced it.  See
:class:`ClusterBootstrapCalibration` for a precise definition of the
statistical method.
"""

from collections import Counter
import warnings
import weakref

import numpy as np
import pandas as pd
from scipy import linalg

from statsmodels.regression._prediction import PredictionResults
from statsmodels.stats.contrast import ContrastResults
from statsmodels.tools.validation import bool_like, float_like, int_like

__all__ = [
    "ClusterBootstrapCalibration",
    "ClusterBootstrapContrastResults",
    "ClusterBootstrapPredictionResults",
    "get_cluster_bootstrap",
]

#: Maximum number of draws tried for a single bootstrap resample before the
#: resample is counted as failed.
MAX_ATTEMPTS = 10

#: Minimum share of valid resamples required for a calibration to succeed.
MIN_VALID_SHARE = 0.9

#: Minimum number of distinct clusters required (after row deletion).
MIN_CLUSTERS = 6


def _check_alpha(alpha):
    """Validate a significance level, returning it as a float."""
    alpha = float_like(alpha, "alpha", optional=False, strict=False)
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    return alpha


def _critical_rank(alpha, n_boot):
    """
    Zero-based rank of the two-sided critical value.

    The critical value at level ``alpha`` is the ``k``-th smallest of the
    ``n_boot`` absolute studentized bootstrap statistics with
    ``k = min(max(ceil((n_boot + 1) * (1 - alpha)), 1), n_boot)``, evaluated
    in float64.  With the p-value convention
    ``(1 + #{stat >= stat_obs}) / (n_boot + 1)`` this rank rule makes a
    tested value lie inside the ``1 - alpha`` interval if and only if its
    p-value is >= alpha (up to ties).
    """
    k = int(np.ceil((n_boot + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n_boot)
    return k - 1


def _linear_dev(r_chunk, boot_centered):
    """
    Row-wise dot products with the bootstrap coefficient deviations.

    Computes ``D[i, b] = sum_k r_chunk[i, k] * boot_centered[b, k]`` by
    accumulating one parameter at a time.  Every element is summed in the
    same parameter order regardless of the chunk size, so results are
    bitwise identical for any chunking of the rows of ``r_chunk``.
    """
    out = np.zeros((r_chunk.shape[0], boot_centered.shape[0]))
    for k in range(r_chunk.shape[1]):
        out += r_chunk[:, k:k + 1] * boot_centered[:, k:k + 1].T
    return out


def _fit_resampled(x, y, cluster_id, counts):
    """
    OLS refit on a cluster-resampled data set.

    ``counts[g]`` is the multiplicity of cluster ``g`` in the resample.
    Returns the resampled coefficient vector.  Raises
    ``numpy.linalg.LinAlgError`` if the resampled design is rank
    deficient (the normal equations are not positive definite).
    """
    w = counts[cluster_id].astype(np.float64)
    xw = x * w[:, None]
    xtwx = x.T @ xw
    xtwy = xw.T @ y
    cho = linalg.cho_factor(xtwx)
    return linalg.cho_solve(cho, xtwy)


def _first_occurrence_ids(labels):
    """
    Map cluster labels to integer ids in order of first appearance.

    Numbering clusters by first occurrence (rather than by sorted label
    value) makes the calibration invariant to relabeling of the clusters:
    integer and string labels of the same partition in the same row order
    produce identical resamples.
    """
    _, first_pos, inverse = np.unique(labels, return_index=True,
                                      return_inverse=True)
    rank = np.empty(len(first_pos), dtype=np.intp)
    rank[np.argsort(first_pos)] = np.arange(len(first_pos))
    return rank[inverse]


def _validate_groups(groups):
    """
    Validate and copy the cluster labels.

    Returns a 1-D ndarray containing either only non-empty strings or only
    non-boolean integers.  Raises TypeError for unsupported containers or
    element types and ValueError for missing or empty labels.
    """
    if isinstance(groups, pd.Series):
        arr = groups.to_numpy()
    elif isinstance(groups, (list, tuple)):
        # object dtype preserves element types; np.asarray would coerce
        # mixed input (e.g. str and int) to a single dtype
        arr = np.asarray(groups, dtype=object)
    elif isinstance(groups, np.ndarray):
        arr = groups
    else:
        raise TypeError(
            "groups must be a 1-D list, tuple, ndarray, or Series, got "
            f"{type(groups).__name__}"
        )
    if arr.ndim != 1:
        raise ValueError(
            f"groups must be one-dimensional, got shape {arr.shape}"
        )
    # own copy so later mutation of the caller's container cannot leak in
    arr = np.array(arr, copy=True)

    if arr.dtype.kind == "U":
        if np.any(arr == ""):
            raise ValueError("groups contains empty string cluster labels")
        return arr
    if arr.dtype.kind == "S":
        raise TypeError("groups labels must be str, not bytes")
    if arr.dtype.kind in "iu":
        return arr
    if arr.dtype.kind == "b":
        raise TypeError("groups labels must not be boolean")
    if arr.dtype.kind == "f":
        if np.any(np.isnan(arr)):
            raise ValueError("groups contains missing cluster labels")
        raise TypeError(
            "groups elements must be all non-empty strings or all "
            "non-boolean integers, got float dtype"
        )
    if arr.dtype.kind == "O":
        flat = arr.ravel()
        has_missing = any(
            v is None or (isinstance(v, float) and np.isnan(v)) for v in flat
        )
        if has_missing:
            raise ValueError("groups contains missing cluster labels")
        if all(isinstance(v, str) for v in flat):
            if any(len(v) == 0 for v in flat):
                raise ValueError(
                    "groups contains empty string cluster labels"
                )
            return arr.astype(str)
        if all(
            isinstance(v, (int, np.integer))
            and not isinstance(v, (bool, np.bool_))
            for v in flat
        ):
            return arr.astype(np.int64)
        raise TypeError(
            "groups elements must be all non-empty strings or all "
            "non-boolean integers"
        )
    raise TypeError(
        "groups elements must be all non-empty strings or all non-boolean "
        f"integers, got dtype {arr.dtype}"
    )


def _validate_results(results):
    """
    Check that a results instance is an unweighted, full-rank OLS fit.

    Returns the model instance.  Raises TypeError or ValueError otherwise.
    """
    from statsmodels.regression.linear_model import OLS, RegressionResults

    if not isinstance(results, RegressionResults):
        raise TypeError(
            "get_cluster_bootstrap requires a results instance from "
            "OLS.fit(), got "
            f"{type(results).__name__}"
        )
    model = results.model
    if type(model) is not OLS:
        raise ValueError(
            "cluster bootstrap calibration is only supported for OLS "
            f"models, got {type(model).__name__}"
        )
    if getattr(model, "offset", None) is not None:
        raise ValueError("OLS models with an offset are not supported")
    if model.endog.ndim != 1 or np.ndim(results.params) != 1:
        raise ValueError(
            "only a one-dimensional dependent variable is supported"
        )
    if not np.all(np.asarray(model.weights) == 1.0):
        raise ValueError("only unweighted OLS models are supported")
    exog = np.asarray(model.exog)
    endog = np.asarray(model.endog)
    if not (np.all(np.isfinite(exog)) and np.all(np.isfinite(endog))):
        raise ValueError(
            "the design matrix and the response must contain only finite "
            "values"
        )
    k_vars = exog.shape[1]
    rank = getattr(model, "rank", None)
    if rank is None:
        rank = np.linalg.matrix_rank(exog)
    if int(rank) != k_vars:
        raise ValueError(
            "the design matrix must have full column rank, got rank "
            f"{int(rank)} for {k_vars} columns"
        )
    if not float(results.df_resid) > 0.0:
        raise ValueError(
            "the model must have positive residual degrees of freedom"
        )
    return model


def get_cluster_bootstrap(results, groups, *, n_resamples=999, random_state=0,
                          batch_size=32):
    """
    Calibrate a pairs cluster bootstrap for an OLS results instance.

    This is the implementation behind
    ``RegressionResults.get_cluster_bootstrap``; see that method and
    :class:`ClusterBootstrapCalibration` for documentation.
    """
    model = _validate_results(results)

    n_resamples = int_like(n_resamples, "n_resamples", strict=True)
    if n_resamples < 99:
        raise ValueError(
            f"n_resamples must be at least 99, got {n_resamples}"
        )
    batch_size = int_like(batch_size, "batch_size", strict=True)
    if batch_size < 1:
        raise ValueError(f"batch_size must be a positive integer, got "
                         f"{batch_size}")
    random_state = int_like(random_state, "random_state", strict=True)
    if not 0 <= random_state < 2 ** 32:
        raise ValueError(
            "random_state must be an integer in [0, 2**32), got "
            f"{random_state}"
        )

    groups_arr = _validate_groups(groups)

    nobs = int(model.nobs)
    missing_row_idx = getattr(model.data, "missing_row_idx", None) or []
    n_input = nobs + len(missing_row_idx)
    if groups_arr.shape[0] != n_input:
        raise ValueError(
            f"groups has length {groups_arr.shape[0]} but the model input "
            f"has {n_input} rows ({nobs} rows kept in the fit plus "
            f"{len(missing_row_idx)} rows dropped by missing-value "
            "handling); groups must be given in the physical row order of "
            "the modeling input before any rows were dropped"
        )
    keep = np.ones(n_input, dtype=bool)
    if len(missing_row_idx):
        keep[np.asarray(missing_row_idx, dtype=np.intp)] = False
    groups_kept = groups_arr[keep]

    cluster_id = _first_occurrence_ids(groups_kept)
    cluster_sizes = np.bincount(cluster_id)
    n_clusters = cluster_sizes.shape[0]
    if n_clusters < MIN_CLUSTERS:
        raise ValueError(
            f"at least {MIN_CLUSTERS} distinct clusters are required after "
            f"dropping rows, got {n_clusters}"
        )

    x = np.asarray(model.exog, dtype=np.float64)
    y = np.asarray(model.endog, dtype=np.float64)
    params = np.asarray(results.params, dtype=np.float64).copy()
    n_params = x.shape[1]

    seed_seq = np.random.SeedSequence(random_state)
    children = seed_seq.spawn(n_resamples)
    boot = np.empty((n_resamples, n_params), dtype=np.float64)
    failures = []
    n_valid = 0
    for b in range(n_resamples):
        gen = np.random.default_rng(children[b])
        beta_b = None
        reason = None
        for _ in range(MAX_ATTEMPTS):
            idx = gen.integers(0, n_clusters, size=n_clusters)
            counts = np.bincount(idx, minlength=n_clusters)
            try:
                candidate = _fit_resampled(x, y, cluster_id, counts)
            except np.linalg.LinAlgError:
                reason = "rank-deficient resampled design matrix"
                continue
            if not np.all(np.isfinite(candidate)):
                reason = "non-finite resampled parameter estimates"
                continue
            beta_b = candidate
            break
        if beta_b is None:
            failures.append(reason)
        else:
            boot[n_valid] = beta_b
            n_valid += 1
    boot = boot[:n_valid]
    failure_reasons = dict(Counter(failures))
    n_failed = n_resamples - n_valid

    if n_valid < int(np.ceil(MIN_VALID_SHARE * n_resamples)):
        raise ValueError(
            "cluster bootstrap calibration failed: only "
            f"{n_valid} of {n_resamples} resamples were estimable "
            f"(at least {MIN_VALID_SHARE:.0%} are required); failure "
            f"reasons: {failure_reasons}"
        )
    if n_failed:
        warnings.warn(
            f"{n_failed} of {n_resamples} bootstrap resamples were invalid "
            f"and were dropped; failure reasons: {failure_reasons}; "
            f"inference uses the remaining {n_valid} resamples",
            UserWarning,
            stacklevel=3,
        )

    return ClusterBootstrapCalibration(
        results,
        params,
        boot,
        n_resamples=n_resamples,
        failure_reasons=failure_reasons,
        cluster_sizes=cluster_sizes,
        nobs=nobs,
        df_resid=float(results.df_resid),
        random_state=random_state,
        batch_size=batch_size,
        max_attempts=MAX_ATTEMPTS,
    )


class ClusterBootstrapCalibration:
    """
    Pairs cluster bootstrap calibration for an OLS fit.

    Instances are created by
    :meth:`statsmodels.regression.linear_model.RegressionResults.get_cluster_bootstrap`
    and are bound to that fitted results instance; passing the calibration
    as ``inference=...`` to ``t_test`` or ``get_prediction`` of any other
    results instance raises an error.

    Parameters
    ----------
    results : RegressionResults
        The fitted OLS results instance the calibration belongs to.
    params : ndarray
        The ``p`` coefficient estimates of the original fit.
    boot_params : ndarray
        The ``(n_valid, p)`` array of coefficients refit on
        cluster-resampled data.
    n_resamples : int
        The number of requested resamples.
    failure_reasons : dict
        Mapping from failure description to the number of occurrences.
    cluster_sizes : ndarray
        The number of kept observations in each cluster.
    nobs : int
        The number of observations used in the fit.
    df_resid : float
        The residual degrees of freedom of the original fit.
    random_state : int
        The seed the calibration was created with.
    batch_size : int
        Workspace chunk size used by queries.
    max_attempts : int
        Maximum number of draws tried per resample.

    Attributes
    ----------
    params : ndarray
        The ``p`` coefficient estimates of the original fit.
    boot_params : ndarray
        The ``(n_valid, p)`` array of coefficients refit on
        cluster-resampled data.  This is the only resampling output that is
        stored; resampled data sets are never materialized.
    n_resamples : int
        The number of requested resamples.
    n_valid : int
        The number of valid resamples actually used (``B`` below).
    n_failed : int
        The number of requested resamples that could not be estimated.
    failure_reasons : dict
        Mapping from failure description to the number of occurrences.
    n_clusters : int
        The number of distinct clusters ``G`` after row deletion.
    cluster_sizes : ndarray
        The number of kept observations in each cluster.
    nobs : int
        The number of observations used in the fit.
    df_resid : float
        The residual degrees of freedom of the original fit.
    random_state : int
        The seed the calibration was created with.
    batch_size : int
        Workspace chunk size used by queries.  It does not affect any
        numerical result.
    max_attempts : int
        Maximum number of draws tried per resample before it is counted as
        failed.

    See Also
    --------
    statsmodels.regression.linear_model.RegressionResults.get_cluster_bootstrap :
        Create a cluster bootstrap calibration for an OLS fit.
    statsmodels.regression.linear_model.RegressionResults.get_prediction :
        Predictions that can use the calibration as ``inference``.
    statsmodels.base.model.LikelihoodModelResults.t_test :
        Linear hypothesis tests that can use the calibration as
        ``inference``.

    Notes
    -----
    **Statistical method.**  The resampling unit is the whole cluster.
    Each bootstrap resample draws ``G`` clusters independently and with
    replacement from the ``G`` observed clusters (pairs cluster
    bootstrap), so cluster ``g`` enters with integer multiplicity
    ``c_g >= 0`` and the resampled coefficient is the OLS refit

    .. math::

        \\hat\\beta^*_b = (X^\\top W_b X)^{-1} X^\\top W_b y,
        \\qquad W_b = \\operatorname{diag}(c_{g(i)}),

    computed from the cluster multiplicities without materializing the
    resampled data.  The target is the sampling distribution of
    ``beta_hat - beta`` under independent sampling of clusters; it is
    approximated by the bootstrap distribution of
    ``beta*_b - beta_hat`` (empirical centering at the observed estimate;
    the bootstrap mean is *not* subtracted).

    For a linear direction ``r`` the scale is the bootstrap standard
    deviation ``s(r) = sd_b(r' beta*_b)`` with ``ddof = 1``, and the
    studentized reference statistics are
    ``T_b(r) = r'(beta*_b - beta_hat) / s(r)``.  With the observed
    statistic ``t_obs = (r' beta_hat - q) / s(r)`` the two-sided p-value
    of ``H0: r' beta = q`` is

    .. math::

        p = \\frac{1 + \\#\\{b : |T_b(r)| \\ge |t_{obs}|\\}}{B + 1},

    which is never smaller than ``1 / (B + 1)`` and therefore never a
    spurious zero.  The two-sided ``1 - alpha`` interval is
    ``r' beta_hat +/- c s(r)`` where ``c`` is the ``k``-th order statistic
    of ``{|T_b(r)|}`` with
    ``k = min(max(ceil((B + 1) (1 - alpha)), 1), B)`` evaluated in
    float64; this rank rule makes the interval the exact inversion of the
    p-value above (up to ties).

    A simultaneous family ``R = (r_1, ..., r_k)`` declared in one call
    uses the family-wise statistic ``M_b = max_j |T_b(r_j)|``: the joint
    band is ``r_j' beta_hat +/- c_sim s(r_j)`` with ``c_sim`` the same
    order-statistic rule applied to ``{M_b}``, and the adjusted p-value of
    direction ``j`` replaces ``|T_b(r_j)|`` by ``M_b`` in the p-value
    formula.  The maximum is taken over the actual joint bootstrap
    distribution, so the dependence between directions is preserved (no
    Bonferroni or independence approximation).  Duplicating a direction
    leaves ``c_sim`` unchanged and enlarging the family cannot decrease
    it.  The tested values ``q`` enter only ``t_obs``; changing ``q``
    never changes the resampling reference.

    **Random number generation.**  Clusters are numbered ``0..G-1`` in
    order of first appearance in the kept rows, so relabeling the
    clusters does not change the calibration.
    ``np.random.SeedSequence(random_state)`` is spawned into
    ``n_resamples`` independent child sequences, one per resample.
    Resample ``b`` draws ``G`` uniform cluster indices via
    ``integers(0, G, G)`` whose bincount gives the multiplicities
    ``c_g``.  Because each resample has its own sequence, results do not
    depend on ``batch_size``, and the global NumPy random state is never
    touched.

    **Invalid resamples.**  A resample is invalid if the resampled design
    is rank deficient (Cholesky failure) or the refit is non-finite.
    Each resample is re-drawn up to ``max_attempts`` (10) times with fresh
    draws from its own sequence; if all attempts fail, the resample is
    dropped and counted in ``n_failed`` and ``failure_reasons``, and a
    ``UserWarning`` is issued.  Calibration raises ``ValueError`` if fewer
    than 90% of the requested resamples are valid.  All tail
    probabilities use ``B = n_valid``.

    **Assumptions and limitations.**  Observations are independent
    across clusters; within a cluster, arbitrary dependence and
    heteroskedasticity are allowed because the whole cluster is
    resampled.  The observed clusters are treated as draws from a common
    population of clusters, so cluster sizes and within-cluster
    covariance patterns may differ freely as long as they are part of
    that draw.  The conditional mean must be linear with a full column
    rank design.  With few clusters (close to the minimum of 6), strongly
    unequal cluster sizes, or systematically different cluster-level
    error variances, coverage can deviate from the nominal level; the
    method is asymptotic in ``G``, not in the cluster sizes.  Singleton
    clusters are allowed.  The bootstrap reference is computed once at
    calibration; queries are linear algebra on the stored
    ``boot_params`` and never refit.

    Examples
    --------
    >>> import numpy as np
    >>> import statsmodels.api as sm
    >>> rng = np.random.default_rng(0)
    >>> sizes = [6, 9, 4, 7, 8, 5, 10, 6]
    >>> rows, groups = [], []
    >>> for g, s in enumerate(sizes):
    ...     x = rng.normal(size=s)
    ...     y = 1.0 + 0.5 * x + rng.normal() + rng.normal(size=s) * 0.5
    ...     rows.append(np.column_stack([np.ones(s), x, y]))
    ...     groups += [g] * s
    >>> data = np.vstack(rows)
    >>> res = sm.OLS(data[:, 2], data[:, :2]).fit()
    >>> cal = res.get_cluster_bootstrap(groups, random_state=0)
    >>> cal.n_valid
    999
    >>> res.t_test([[0, 1]], inference=cal).conf_int(0.1).shape
    (1, 2)
    """

    def __init__(self, results, params, boot_params, *, n_resamples,
                 failure_reasons, cluster_sizes, nobs, df_resid,
                 random_state, batch_size, max_attempts):
        self._owner_ref = weakref.ref(results)
        self.params = params
        self.boot_params = boot_params
        self.n_resamples = n_resamples
        self.n_valid = int(boot_params.shape[0])
        self.n_failed = n_resamples - self.n_valid
        self.failure_reasons = dict(failure_reasons)
        self.cluster_sizes = cluster_sizes
        self.n_clusters = int(cluster_sizes.shape[0])
        self.nobs = nobs
        self.df_resid = df_resid
        self.random_state = random_state
        self.batch_size = batch_size
        self.max_attempts = max_attempts

    def __repr__(self):
        return (
            "ClusterBootstrapCalibration("
            f"n_clusters={self.n_clusters}, nobs={self.nobs}, "
            f"n_resamples={self.n_resamples}, n_valid={self.n_valid}, "
            f"n_failed={self.n_failed}, random_state={self.random_state})"
        )

    # ------------------------------------------------------------------
    # ownership
    # ------------------------------------------------------------------
    def _check_owner(self, results):
        """Reject use with a results instance other than the owner's."""
        owner = self._owner_ref()
        if owner is None:
            raise ValueError(
                "the fitted results instance that produced this "
                "calibration no longer exists"
            )
        if owner is not results:
            raise ValueError(
                "this calibration belongs to a different fitted results "
                "instance; create a separate calibration with "
                "get_cluster_bootstrap for each fit"
            )

    # ------------------------------------------------------------------
    # core linear inference
    # ------------------------------------------------------------------
    def _infer(self, r_matrix, q, alpha, simultaneous, need_pvalues):
        """
        Empirical-reference inference for linear directions.

        Parameters
        ----------
        r_matrix : ndarray, (k, p)
            The directions (rows) to evaluate.
        q : ndarray, (k,) or None
            The tested values; zeros if None.
        alpha : float or None
            Significance level for critical values; None skips them.
        simultaneous : bool
            Whether the directions form one simultaneous family.
        need_pvalues : bool
            Whether to compute p-values against ``q``.

        Returns
        -------
        dict with keys ``effect``, ``sd`` and, depending on the flags,
        ``statistic``, ``pvalue``, ``crit``.
        """
        r_matrix = np.ascontiguousarray(r_matrix, dtype=np.float64)
        k_rows = r_matrix.shape[0]
        n_boot = self.n_valid
        boot_centered = self.boot_params - self.params

        effect = r_matrix.dot(self.params)
        if q is None:
            q = np.zeros(k_rows)
        else:
            q = np.asarray(q, dtype=np.float64)

        chunk = max(1, min(self.batch_size, k_rows))

        # pass 1: first and second moments of the bootstrap deviations
        sum_d = np.zeros(k_rows)
        sumsq_d = np.zeros(k_rows)
        for start in range(0, k_rows, chunk):
            rows = slice(start, min(start + chunk, k_rows))
            dev = _linear_dev(r_matrix[rows], boot_centered)
            sum_d[rows] = dev.sum(axis=1)
            sumsq_d[rows] = (dev * dev).sum(axis=1)
        var = (sumsq_d - sum_d * sum_d / n_boot) / (n_boot - 1.0)
        # guard against catastrophic cancellation in the one-pass formula
        var = np.where((var < 0) & (var > -1e-10 * sumsq_d / n_boot), 0.0,
                       var)
        if np.any(~np.isfinite(var)) or np.any(var < 0):
            raise ValueError(
                "could not compute a finite bootstrap variance for at "
                "least one direction"
            )
        sd = np.sqrt(var)
        if np.any(sd == 0.0):
            raise ValueError(
                "at least one direction has zero bootstrap variance and "
                "is not estimable from the cluster bootstrap reference"
            )

        out = {"effect": effect, "sd": sd}
        if alpha is None and not need_pvalues:
            return out

        if q.shape[0] != k_rows:
            raise ValueError(
                "r_matrix and q_matrix must have the same number of rows"
            )
        t_obs = (effect - q) / sd
        abs_t_obs = np.abs(t_obs)

        # pass 2: studentized statistics, counts, and the family maximum
        counts = np.zeros(k_rows, dtype=np.int64) if need_pvalues else None
        crit = np.zeros(k_rows) if alpha is not None else None
        max_t = np.zeros(n_boot) if simultaneous else None
        rank = _critical_rank(alpha, n_boot) if alpha is not None else None
        for start in range(0, k_rows, chunk):
            rows = slice(start, min(start + chunk, k_rows))
            dev = _linear_dev(r_matrix[rows], boot_centered)
            t_abs = np.abs(dev) / sd[rows, None]
            if simultaneous:
                max_t = np.maximum(max_t, t_abs.max(axis=0))
            if need_pvalues:
                counts[rows] = (
                    t_abs >= abs_t_obs[rows, None]
                ).sum(axis=1)
            if alpha is not None:
                crit[rows] = np.partition(t_abs, rank, axis=1)[:, rank]

        if simultaneous:
            if alpha is not None:
                crit_sim = np.partition(max_t, rank)[rank]
                crit = np.full(k_rows, crit_sim)
            if need_pvalues:
                counts_sim = np.zeros(k_rows, dtype=np.int64)
                for start in range(0, k_rows, chunk):
                    rows = slice(start, min(start + chunk, k_rows))
                    counts_sim[rows] = (
                        max_t[None, :] >= abs_t_obs[rows, None]
                    ).sum(axis=1)
                counts = counts_sim

        if need_pvalues:
            out["statistic"] = t_obs
            out["pvalue"] = (1.0 + counts) / (n_boot + 1.0)
        if alpha is not None:
            out["crit"] = crit
        return out

    # ------------------------------------------------------------------
    # t_test interface
    # ------------------------------------------------------------------
    def _t_test(self, results, r_matrix, q_matrix, simultaneous):
        """Build a ContrastResults for the calibration reference."""
        self._check_owner(results)
        r_matrix = np.asarray(r_matrix, dtype=np.float64)
        if r_matrix.ndim != 2 or r_matrix.shape[0] == 0:
            raise ValueError(
                "r_matrix must contain at least one constraint row"
            )
        if r_matrix.shape[1] != self.params.shape[0]:
            raise ValueError("r_matrix and params are not aligned")
        if not np.all(np.isfinite(r_matrix)):
            raise ValueError("r_matrix contains non-finite values")
        if np.any(np.all(r_matrix == 0.0, axis=1)):
            raise ValueError(
                "r_matrix contains an all-zero row; the corresponding "
                "direction is not estimable"
            )
        q = np.asarray(q_matrix, dtype=np.float64).squeeze()
        if q.ndim == 0:
            q = np.full(r_matrix.shape[0], float(q))
        else:
            q = q.ravel()
        if q.shape[0] != r_matrix.shape[0]:
            raise ValueError(
                "r_matrix and q_matrix must have the same number of rows"
            )
        if not np.all(np.isfinite(q)):
            raise ValueError("q_matrix contains non-finite values")

        out = self._infer(r_matrix, q, alpha=None, simultaneous=simultaneous,
                          need_pvalues=True)
        return ClusterBootstrapContrastResults(
            self, r_matrix, q, out, simultaneous
        )

    # ------------------------------------------------------------------
    # get_prediction interface
    # ------------------------------------------------------------------
    def _prediction_results(self, results, exog, predicted_mean, row_labels):
        """Build a PredictionResults using the calibration reference."""
        self._check_owner(results)
        design = np.ascontiguousarray(exog, dtype=np.float64)
        if design.ndim != 2 or design.shape[1] != self.params.shape[0]:
            raise ValueError(
                "exog does not align with the calibrated parameters"
            )
        out = self._infer(design, None, alpha=None, simultaneous=False,
                          need_pvalues=False)
        return ClusterBootstrapPredictionResults(
            self, design, np.asarray(predicted_mean, dtype=np.float64),
            out["sd"], row_labels
        )


class ClusterBootstrapContrastResults(ContrastResults):
    """
    Results of linear hypotheses calibrated by a cluster bootstrap.

    Created by ``t_test(..., inference=calibration)``.  The attributes
    mirror those of a classical ``t_test`` result, but ``sd``, ``pvalue``
    and :meth:`conf_int` are computed from the empirical bootstrap
    reference of the calibration instead of the t or normal
    distribution.
    """

    def __init__(self, calibration, r_matrix, q, infer_out, simultaneous):
        super().__init__()
        self._calibration = calibration
        self._r_matrix = r_matrix
        self._q = q
        self._simultaneous = simultaneous
        self.distribution = "cluster-bootstrap"
        self.effect = infer_out["effect"]
        self.sd = infer_out["sd"]
        self.tvalue = infer_out["statistic"]
        self.statistic = self.tvalue
        self.df_denom = calibration.n_clusters - 1
        self.pvalue = np.squeeze(infer_out["pvalue"])
        self.c_names = ["c%d" % ii for ii in range(len(self.effect))]

    def conf_int(self, alpha=0.05):
        """
        Confidence interval for the tested linear combinations.

        Parameters
        ----------
        alpha : float, optional
            The significance level.  The default ``alpha`` = .05 returns
            a 95% confidence interval.

        Returns
        -------
        ci : ndarray, (k_constraints, 2)
            Lower and upper limits.  If the ``t_test`` call used
            ``simultaneous=True`` these are joint (family-wise) limits,
            otherwise they are pointwise limits.
        """
        alpha = _check_alpha(alpha)
        out = self._calibration._infer(
            self._r_matrix, self._q, alpha=alpha,
            simultaneous=self._simultaneous, need_pvalues=False,
        )
        crit = out["crit"]
        lower = self.effect - crit * self.sd
        upper = self.effect + crit * self.sd
        return np.column_stack((lower, upper))

    def summary(self, xname=None, alpha=0.05, title=None):
        """Summarize the results of the hypothesis test."""
        if title is None:
            title = "Test for Constraints"
        elif title == "":
            title = None
        if xname is None:
            xname = self.c_names
        from statsmodels.iolib.summary import summary_params

        pvalues = np.atleast_1d(self.pvalue)
        summ = summary_params(
            (self, self.effect, self.sd, self.statistic, pvalues,
             self.conf_int(alpha)),
            yname="constraints", xname=xname, use_t=True, title=title,
            alpha=alpha,
        )
        return summ

    def summary_frame(self, xname=None, alpha=0.05):
        """Return the parameter table as a pandas DataFrame."""
        if xname is None:
            xname = self.c_names
        from statsmodels.iolib.summary import summary_params_frame

        summ = summary_params_frame(
            (self, self.effect, self.sd, self.statistic, self.pvalue,
             self.conf_int(alpha)),
            yname="constraints", xname=xname, use_t=True, alpha=alpha,
        )
        return summ


class ClusterBootstrapPredictionResults(PredictionResults):
    """
    Prediction results with cluster bootstrap calibrated intervals.

    Created by ``get_prediction(..., inference=calibration)``.  The
    predicted values are the conditional means of the original fit;
    ``mean_se`` and the ``mean_ci`` columns come from the empirical
    bootstrap reference.  Intervals for individual new observations
    (``obs=True``) are not available in this mode and the ``obs_ci``
    columns of :meth:`summary_frame` are NaN.
    """

    def __init__(self, calibration, design, predicted_mean, sd, row_labels):
        self._calibration = calibration
        self._design = design
        self.predicted = predicted_mean
        self.var_pred = sd ** 2
        self.var_resid = np.nan
        self.df = calibration.n_clusters - 1
        self.dist = None
        self.dist_args = ()
        self.row_labels = row_labels

    def conf_int(self, obs=False, alpha=0.05, simultaneous=False):
        """
        Confidence interval for the predicted conditional means.

        Parameters
        ----------
        obs : bool, optional
            Observation intervals (``obs=True``) are not available with
            cluster bootstrap inference and raise
            ``NotImplementedError``.
        alpha : float, optional
            The significance level.  The default ``alpha`` = .05 returns
            a 95% confidence interval.
        simultaneous : bool, optional
            If True, the interval is a joint confidence band for the
            conditional means of all prediction rows of this call,
            calibrated by the family-wise maximum statistic.  If False
            (default), pointwise intervals are returned.

        Returns
        -------
        ci : ndarray, (n_rows, 2)
            The lower and upper limits of the interval.
        """
        if obs:
            raise NotImplementedError(
                "intervals for individual new observations (obs=True) are "
                "not available with cluster bootstrap inference; only "
                "conditional mean intervals are supported"
            )
        alpha = _check_alpha(alpha)
        simultaneous = bool_like(simultaneous, "simultaneous", strict=True,
                                 optional=False)
        out = self._calibration._infer(
            self._design, None, alpha=alpha, simultaneous=simultaneous,
            need_pvalues=False,
        )
        crit = out["crit"]
        se = self.se_mean
        lower = self.predicted_mean - crit * se
        upper = self.predicted_mean + crit * se
        return np.column_stack((lower, upper))

    def summary_frame(self, alpha=0.05, simultaneous=False):
        """
        Summarize the predictions as a pandas DataFrame.

        Parameters
        ----------
        alpha : float, optional
            The significance level for the mean confidence intervals.
        simultaneous : bool, optional
            If True, ``mean_ci`` columns form a joint confidence band for
            the conditional means of all rows.  If False (default), they
            are pointwise intervals.

        Returns
        -------
        DataFrame
            Columns ``mean``, ``mean_se``, ``mean_ci_lower``,
            ``mean_ci_upper``, and the two ``obs_ci`` columns, which are
            NaN in this mode.
        """
        ci_mean = self.conf_int(obs=False, alpha=alpha,
                                simultaneous=simultaneous)
        n_rows = len(self.predicted_mean)
        to_include = {
            "mean": self.predicted_mean,
            "mean_se": self.se_mean,
            "mean_ci_lower": ci_mean[:, 0],
            "mean_ci_upper": ci_mean[:, 1],
            "obs_ci_lower": np.full(n_rows, np.nan),
            "obs_ci_upper": np.full(n_rows, np.nan),
        }
        self.table = to_include
        res = pd.DataFrame(to_include, index=self.row_labels,
                           columns=to_include.keys())
        return res
