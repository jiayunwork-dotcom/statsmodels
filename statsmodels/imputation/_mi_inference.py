"""
Finite-sample multiple-imputation inference.

This module contains the machinery shared by the MICE and MI analysis
chains for combining repeated OLS fits of the same analysis model under
Rubin's rules and carrying out finite-sample inference on linear
contrasts of the parameters.

Two pieces of statistical methodology are implemented:

1. Scalar contrasts (one linear restriction at a time, including each
   marginal coefficient) use the Barnard--Rubin adjusted degrees of
   freedom (Barnard & Rubin, 1999), which extends the classic Rubin
   (1987) combining rules by accounting for the finite residual
   degrees of freedom of the analysis model.  The point estimates and
   total (Rubin) covariance are unchanged from the normal-approximation
   implementation; only the reference distribution and its degrees of
   freedom change.

2. Joint tests of several linear restrictions use the Li--Raghunathan--
   Rubin (LRR; Li, Raghunathan & Rubin, 1991) Wald-type combining rule,
   often called the "D1" statistic in the multiple-imputation
   literature (van Buuren, 2018, *Flexible Imputation of Missing Data*,
   2nd ed., ch. 5).  The pooled (Rubin) covariance of the estimands is
   used to form a Wald quadratic form, and an LRR denominator degrees
   of freedom is computed from the relative increase in variance of the
   *joint* contrast.  For a rank-one restriction it reduces exactly to
   the scalar Barnard--Rubin result, so F = t**2 with the same p-value
   and denominator degrees of freedom.

Choice of joint method
----------------------
Two moment-based combining rules for joint hypotheses are discussed in
the cited literature:

* **D1 (implemented here).**  Pool the parameter estimates and their
  covariance with Rubin's rules, then form one Wald quadratic form on
  the pooled covariance of the contrast (Li, Raghunathan & Rubin,
  1991; van Buuren, 2018, sec. 5.4).  It needs only the per-imputation
  point estimates and covariance matrices -- the quantities already
  retained for scalar inference -- so no extra model fits or
  likelihood evaluations are required.  It assumes the completed-data
  estimator is approximately normal and that the average
  within-imputation covariance is positive definite on the effective
  constraint space.  The F calibration is a finite-sample approximation
  whose quality improves with the number of imputations; with few
  imputations the LRR denominator df becomes small, appropriately
  widening the test.  Because it reuses the pooled covariance, D1 is
  exactly consistent with the scalar Barnard--Rubin tests at rank one.

* **D2 (not implemented).**  Average the completed-data Wald
  statistics across imputations and compare the average with a
  chi-square/F reference with an adjusted variance (Li, Raghunathan &
  Rubin, 1991).  It only requires the per-imputation test statistics,
  not the full contrast covariance, but it discards the
  between-imputation covariance structure of the parameters, is known
  to be conservative and can be poorly calibrated for small m, and does
  not generally reduce to the scalar Rubin/Barnard--Rubin test at rank
  one.  Averaging p-values directly is not a valid special case of D2
  and is not used anywhere in this module.

D1 is chosen because it works directly from the same Rubin pooled
covariance as the scalar contrasts (guaranteeing the F = t**2 rank-one
identity), supports correlated restrictions and non-zero q through a
single covariance calculation, and exposes a clearly defined joint
relative increase in variance that drives the denominator degrees of
freedom.  Its extra cost over D2 is storing and inverting the
(small, contrast-sized) pooled covariance, which is negligible relative
to the analysis fits.

References
----------
Barnard, J. and Rubin, D.B. (1999).  Small-sample degrees of freedom
    with multiple imputation.  Biometrika 86(4), 948-955.
Li, K.-H., Raghunathan, T.E. and Rubin, D.B. (1991).  Large-sample
    significance levels from multiply imputed data using moment-based
    statistics and an F reference distribution.  JASA 86(416),
    1065-1073.
Rubin, D.B. (1987).  Multiple Imputation for Nonresponse in Surveys.
    Wiley.
van Buuren, S. (2018).  Flexible Imputation of Missing Data, 2nd ed.
    Chapman & Hall/CRC.
"""

import numpy as np
from scipy import stats

from statsmodels.stats.contrast import ContrastResults, WaldTestResults
from statsmodels.tools.validation import bool_like


# Default tolerance used when determining the numerical (effective)
# rank of a contrast matrix.  The singular values of R are compared
# against ``s1 * MI_RANK_TOL`` where ``s1`` is the largest singular
# value, so the tolerance is relative and scale free in the rows of R
# but still depends on the *absolute* scale of R: rescaling a row
# changes its singular values.  Contradictory (incompatible) rows are
# detected separately and always raise an error.
MI_RANK_TOL = 1e-10

# Relative threshold below which between-imputation variation is
# treated as numerically zero (compared with the within-imputation
# variance).  Exact-zero and floating-point-noise-only between
# covariance then give the same finite/complete-data degrees of
# freedom and never produce NaN.
MI_ZERO_VAR_TOL = 1e-14

# Method names and bibliographic references exposed on every small
# sample result object and in the diagnostic output.
SCALAR_METHOD = "Barnard-Rubin adjusted degrees of freedom"
JOINT_METHOD = "Li-Raghunathan-Rubin (LRR/D1) pooled Wald F"
MI_METHOD_REFERENCES = {
    "scalar": ("Barnard, J. and Rubin, D.B. (1999). Small-sample "
               "degrees of freedom with multiple imputation. "
               "Biometrika 86(4), 948-955."),
    "joint": ("Li, K.-H., Raghunathan, T.E. and Rubin, D.B. (1991). "
              "Large-sample significance levels from multiply imputed "
              "data using moment-based statistics and an F reference "
              "distribution. JASA 86(416), 1065-1073."),
    "pooling": ("Rubin, D.B. (1987). Multiple Imputation for "
                "Nonresponse in Surveys. Wiley."),
}


def _validate_inference_kwds(inference, df_complete, n_imputations):
    """Validate fit/combine keyword arguments before any imputation runs."""
    if inference not in ("legacy", "small_sample"):
        raise ValueError(
            "inference must be 'legacy' or 'small_sample', got %r"
            % (inference,))
    if not isinstance(n_imputations, (int, np.integer)) or isinstance(
            n_imputations, bool):
        raise ValueError("n_imputations must be an integer")
    if inference == "small_sample" and n_imputations < 2:
        raise ValueError(
            "small_sample inference requires at least two imputations "
            "(got %d)" % n_imputations)
    if df_complete is not None:
        if isinstance(df_complete, bool):
            raise ValueError(
                "df_complete must not be a bool; give a positive finite "
                "number or np.inf")
        df_complete = float(df_complete)
        if not np.isfinite(df_complete) and df_complete != np.inf:
            raise ValueError(
                "df_complete must be a positive finite number or np.inf")
        if df_complete <= 0:
            raise ValueError("df_complete must be positive")
    return df_complete


def _extract_fit(fit, index):
    """
    Pull the compact information needed for MI inference from one
    completed analysis fit.

    Only the parameter names, point estimates, covariance and residual
    degrees of freedom are retained -- never the full data set or the
    complete result object.  All fits must be non-robust, full column
    rank OLS fits with positive residual degrees of freedom and finite
    real inputs.
    """
    # Unwrap statsmodels wrapper objects without requiring the user to
    # keep any particular object alive through results_cb.
    res = getattr(fit, "_results", fit)

    model = getattr(res, "model", None)
    model_class = type(model)
    # Non-robust OLS only: robust covariance would invalidate the
    # residual-df based small sample calibration.
    from statsmodels.regression.linear_model import OLS
    if model is None or not isinstance(model, OLS):
        raise ValueError(
            "imputation %d: small_sample inference currently supports "
            "non-robust OLS analysis models only" % index)
    if getattr(res, "cov_type", "nonrobust") != "nonrobust":
        raise ValueError(
            "imputation %d: small_sample inference does not support "
            "robust covariance (cov_type=%r)"
            % (index, getattr(res, "cov_type", None)))

    params = np.asarray(getattr(res, "params", None), dtype=float)
    if params.ndim != 1:
        raise ValueError(
            "imputation %d: params must be a 1-d array" % index)
    if not np.all(np.isfinite(params)):
        raise ValueError(
            "imputation %d: params contain non-finite values" % index)

    cov = np.asarray(res.cov_params(), dtype=float)
    if cov.shape != (params.size, params.size):
        raise ValueError(
            "imputation %d: covariance has shape %s but there are %d "
            "parameters" % (index, cov.shape, params.size))
    if not np.all(np.isfinite(cov)):
        raise ValueError(
            "imputation %d: covariance contains non-finite values" % index)

    # Full column rank design and positive residual degrees of freedom.
    rank = int(getattr(model, "rank", -1))
    nobs = int(getattr(model, "nobs", -1))
    if rank < 0 or nobs < 0 or rank != params.size:
        raise ValueError(
            "imputation %d: analysis design must be full column rank "
            "(rank=%d, number of parameters=%d)"
            % (index, rank, params.size))
    df_resid = float(getattr(res, "df_resid", np.nan))
    if not np.isfinite(df_resid) or df_resid <= 0:
        raise ValueError(
            "imputation %d: residual degrees of freedom must be a "
            "positive finite number (got %r)" % (index, df_resid))

    names = list(model.exog_names)
    if len(names) != params.size or len(set(names)) != len(names):
        raise ValueError(
            "imputation %d: parameter names are missing or duplicated"
            % index)

    return {
        "names": names,
        "params": params,
        "cov": cov,
        "df_resid": df_resid,
        "model_class": model_class,
    }


class MIPooledInference:
    """
    Rubin pooling plus Barnard--Rubin finite-sample calibration.

    The object stores only pooled, aggregated quantities (and the
    per-imputation point estimates needed to rebuild between-imputation
    covariance).  It never retains the completed data sets or the
    individual regression result objects.

    Parameters
    ----------
    fits : list of dict
        Compact per-fit information produced by :func:`_extract_fit`.
    df_complete : float or np.inf
        Complete-data residual degrees of freedom used for every
        contrast.  When ``np.inf``, the normal/chi-square large-sample
        limit is returned.
    """

    def __init__(self, fits, df_complete):
        m = len(fits)
        if m < 2:
            raise ValueError(
                "small_sample inference requires at least two "
                "imputations (got %d)" % m)

        if df_complete is not None:
            if isinstance(df_complete, bool):
                raise ValueError(
                    "df_complete must not be a bool; give a positive "
                    "finite number or np.inf")
            df_complete = float(df_complete)
            if not np.isfinite(df_complete) and df_complete != np.inf:
                raise ValueError(
                    "df_complete must be a positive finite number or "
                    "np.inf")
            if df_complete <= 0:
                raise ValueError("df_complete must be positive")

        # Same analysis model class across imputations.
        model_classes = {f["model_class"] for f in fits}
        if len(model_classes) != 1:
            raise ValueError(
                "all imputations must fit the same analysis model, "
                "got %r" % [(c.__name__) for c in model_classes])
        self.model_class = fits[0]["model_class"]

        # Align parameters by name, not by vector position.
        ref_names = fits[0]["names"]
        if len(set(ref_names)) != len(ref_names):
            raise ValueError(
                "imputation 1: duplicated parameter names %r"
                % [n for n in ref_names if ref_names.count(n) > 1])
        ref_set = set(ref_names)
        aligned_params = []
        aligned_cov = []
        for j, f in enumerate(fits, start=1):
            names = f["names"]
            if len(set(names)) != len(names):
                raise ValueError(
                    "imputation %d: duplicated parameter names" % j)
            s = set(names)
            if s != ref_set:
                missing = ref_set - s
                extra = s - ref_set
                raise ValueError(
                    "imputation %d: parameter identity does not match "
                    "imputation 1 (missing=%r, extra=%r)"
                    % (j, sorted(missing), sorted(extra)))
            pos = [names.index(n) for n in ref_names]
            p = f["params"][pos]
            c = f["cov"][np.ix_(pos, pos)]
            if not np.all(np.isfinite(p)) or not np.all(np.isfinite(c)):
                raise ValueError(
                    "imputation %d: non-finite fit quantities" % j)
            aligned_params.append(p)
            aligned_cov.append(c)

        self.param_names = list(ref_names)
        self.k_params = len(ref_names)
        self.m = m
        params_arr = np.asarray(aligned_params)          # (m, k)
        cov_arr = np.asarray(aligned_cov)                # (m, k, k)

        # Residual df handling.
        df_resids = np.asarray([f["df_resid"] for f in fits], dtype=float)
        if df_complete is None:
            if not np.all(np.isfinite(df_resids)) or \
                    np.any(df_resids <= 0):
                raise ValueError(
                    "residual degrees of freedom are missing or invalid; "
                    "supply df_complete explicitly")
            if not np.allclose(df_resids, df_resids[0]):
                raise ValueError(
                    "residual degrees of freedom differ across "
                    "imputations %r; supply df_complete explicitly"
                    % (df_resids.tolist(),))
            self.df_complete = float(df_resids[0])
        else:
            self.df_complete = float(df_complete)
        self.df_resids = df_resids

        # Rubin's point estimate.
        self.params = params_arr.mean(axis=0)
        # Average within-imputation covariance.
        self.cov_within = cov_arr.mean(axis=0)
        # Between-imputation covariance (unbiased sample covariance,
        # ddof=1; np.cov is well defined for m >= 2, including m == 2).
        self.cov_between = np.atleast_2d(np.cov(params_arr.T, ddof=1))
        self.cov_between = 0.5 * (self.cov_between + self.cov_between.T)

        # Rubin total covariance.
        self.f_inflation = 1.0 + 1.0 / m
        self.cov_total = (self.cov_within
                          + self.f_inflation * self.cov_between)
        self.cov_total = 0.5 * (self.cov_total + self.cov_total.T)

        # Fraction of missing information (per parameter).
        diag_total = np.diag(self.cov_total)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.fmi = np.where(
                diag_total > 0,
                self.f_inflation * np.diag(self.cov_between) / diag_total,
                0.0)
        self.frac_miss_info = self.fmi

        # Diagnostics.
        self.method_scalar = SCALAR_METHOD
        self.method_joint = JOINT_METHOD
        self.references = dict(MI_METHOD_REFERENCES)
        self.rank_tol = MI_RANK_TOL

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _parse_contrast(self, r_matrix, q):
        """Resolve (R, q) against parameter names and validate shapes."""
        from patsy import DesignInfo

        if isinstance(r_matrix, tuple):
            r_matrix, q = r_matrix

        design = DesignInfo(self.param_names)
        if isinstance(r_matrix, str):
            lc = design.linear_constraint(r_matrix)
            R = np.asarray(lc.coefs, dtype=float)
            qv = np.asarray(lc.constants, dtype=float).reshape(-1)
        else:
            R = np.atleast_2d(np.asarray(r_matrix, dtype=float))
            if R.shape[1] != self.k_params:
                raise ValueError(
                    "constraint width %d does not match number of "
                    "parameters %d" % (R.shape[1], self.k_params))
            if q is None:
                qv = np.zeros(R.shape[0])
            else:
                qv = np.asarray(q, dtype=float).reshape(-1)
            if qv.shape[0] != R.shape[0]:
                raise ValueError(
                    "q must have the same number of rows as R")
        if not np.all(np.isfinite(R)) or not np.all(np.isfinite(qv)):
            raise ValueError("constraints must contain finite values")
        return R, qv

    def _effective_rank(self, R):
        """Effective rank of R using the public relative tolerance."""
        s = np.linalg.svd(R, compute_uv=False)
        if s.size == 0 or s[0] <= 0:
            return 0, s
        return int(np.count_nonzero(s > s[0] * self.rank_tol)), s

    def _check_within_pd(self, M):
        """
        Check that the (average within-imputation) covariance is
        positive definite on the effective constraint space.
        """
        vals = np.linalg.eigvalsh(0.5 * (M + M.T))
        tol = self.rank_tol * max(
            1.0, np.max(np.abs(vals)) if vals.size else 1.0)
        if np.any(vals <= tol):
            raise ValueError(
                "average within-imputation covariance is singular or "
                "not positive definite on the effective constraint "
                "space (smallest eigenvalue=%.3g, tolerance=%.3g); "
                "check for redundant/incompatible restrictions"
                % (vals[0], tol))

    def _barnard_rubin_df(self, r_iv):
        """
        Barnard--Rubin (1999) adjusted denominator df for one scalar
        contrast, given its relative increase in variance ``r_iv``.
        """
        dfcom = self.df_complete
        if r_iv <= MI_ZERO_VAR_TOL:
            # No (numerically meaningful) between-imputation variance
            # for this contrast: use the complete-data df directly.
            return dfcom if np.isfinite(dfcom) else np.inf
        df_old = (self.m - 1) * (1.0 + 1.0 / r_iv) ** 2
        if not np.isfinite(dfcom):
            # Large-sample limit: no complete-data df contribution.
            return df_old
        df_obs = dfcom * (dfcom + 1) / (dfcom + 3)
        df_br = df_old * df_obs / (df_old + df_obs)
        return df_br

    def _scalar_contrast(self, row, qval):
        """
        Rubin point estimate, total variance, relative increase in
        variance and Barnard--Rubin df for a single contrast row.
        """
        effect = float(row @ self.params - qval)
        u_bar = float(row @ self.cov_within @ row)
        b = float(row @ self.cov_between @ row)
        var_t = u_bar + self.f_inflation * b
        r_iv = (self.f_inflation * b / u_bar) if u_bar > 0 else 0.0
        if var_t <= 0:
            raise ValueError(
                "contrast has non-positive total variance; cannot "
                "perform a finite-sample test")
        df = self._barnard_rubin_df(r_iv)
        return effect, var_t, r_iv, df

    # ------------------------------------------------------------------
    # Scalar (marginal) t tests
    # ------------------------------------------------------------------
    def t_test(self, r_matrix, q=None, alpha=0.05):
        alpha = float(alpha)
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must lie strictly in (0, 1)")
        R, qv = self._parse_contrast(r_matrix, q)
        n_rows = R.shape[0]
        effects = np.empty(n_rows)
        sds = np.empty(n_rows)
        tvals = np.empty(n_rows)
        dfs = np.empty(n_rows)
        r_ivs = np.empty(n_rows)
        for i in range(n_rows):
            effect, var_t, r_iv, df = self._scalar_contrast(R[i], qv[i])
            effects[i] = effect
            sds[i] = np.sqrt(var_t)
            tvals[i] = effect / sds[i]
            dfs[i] = df
            r_ivs[i] = r_iv

        # ContrastResults evaluates t p-values and critical values
        # element-wise; scipy treats df=inf as the normal limit.
        res = ContrastResults(
            t=tvals, sd=sds, effect=effects,
            df_denom=np.squeeze(dfs), alpha=alpha)
        # Marker read by ContrastResults.summary_frame so that the
        # finite-sample per-contrast denominator df is exposed without
        # changing the default table for non-imputation models.
        res.mi_include_df_denom = True
        res._mi_diagnostics = {
            "method": SCALAR_METHOD,
            "references": MI_METHOD_REFERENCES["scalar"],
            "df_complete": self.df_complete,
            "n_imputations": self.m,
            "relative_increase_variance": np.squeeze(r_ivs),
        }
        return res

    # ------------------------------------------------------------------
    # Joint Wald / F tests (LRR / D1)
    # ------------------------------------------------------------------
    def wald_f_test(self, r_matrix, q=None, rank_tol=None):
        R, qv = self._parse_contrast(r_matrix, q)
        if rank_tol is not None:
            old_tol = self.rank_tol
            self.rank_tol = float(rank_tol)
        try:
            rank, sv = self._effective_rank(R)
            if rank == 0:
                raise ValueError(
                    "zero-rank constraint: nothing to test")
            if rank < R.shape[0]:
                # Distinguish redundant-but-compatible rows from
                # contradictory (incompatible) rows.  With the SVD
                # R = U S V', the left null space U[:, rank:] spans the
                # vectors z satisfying z' R = 0; the system R b = q is
                # solvable iff z' q = 0 for every such z.
                u, s, vt = np.linalg.svd(R, full_matrices=True)
                v_basis = vt[:rank]
                z_space = u[:, rank:]
                if z_space.size:
                    scale = float(np.max(np.abs(qv))) + float(
                        np.linalg.norm(qv)) + 1.0
                    if np.max(np.abs(z_space.T @ qv)) > \
                            self.rank_tol * scale:
                        raise ValueError(
                            "contradictory (incompatible) constraints: "
                            "the restriction rows are linearly "
                            "dependent but their target values q cannot "
                            "be satisfied simultaneously")
                # Redundant but compatible: re-express the hypothesis in
                # the effective (full rank) row space of R.
                R_eff = v_basis
                q_eff = v_basis @ np.linalg.pinv(R) @ qv
                R, qv = R_eff, q_eff
                rank, sv = self._effective_rank(R)

            # Pooled covariance of the joint contrast.
            u_bar = R @ self.cov_within @ R.T
            b_cov = R @ self.cov_between @ R.T
            total_cov = u_bar + self.f_inflation * b_cov
            total_cov = 0.5 * (total_cov + total_cov.T)
            self._check_within_pd(u_bar)

            # LRR relative increase in variance for the joint contrast:
            # r_bar = (1/k) * tr[(1+1/m) B_c U_c^{-1}].  For k = 1 this
            # is exactly the scalar Barnard--Rubin relative increase in
            # variance r = (1+1/m) * (row B row') / (row U row').
            u_inv = np.linalg.inv(u_bar)
            r_joint = float(
                np.trace(self.f_inflation * b_cov @ u_inv) / rank)

            d = R @ self.params - qv
            wald = float(d @ np.linalg.solve(total_cov, d))
            f_stat = wald / rank

            # Li--Raghunathan--Rubin denominator degrees of freedom
            # (Li, Raghunathan & Rubin 1991; van Buuren 2018, eq.
            # 5.21).  For rank == 1 this is identical to the
            # Barnard--Rubin scalar adjusted df.
            dfcom = self.df_complete
            kk = rank
            km1 = self.m - 1
            if not np.isfinite(dfcom):
                # Large-sample limit: no complete-data df contribution.
                if r_joint <= MI_ZERO_VAR_TOL:
                    df_denom = np.inf
                else:
                    a = kk * km1
                    if a <= 4:
                        # The LRR formula assumes k(m-1) > 4; fall back
                        # to the direct (m-1)(1+1/r)^2 form which is
                        # well defined for m >= 2 and coincides with it
                        # when k = 1.
                        df_denom = km1 * (1.0 + 1.0 / r_joint) ** 2
                    else:
                        df_denom = 4.0 + (a - 4.0) * (
                            1.0 + (1.0 - 2.0 / a) / r_joint) ** 2
            else:
                if r_joint <= MI_ZERO_VAR_TOL:
                    # No (numerically meaningful) between-imputation
                    # variance on this contrast: use the complete-data df.
                    df_denom = dfcom
                else:
                    a = kk * km1
                    if a <= 4:
                        df_lrr = km1 * (1.0 + 1.0 / r_joint) ** 2
                    else:
                        df_lrr = 4.0 + (a - 4.0) * (
                            1.0 + (1.0 - 2.0 / a) / r_joint) ** 2
                    # Barnard--Rubin observed-df adjustment.
                    df_obs = dfcom * (dfcom + 1) / (dfcom + 3)
                    df_denom = df_lrr * df_obs / (df_lrr + df_obs)

            if np.isfinite(df_denom):
                p_value = float(stats.f.sf(f_stat, rank, df_denom))
            else:
                # Large-sample limit: rank * F ~ chi2(rank).
                p_value = float(stats.chi2.sf(wald, rank))

            res = ContrastResults(F=np.array([[f_stat]]),
                                  df_denom=df_denom, df_num=rank)
            res.pvalue = p_value
            res.fvalue = float(np.squeeze(f_stat))
            res.statistic = res.fvalue
            res.mi_wald = wald
            res.relative_increase_variance = r_joint
            res.rank_tol = self.rank_tol
            res.singular_values = sv
            res._mi_diagnostics = {
                "method": JOINT_METHOD,
                "references": MI_METHOD_REFERENCES["joint"],
                "df_complete": self.df_complete,
                "n_imputations": self.m,
                "effective_rank": rank,
                "n_rows_input": R.shape[0],
                "relative_increase_variance": r_joint,
                "rank_tol": self.rank_tol,
                "wald": wald,
            }
            return res
        finally:
            if rank_tol is not None:
                self.rank_tol = old_tol


class _MIResultData:
    """Minimal data holder exposing parameter-name metadata.

    The pooled result no longer owns any individual completed data set,
    but the linear-contrast machinery resolves parameter names through
    ``model.data.cov_names`` / ``param_names``.  This lightweight object
    supplies that metadata without retaining any observations.
    """

    def __init__(self, param_names, design_info=None):
        self.param_names = list(param_names)
        self.cov_names = list(param_names)
        self.xnames = list(param_names)
        self.design_info = design_info


class SmallSampleMIResultsMixin:
    """
    Inference overrides for pooled MI results in ``small_sample`` mode.

    The class holding this mixin must expose a :class:`MIPooledInference`
    as ``self._mi_inference``.  Legacy (normal approximation) results do
    not use this mixin and keep their original behaviour unchanged.
    """

    def _attach_name_data(self, design_info=None):
        """Attach pooled parameter-name metadata to the result model."""
        names = self._mi_inference.param_names
        if design_info is None:
            # Reuse the design information of any completed fit when
            # available (formula models), without retaining the data.
            design_info = getattr(self.model.data, "design_info", None)
        self._mi_data = _MIResultData(names, design_info=design_info)
        # Keep the conventional attribute in sync with the pooled names.
        self.exog_names = list(names)

    # Small sample inference is always t/F based.
    @property
    def use_t(self):
        return True

    @use_t.setter
    def use_t(self, value):
        if value is False:
            raise ValueError(
                "use_t=False is not supported in small_sample "
                "inference mode")
        self._use_t = True

    @property
    def mi_inference(self):
        """The underlying :class:`MIPooledInference` object."""
        return self._mi_inference

    def _check_inference_args(self, cov_p, invcov, df_constraints, use_t):
        if cov_p is not None:
            raise ValueError(
                "supplying cov_p is not supported in small_sample "
                "inference mode: it would bypass the between-"
                "imputation variance")
        if invcov is not None:
            raise ValueError(
                "supplying a custom inverse covariance (invcov) is not "
                "supported in small_sample inference mode")
        if df_constraints is not None:
            raise ValueError(
                "overriding the denominator degrees of freedom is not "
                "supported in small_sample inference mode")
        if use_t is not None:
            use_t = bool_like(use_t, "use_t", strict=True, optional=True)
            if not use_t:
                raise ValueError(
                    "use_t=False is not supported in small_sample "
                    "inference mode")

    @staticmethod
    def _check_alpha(alpha):
        alpha = float(alpha)
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must lie strictly in (0, 1)")
        return alpha

    # ------------------------------------------------------------------
    # Scalar inference aligned with unit-contrast t tests
    # ------------------------------------------------------------------
    def t_test(self, r_matrix, cov_p=None, use_t=None, alpha=0.05):
        self._check_inference_args(cov_p, None, None, use_t)
        self._check_alpha(alpha)
        return self._mi_inference.t_test(r_matrix, alpha=alpha)

    def conf_int(self, alpha=0.05, cols=None):
        alpha = self._check_alpha(alpha)
        # One unit contrast per parameter, built by name position.
        k = self._mi_inference.k_params
        R = np.eye(k)
        tt = self._mi_inference.t_test(R, alpha=alpha)
        ci = tt.conf_int(alpha)
        if cols is not None:
            import warnings
            warnings.warn(
                "cols is deprecated; slice the returned confidence "
                "interval array instead", FutureWarning)
            ci = ci[np.asarray(cols)]
        return ci

    @property
    def df_denom_params(self):
        """Barnard--Rubin denominator df used for each parameter."""
        k = self._mi_inference.k_params
        dfs = np.empty(k)
        for i, row in enumerate(np.eye(k)):
            _, _, _, dfs[i] = self._mi_inference._scalar_contrast(row, 0.0)
        return dfs

    @property
    def pvalues(self):
        k = self._mi_inference.k_params
        tt = self._mi_inference.t_test(np.eye(k))
        return np.asarray(tt.pvalue).reshape(-1)

    @property
    def tvalues(self):
        return self.params / self.bse

    # ------------------------------------------------------------------
    # Joint inference
    # ------------------------------------------------------------------
    def wald_test(self, r_matrix, cov_p=None, invcov=None, use_f=None,
                  df_constraints=None, scalar=None):
        self._check_inference_args(cov_p, invcov, df_constraints, None)
        if use_f is not None:
            use_f = bool_like(use_f, "use_f", strict=True, optional=True)
            if not use_f:
                raise ValueError(
                    "chi-square (use_f=False) Wald tests are not "
                    "supported in small_sample inference mode")
        res = self._mi_inference.wald_f_test(r_matrix)
        if scalar is not None:
            bool_like(scalar, "scalar", strict=True, optional=True)
        return res

    def f_test(self, r_matrix, cov_p=None, invcov=None):
        self._check_inference_args(cov_p, invcov, None, None)
        return self._mi_inference.wald_f_test(r_matrix)

    # ------------------------------------------------------------------
    # Term-level tests, reusing the formula design information
    # ------------------------------------------------------------------
    def wald_test_terms(self, skip_single=False, extra_constraints=None,
                        combine_terms=None, scalar=None):
        from collections import defaultdict
        from pandas import DataFrame

        if combine_terms is None:
            combine_terms = []
        if extra_constraints is None:
            extra_constraints = []
        design_info = getattr(self._mi_data, "design_info", None)
        identity = np.eye(len(self.params))
        constraints = []
        combined = defaultdict(list)

        if design_info is not None:
            for term in design_info.terms:
                cols = design_info.slice(term)
                name = term.name()
                cmat = identity[cols]
                for cname in combine_terms:
                    if cname in name:
                        combined[cname].append(cmat)
                if skip_single and cmat.shape[0] == 1:
                    continue
                constraints.append((name, cmat))
            combined_constraints = [
                (cname, np.vstack(combined[cname]))
                for cname in combine_terms]
        else:
            for col, name in enumerate(self._mi_data.param_names):
                cmat = np.atleast_2d(identity[col])
                for cname in combine_terms:
                    if cname in name:
                        combined[cname].append(cmat)
                if not skip_single:
                    constraints.append((name, cmat))
            combined_constraints = [
                (cname, np.vstack(combined[cname]))
                for cname in combine_terms]

        rows = []
        index = []
        for name, cmat in (constraints + combined_constraints
                           + list(extra_constraints)):
            wt = self._mi_inference.wald_f_test(cmat)
            rows.append([np.asarray(wt.statistic).squeeze().item(),
                         np.asarray(wt.pvalue).squeeze().item(),
                         int(wt.df_num), float(wt.df_denom)])
            index.append(name)

        table = DataFrame(
            rows, index=index,
            columns=["statistic", "pvalue", "df_constraint", "df_denom"])
        res = WaldTestResults(None, "F", None, table=table)
        return res

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def _small_sample_summary(self, top_info, alpha=0.05):
        from statsmodels.iolib import summary2

        alpha = self._check_alpha(alpha)
        smry = summary2.Summary()
        float_format = "%8.3f"
        top_info = dict(top_info)
        top_info["Inference:"] = "finite sample (t / F)"
        top_info["Scalar method:"] = SCALAR_METHOD
        top_info["Joint method:"] = JOINT_METHOD
        smry.add_dict(top_info, align="l", float_format=float_format)

        k = self._mi_inference.k_params
        R = np.eye(k)
        tt = self._mi_inference.t_test(R, alpha=alpha)
        frame = tt.summary_frame(xname=self._mi_inference.param_names,
                                 alpha=alpha)
        frame["df_denom"] = self.df_denom_params
        if hasattr(self, "frac_miss_info"):
            frame["FMI"] = self.frac_miss_info
        smry.add_df(frame, float_format=float_format)
        smry.add_title(results=self)
        return smry
