import numpy as np
import pandas as pd
from scipy.stats.distributions import chi2, norm
from statsmodels.graphics import utils


# Column name used for the failure-free (no event of any cause)
# probability in query results; leading underscores make it unlikely to
# collide with a user supplied cause label.
FAILURE_FREE_LABEL = "__failure_free__"


def _calc_survfunc_right(time, status, weights=None, entry=None, compress=True,
                         retall=True):
    """
    Calculate the survival function and its standard error for a single
    group.
    """

    # Convert the unique times to ranks (0, 1, 2, ...)
    if entry is None:
        utime, rtime = np.unique(time, return_inverse=True)
    else:
        tx = np.concatenate((time, entry))
        utime, rtime = np.unique(tx, return_inverse=True)
        rtime = rtime[0:len(time)]

    # Number of deaths at each unique time.
    ml = len(utime)
    if weights is None:
        d = np.bincount(rtime, weights=status, minlength=ml)
    else:
        d = np.bincount(rtime, weights=status*weights, minlength=ml)

    # Size of risk set just prior to each event time.
    if weights is None:
        n = np.bincount(rtime, minlength=ml)
    else:
        n = np.bincount(rtime, weights=weights, minlength=ml)
    if entry is not None:
        n = np.cumsum(n) - n
        rentry = np.searchsorted(utime, entry, side='left')
        if weights is None:
            n0 = np.bincount(rentry, minlength=ml)
        else:
            n0 = np.bincount(rentry, weights=weights, minlength=ml)
        n0 = np.cumsum(n0) - n0
        n = n0 - n
    else:
        n = np.cumsum(n[::-1])[::-1]

    # Only retain times where an event occurred.
    if compress:
        ii = np.flatnonzero(d > 0)
        d = d[ii]
        n = n[ii]
        utime = utime[ii]

    # The survival function probabilities.
    sp = 1 - d / n.astype(np.float64)
    ii = sp < 1e-16
    sp[ii] = 1e-16
    sp = np.log(sp)
    sp = np.cumsum(sp)
    sp = np.exp(sp)
    sp[ii] = 0

    if not retall:
        return sp, utime, rtime, n, d

    # Standard errors
    if weights is None:
        # Greenwood's formula
        denom = n * (n - d)
        denom = np.clip(denom, 1e-12, np.inf)
        se = d / denom.astype(np.float64)
        se[(n == d) | (n == 0)] = np.nan
        se = np.cumsum(se)
        se = np.sqrt(se)
        locs = np.isfinite(se) | (sp != 0)
        se[locs] *= sp[locs]
        se[~locs] = np.nan
    else:
        # Tsiatis' (1981) formula
        se = d / (n * n).astype(np.float64)
        se = np.cumsum(se)
        se = np.sqrt(se)

    return sp, se, utime, rtime, n, d


def _calc_incidence_right(time, status, weights=None):
    """
    Calculate the cumulative incidence function and its standard error.
    """

    # Calculate the all-cause survival function.
    status0 = (status >= 1).astype(np.float64)
    sp, utime, rtime, n, d = _calc_survfunc_right(time, status0, weights,
                                                  compress=False, retall=False)

    ngrp = int(status.max())

    # Number of cause-specific deaths at each unique time.
    d = []
    for k in range(ngrp):
        status0 = (status == k + 1).astype(np.float64)
        if weights is None:
            d0 = np.bincount(rtime, weights=status0, minlength=len(utime))
        else:
            d0 = np.bincount(rtime, weights=status0*weights,
                             minlength=len(utime))
        d.append(d0)

    # The cumulative incidence function probabilities.
    ip = []
    sp0 = np.r_[1, sp[:-1]] / n
    for k in range(ngrp):
        ip0 = np.cumsum(sp0 * d[k])
        ip.append(ip0)

    # The standard error of the cumulative incidence function.
    if weights is not None:
        return ip, None, utime
    se = []
    da = sum(d)
    for k in range(ngrp):

        ra = da / (n * (n - da))
        v = ip[k]**2 * np.cumsum(ra)
        v -= 2 * ip[k] * np.cumsum(ip[k] * ra)
        v += np.cumsum(ip[k]**2 * ra)

        ra = (n - d[k]) * d[k] / n
        v += np.cumsum(sp0**2 * ra)

        ra = sp0 * d[k] / n
        v -= 2 * ip[k] * np.cumsum(ra)
        v += 2 * np.cumsum(ip[k] * ra)

        se.append(np.sqrt(v))

    return ip, se, utime


def _checkargs(time, status, entry, freq_weights, exog):

    if len(time) != len(status):
        raise ValueError("time and status must have the same length")

    if entry is not None and (len(entry) != len(time)):
        msg = "entry times and event times must have the same length"
        raise ValueError(msg)

    if entry is not None and np.any(entry >= time):
        msg = "Entry times must not occur on or after event times"
        raise ValueError(msg)

    if freq_weights is not None and (len(freq_weights) != len(time)):
        raise ValueError("weights, time and status must have the same length")

    if exog is not None and (exog.shape[0] != len(time)):
        raise ValueError("the rows of exog should align with time")


class CumIncidenceRight:
    """
    Estimation and inference for a cumulative incidence function.

    If J = 1, 2, ... indicates the event type, the cumulative
    incidence function for cause j is:

    I(t, j) = P(T <= t and J=j)

    Only right censoring is supported.  If frequency weights are provided,
    the point estimate is returned without a standard error.

    Two construction modes are supported:

    * The usual empirical Aalen-Johansen estimator, obtained by calling
      the class directly with raw ``time`` and ``status`` arrays.
    * A model-based estimator combining cause-specific Cox
      (proportional hazards) fits, obtained through
      :meth:`from_phreg`.  Each cause is fit with
      :class:`statsmodels.duration.hazard_regression.PHReg` and the
      cause-specific baseline cumulative hazards are combined into a
      cumulative incidence curve on a shared time grid.

    Parameters
    ----------
    time : array_like
        An array of times (censoring times or event times)
    status : array_like
        If status >= 1 indicates which event occurred at time t.  If
        status = 0, the subject was censored at time t.
    title : str
        Optional title used for plots and summary output.
    freq_weights : array_like
        Optional frequency weights
    exog : array_like
        Optional, if present used to account for violation of
        independent censoring.
    bw_factor : float
        Band-width multiplier for kernel-based estimation.  Only
        used if exog is provided.
    dimred : bool
        If True, proportional hazards regression models are used to
        reduce exog to two columns by predicting overall events and
        censoring in two separate models.  If False, exog is used
        directly for calculating kernel weights without dimension
        reduction.

    Attributes
    ----------
    times : array_like
        The distinct times at which the incidence rates are estimated
    cinc : list of arrays
        cinc[k-1] contains the estimated cumulative incidence rates
        for outcome k=1,2,...
    cinc_se : list of arrays
        The standard errors for the values in `cinc`.  Not available when
        exog and/or frequency weights are provided.

    Notes
    -----
    When exog is provided, a local estimate of the cumulative incidence
    rate around each point is provided, and these are averaged to
    produce an estimate of the marginal cumulative incidence
    functions.  The procedure is analogous to that described in Zeng
    (2004) for estimation of the marginal survival function.  The
    approach removes bias resulting from dependent censoring when the
    censoring becomes independent conditioned on the columns of exog.

    References
    ----------
    The Stata stcompet procedure:
        http://www.stata-journal.com/sjpdf.html?articlenum=st0059

    Dinse, G. E. and M. G. Larson. 1986. A note on semi-Markov models
    for partially censored data. Biometrika 73: 379-386.

    Marubini, E. and M. G. Valsecchi. 1995. Analysing Survival Data
    from Clinical Trials and Observational Studies. Chichester, UK:
    John Wiley & Sons.

    D. Zeng (2004).  Estimating marginal survival function by
    adjusting for dependent censoring using many covariates.  Annals
    of Statistics 32:4.
    https://arxiv.org/pdf/math/0409180.pdf
    """

    def __init__(self, time, status, title=None, freq_weights=None,
                 exog=None, bw_factor=1., dimred=True):

        _checkargs(time, status, None, freq_weights, None)
        time = self.time = np.asarray(time)
        status = self.status = np.asarray(status)
        if freq_weights is not None:
            freq_weights = self.freq_weights = np.asarray(freq_weights)

        if exog is not None:
            from ._kernel_estimates import _kernel_cumincidence
            exog = self.exog = np.asarray(exog)
            nobs = exog.shape[0]
            kw = nobs**(-1/3.0) * bw_factor
            kfunc = lambda x: np.exp(-x**2 / kw**2).sum(1)
            x = _kernel_cumincidence(time, status, exog, kfunc, freq_weights,
                                     dimred)
            self.times = x[0]
            self.cinc = x[1]
            self._fitted = False
            return

        x = _calc_incidence_right(time, status, freq_weights)
        self.cinc = x[0]
        self.cinc_se = x[1]
        self.times = x[2]
        self.title = "" if not title else title
        self._fitted = False

        # Precompute the all-cause (Kaplan-Meier) survival step function
        # needed for interval queries.  Only the unweighted, unadjusted
        # estimator supports queries.
        if freq_weights is None and exog is None:
            status0 = (np.asarray(status) >= 1).astype(np.float64)
            sp, _, utime, _, n_risk, n_event = \
                _calc_survfunc_right(time, status0, weights=None,
                                     entry=None, compress=True, retall=True)
            self._emp_times = np.asarray(utime, dtype=np.float64)
            # Post-event survival at each event time.
            self._emp_surv = np.asarray(sp, dtype=np.float64)
            self._emp_n_risk = np.asarray(n_risk, dtype=np.float64)
            self._emp_n_event = np.asarray(n_event, dtype=np.float64)
            # Cause-specific event counts at each compressed event time.
            d_list = []
            event_pos = np.searchsorted(utime, np.asarray(time),
                                        side="left")
            n_grid = len(utime)
            for k in range(len(self.cinc)):
                ind = (np.asarray(status) == k + 1).astype(np.float64)
                d0 = np.bincount(event_pos, weights=ind,
                                 minlength=n_grid)[:n_grid]
                d_list.append(np.asarray(d0, dtype=np.float64))
            self._emp_cause_counts = d_list

    # ------------------------------------------------------------------
    # Fitted (PHReg based) construction
    # ------------------------------------------------------------------

    @classmethod
    def from_phreg(cls, fits):
        """
        Combine fitted cause-specific PHReg models into a cumulative
        incidence estimator.

        Each fit is an unregularized
        :class:`statsmodels.duration.hazard_regression.PHRegResults`
        (or a results wrapper around one) for one failure cause.  All
        fits must originate from the same analysis table: the retained
        analysis rows must agree, in position, on exit time, entry
        time, strata, numeric design matrix and offset.  A row may be
        an event in at most one model; in every other model the same
        row must participate at the same exit time with status zero
        (right censored).

        Parameters
        ----------
        fits : dict
            A mapping ``{cause_label: fitted_phreg_results}``.  The
            labels (keys) identify the causes in query output.  A
            mapping with a single entry is allowed and gives the
            single-outcome (no competing risk) special case.

        Returns
        -------
        CumIncidenceRight
            A fitted cumulative incidence object.

        Notes
        -----
        Efron and Breslow tie handling are both supported, as are
        delayed entry (left truncation), stratification and offsets.
        The baseline hazard jumps used by the combined estimator are
        the Nelson-Aalen jumps of each fit, computed from the fit's
        risk sets and fitting offset.
        """

        self = cls.__new__(cls)
        self._fitted = True
        self.title = ""
        self.cinc_se = None

        if not isinstance(fits, dict) or len(fits) == 0:
            raise ValueError("`fits` must be a non-empty dict mapping "
                             "cause labels to fitted PHRegResults")

        labels = list(fits.keys())
        if len(set(labels)) != len(labels):
            raise ValueError("duplicate cause labels in `fits`")

        unwrapped = []
        for label in labels:
            result = fits[label]
            # Results wrappers store the underlying results object in
            # ``_results``; plain results are used directly.
            inner = getattr(result, "_results", result)
            if not hasattr(inner, "model") or \
                    not hasattr(inner.model, "baseline_cumulative_hazard"):
                raise ValueError("each value in `fits` must be a fitted "
                                 "PHRegResults object (or a wrapper of "
                                 "one); got %r for label %r" %
                                 (type(result), label))
            regularized = \
                "Regularized" in type(result).__name__ or \
                "Regularized" in type(inner).__name__ or \
                bool(getattr(inner, "regularized", False))
            if regularized:
                raise ValueError("regularized PHReg fits are not "
                                 "supported; refit without regularization "
                                 "(label %r)" % label)
            unwrapped.append(inner)

        self.labels = list(labels)
        self._validate_cohort(unwrapped)

        ref = unwrapped[0].model
        self._nobs = len(ref.endog)

        # Each cause-specific fit has its own coefficient vector.
        # Keep them in label order; prediction multiplies cause k's
        # baseline cumulative hazard by the hazard ratio from fit k.
        ref_p = np.asarray(unwrapped[0].params, dtype=np.float64)
        for pos, result in enumerate(unwrapped[1:], start=1):
            if np.asarray(result.params).shape != ref_p.shape:
                raise ValueError("fits do not have parameters of the "
                                 "same length for cause label %r; the "
                                 "design matrices must be identical"
                                 % self.labels[pos])
        self._params = np.vstack(
            [np.asarray(r.params, dtype=np.float64) for r in unwrapped])

        # Union of strata that survive in at least one cause model.
        # Strata without any event of a given cause contribute no jumps
        # for that cause (its hazard is identically zero there).
        name_sets = [set(m.surv.stratum_names) for m in
                     (r.model for r in unwrapped)]
        union_names = []
        for names in name_sets:
            for nm in names:
                if nm not in union_names:
                    union_names.append(nm)
        # Preserve the reference model's ordering where possible.
        ordered = [nm for nm in ref.surv.stratum_names
                   if nm in union_names]
        ordered += [nm for nm in union_names if nm not in ordered]
        self._stratum_names = ordered

        # Per stratum: the merged event-time grid, every cause's
        # post-event baseline cumulative hazard aligned onto it, and
        # the cause-specific hazard jumps placed only at that cause's
        # own event times (interpolating cumulative hazards onto the
        # merged grid before differencing would smear jumps onto other
        # causes' event times).
        grid_per_stratum = {}
        ch_per_stratum = {}
        jump_per_stratum = {}
        n_causes = len(unwrapped)
        for st_name in self._stratum_names:
            grids, chs, jumps_raw = [], [], []
            for result in unwrapped:
                model = result.model
                if st_name in model.surv.stratum_names:
                    st_pos = list(model.surv.stratum_names).index(st_name)
                    base = model.baseline_cumulative_hazard(
                        result.params, post_event=True)
                    t_h = np.asarray(base[st_pos][0], dtype=np.float64)
                    h_h = np.asarray(base[st_pos][1], dtype=np.float64)
                    j_h = np.diff(np.r_[0.0, h_h])
                else:
                    # No event of this cause in this stratum.
                    t_h = np.empty(0, dtype=np.float64)
                    h_h = np.empty(0, dtype=np.float64)
                    j_h = np.empty(0, dtype=np.float64)
                grids.append(t_h)
                chs.append(h_h)
                jumps_raw.append(j_h)
            non_empty = [g for g in grids if g.size > 0]
            if not non_empty:
                # A stratum must have events in at least one cause,
                # otherwise it would have been dropped by every fit.
                continue
            merged = np.unique(np.concatenate(non_empty))
            grid_per_stratum[st_name] = merged
            m = len(merged)
            aligned_ch = np.zeros((n_causes, m), dtype=np.float64)
            aligned_jump = np.zeros((n_causes, m), dtype=np.float64)
            for k in range(n_causes):
                if grids[k].size == 0:
                    continue  # this cause has zero risk in the stratum
                aligned_ch[k] = np.interp(merged, grids[k], chs[k],
                                          left=0.0, right=chs[k][-1])
                pos = np.searchsorted(merged, grids[k])
                aligned_jump[k, pos] = jumps_raw[k]
            ch_per_stratum[st_name] = aligned_ch
            jump_per_stratum[st_name] = aligned_jump

        self._grid = grid_per_stratum
        self._cumhaz = ch_per_stratum
        self._jumps = jump_per_stratum
        self.times = next(iter(grid_per_stratum.values()))
        self.cinc = None
        return self

    def _validate_cohort(self, results):
        """Check that all fits describe the same rows in position."""

        ref = results[0].model
        ref_endog = np.asarray(ref.endog, dtype=np.float64)
        ref_exog = np.asarray(ref.exog, dtype=np.float64)
        ref_entry = None if ref.entry is None else \
            np.asarray(ref.entry, dtype=np.float64)
        ref_strata = None if ref.strata is None else \
            np.asarray(ref.strata)
        ref_offset = None if ref.offset is None else \
            np.asarray(ref.offset, dtype=np.float64)

        event_sum = np.zeros(len(ref_endog), dtype=np.float64)

        for pos, result in enumerate(results):
            mod = result.model
            endog = np.asarray(mod.endog, dtype=np.float64)
            exog = np.asarray(mod.exog, dtype=np.float64)
            entry = None if mod.entry is None else \
                np.asarray(mod.entry, dtype=np.float64)
            strata = None if mod.strata is None else \
                np.asarray(mod.strata)
            offset = None if mod.offset is None else \
                np.asarray(mod.offset, dtype=np.float64)

            if endog.shape != ref_endog.shape:
                raise ValueError("fits do not share the same analysis "
                                 "table: exit time shape mismatch for "
                                 "cause label %r" % self.labels[pos])
            if not np.array_equal(endog, ref_endog):
                raise ValueError("fits do not share the same exit times "
                                 "by row position for cause label %r"
                                 % self.labels[pos])
            if exog.shape != ref_exog.shape:
                raise ValueError("fits do not share the same numeric "
                                 "design matrix shape for cause label %r"
                                 % self.labels[pos])
            if not np.allclose(exog, ref_exog, rtol=1e-12, atol=1e-12):
                raise ValueError("fits do not share the same numeric "
                                 "design matrix values for cause label "
                                 "%r" % self.labels[pos])
            if (entry is None) != (ref_entry is None) or \
                    (entry is not None and
                     not np.array_equal(entry, ref_entry)):
                raise ValueError("fits do not share the same entry "
                                 "(delayed enrollment) times for cause "
                                 "label %r" % self.labels[pos])
            if (strata is None) != (ref_strata is None) or \
                    (strata is not None and
                     not np.array_equal(strata, ref_strata)):
                raise ValueError("fits do not share the same strata "
                                 "values by row position for cause "
                                 "label %r" % self.labels[pos])
            if (offset is None) != (ref_offset is None) or \
                    (offset is not None and
                     not np.allclose(offset, ref_offset, rtol=1e-12,
                                     atol=1e-12)):
                raise ValueError("fits do not share the same fitting "
                                 "offset values for cause label %r"
                                 % self.labels[pos])

            status = np.asarray(mod.status, dtype=np.float64)
            if status.shape != ref_endog.shape:
                raise ValueError("status vector length mismatch for "
                                 "cause label %r" % self.labels[pos])
            if np.any((status != 0) & (status != 1)):
                raise ValueError("PHReg status must be 0/1; cause label "
                                 "%r has other values" % self.labels[pos])
            event_sum += status

        if np.any(event_sum > 1):
            raise ValueError("at least one analysis row is marked as an "
                             "event in more than one cause-specific fit; "
                             "each row may be an event in at most one "
                             "model")

    # ------------------------------------------------------------------
    # Input validation shared by both modes
    # ------------------------------------------------------------------

    def _validate_query_base(self, origin, times):
        origin_arr = np.asarray(origin, dtype=np.float64)
        if origin_arr.ndim != 0 or not np.isfinite(origin_arr) \
                or origin_arr < 0:
            raise ValueError("`origin` must be a finite non-negative "
                             "scalar")
        t = np.asarray(times, dtype=np.float64)
        if t.ndim != 1 or t.size == 0:
            raise ValueError("`times` must be a non-empty 1-d array")
        if not np.all(np.isfinite(t)):
            raise ValueError("all query times must be finite")
        if np.any(t < float(origin_arr)):
            raise ValueError("query times must be greater than or equal "
                             "to the origin")
        if np.any(np.diff(t) < 0):
            raise ValueError("query times must be non-decreasing")
        return float(origin_arr), t

    @staticmethod
    def _surv_step(times_grid, surv_at_events, query):
        # Right-continuous post-event survival step: constant on
        # [event_k, event_{k+1}), equal to the post-event value at each
        # event, 1 before the first event, final value carried forward.
        ix = np.searchsorted(times_grid, query, side="right") - 1
        out = np.ones(len(query), dtype=np.float64)
        ii = ix >= 0
        out[ii] = surv_at_events[ix[ii]]
        return out

    # ------------------------------------------------------------------
    # Empirical mode query
    # ------------------------------------------------------------------

    def _predict_empirical(self, origin, times):
        if hasattr(self, "freq_weights") or hasattr(self, "exog"):
            raise ValueError("interval queries are only supported for "
                             "the unweighted estimator without exog "
                             "adjustment")

        s_origin = self._surv_step(self._emp_times, self._emp_surv,
                                   np.array([origin]))[0]
        if s_origin <= 0.0:
            raise ValueError("no survivable probability mass remains at "
                             "the origin; conditional probabilities are "
                             "not defined")

        s_t = self._surv_step(self._emp_times, self._emp_surv, times)
        # Probability mass of an event in (origin, t] obtained from the
        # increments of the survival curve.
        fail_mass = s_origin - s_t

        cause_cif = []
        for d_k in self._emp_cause_counts:
            # Aalen-Johansen mass assigned to cause k at each event
            # time m: S(t_{m-1}) * d_{k,m} / n_m.
            s_before = np.r_[1.0, self._emp_surv[:-1]]
            mass_k = s_before * d_k / self._emp_n_risk
            cum_mass = np.cumsum(mass_k)
            cum_at = np.zeros(len(times), dtype=np.float64)
            cum_origin = 0.0
            io = np.searchsorted(self._emp_times, origin,
                                 side="right") - 1
            if io >= 0:
                cum_origin = cum_mass[io]
            it = np.searchsorted(self._emp_times, times,
                                 side="right") - 1
            ii = it >= 0
            cum_at[ii] = cum_mass[it[ii]]
            cause_cif.append(cum_at - cum_origin)  # absolute mass (o, t]

        cause_cif = np.vstack(cause_cif)  # (n_causes, n_times)
        total_cause = cause_cif.sum(axis=0)
        # Conditional failure probabilities given survival at the
        # origin: absolute interval mass divided by S(origin).  At
        # times with no intervening event both numerators are zero, so
        # the conditional cause probabilities are simply zero.
        cond_cause = cause_cif / s_origin
        # Keep the apportionment internally consistent even in the
        # presence of floating point roundoff.
        if np.any(np.abs(total_cause - fail_mass) > 1e-12):
            scale = np.where(fail_mass > 0,
                             fail_mass /
                             np.where(total_cause > 0, total_cause, 1.0),
                             1.0)
            cause_cif = cause_cif * scale
            cond_cause = cond_cause * scale
        cond_free = s_t / s_origin
        return s_origin, s_t, fail_mass, cause_cif, cond_cause, cond_free

    # ------------------------------------------------------------------
    # Fitted mode query
    # ------------------------------------------------------------------

    def _check_prediction_rows(self, exog, strata, offset):
        exog_arr = np.asarray(exog, dtype=np.float64)
        if exog_arr.ndim != 2:
            raise ValueError("`exog` must be a 2-d numeric design matrix "
                             "with one row per query row")
        if not np.all(np.isfinite(exog_arr)):
            raise ValueError("`exog` must contain only finite values")
        n_rows = exog_arr.shape[0]
        strata_arr = np.asarray(strata)
        if strata_arr.ndim != 1 or len(strata_arr) != n_rows:
            raise ValueError("`strata` must be a 1-d array with one "
                             "stratum label per query row")
        known = set(self._stratum_names)
        unknown = [s for s in np.unique(strata_arr) if s not in known]
        if unknown:
            raise ValueError("unknown stratum label(s) in prediction: %r"
                             % unknown)
        if offset is None:
            offset_arr = np.zeros(n_rows, dtype=np.float64)
        else:
            offset_arr = np.asarray(offset, dtype=np.float64)
            if offset_arr.ndim != 1 or len(offset_arr) != n_rows:
                raise ValueError("`offset` must be a 1-d array with one "
                                 "value per query row")
            if not np.all(np.isfinite(offset_arr)):
                raise ValueError("prediction offsets must be finite")
        return exog_arr, np.asarray(strata_arr), offset_arr, n_rows

    def _predict_fitted(self, origin, times, exog, strata, offset):
        exog_arr, strata_arr, offset_arr, n_rows = \
            self._check_prediction_rows(exog, strata, offset)
        n_times = len(times)
        n_causes = len(self.labels)
        abs_free = np.empty((n_rows, n_times), dtype=np.float64)
        abs_cause = np.empty((n_rows, n_causes, n_times),
                             dtype=np.float64)
        s_origin = np.empty(n_rows, dtype=np.float64)

        for r in range(n_rows):
            grid = self._grid[strata_arr[r]]
            w = np.exp(np.dot(self._params, exog_arr[r]) +
                       offset_arr[r])  # (K,)
            # Scale the cause-specific jumps onto this prediction row.
            cause_jumps = self._jumps[strata_arr[r]] * w[:, None]
            total_jumps = cause_jumps.sum(axis=0)
            # Total cumulative hazard is the running sum of the aligned
            # jumps, which is exactly consistent with the per-cause
            # jumps used for same-day apportionment.
            surv = np.exp(-np.cumsum(total_jumps))  # post-event S (M,)
            s_before = np.r_[1.0, surv[:-1]]

            # Same-day tie rule: the survival mass lost at a shared
            # event time is split across causes in proportion to their
            # scaled hazard jumps, so equal jumps receive equal mass
            # and zero jumps receive nothing.  Vectorized over the grid
            # to avoid a Python-level loop over (potentially many) event
            # times.
            active = total_jumps > 0.0
            day_mass = np.zeros(len(grid), dtype=np.float64)
            day_mass[active] = s_before[active] * \
                (1.0 - np.exp(-total_jumps[active]))
            share = np.zeros_like(cause_jumps)
            with np.errstate(invalid="ignore", divide="ignore"):
                share[:, active] = cause_jumps[:, active] / \
                    total_jumps[active]
            # CIF_k(t) = cumsum over event days of day_mass * share_k.
            cif_grid = np.cumsum(day_mass[None, :] * share, axis=1)

            s0 = float(self._surv_step(grid, surv,
                                       np.array([origin]))[0])
            if s0 <= 0.0:
                raise ValueError("no failure-free probability mass "
                                 "remains at the origin for query row %d; "
                                 "conditional probabilities are not "
                                 "defined" % r)
            s_origin[r] = s0
            abs_free[r] = self._surv_step(grid, surv, times)

            cif_origin = np.zeros(n_causes, dtype=np.float64)
            io = np.searchsorted(grid, origin, side="right") - 1
            if io >= 0:
                cif_origin = cif_grid[:, io]
            it = np.searchsorted(grid, times, side="right") - 1
            cif_at = np.zeros((n_causes, n_times), dtype=np.float64)
            ii = it >= 0
            if np.any(ii):
                cif_at[:, ii] = cif_grid[:, it[ii]]
            abs_cause[r] = cif_at - cif_origin[:, None]

        return s_origin, abs_free, abs_cause

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def predict(self, origin, times, exog=None, strata=None,
                offset=None):
        """
        Probability of remaining failure-free and of failure by each
        cause over an interval.

        The origin is assumed to have been reached without failure, so
        only events occurring strictly after the origin and no later
        than each query time are counted.

        Parameters
        ----------
        origin : float
            A finite, non-negative start time, already survived
            failure-free.
        times : array_like
            A non-decreasing 1-d sequence of finite query times, each
            no smaller than the origin.  Repeated times and times equal
            to the origin are allowed.
        exog : array_like
            Fitted mode only.  A 2-d numeric design matrix with one row
            per prediction row.  Not allowed for empirical curves.
        strata : array_like
            Fitted mode only.  A 1-d array of stratum labels, one per
            prediction row.  Unknown stratum labels raise an error.
        offset : array_like, optional
            Fitted mode only.  Prediction offsets, one per prediction
            row.  If omitted, a zero offset is used.

        Returns
        -------
        pandas.DataFrame
            Cause columns use the labels supplied to
            :meth:`from_phreg` (or the integer labels 1, 2, ... for
            empirical curves), followed by the failure-free column
            ``"__failure_free__"``.  Empirical curves return one row
            per query time, indexed by time.  Fitted curves return a
            MultiIndex ``(row, time)`` with one block of times per
            prediction row.  A cause column holds the probability of
            failure from that cause within ``(origin, time]``
            conditional on surviving the origin failure-free; the
            failure-free column holds ``P(T > t | T > origin)``.
            Every row therefore sums to one, cause probabilities are
            non-decreasing in time and the failure-free probability is
            non-increasing.
        """

        origin, times = self._validate_query_base(origin, times)

        if not self._fitted:
            if exog is not None or strata is not None or \
                    offset is not None:
                raise ValueError("exog, strata and offset are only "
                                 "accepted in the fitted (from_phreg) "
                                 "mode")
            s_origin, s_t, fail_mass, abs_k, cond_k, cond_free = \
                self._predict_empirical(origin, times)
            labels = list(range(1, len(self.cinc) + 1))
            data = {label: cond_k[j] for j, label in enumerate(labels)}
            data[FAILURE_FREE_LABEL] = cond_free
            frame = pd.DataFrame(data,
                                 index=pd.Index(times, name="time"))
            frame.attrs["origin"] = origin
            frame.attrs["survival_at_origin"] = s_origin
            return frame

        if exog is None or strata is None:
            raise ValueError("fitted mode requires both `exog` and "
                             "`strata`, with one entry per prediction row")
        s_origin, abs_free, abs_cause = self._predict_fitted(
            origin, times, exog, strata, offset)
        n_rows = abs_cause.shape[0]

        frames = []
        time_index = pd.Index(times, name="time")
        for r in range(n_rows):
            cond_k = abs_cause[r] / s_origin[r]
            cond_free = abs_free[r] / s_origin[r]
            data = {self.labels[j]: cond_k[j]
                    for j in range(len(self.labels))}
            data[FAILURE_FREE_LABEL] = cond_free
            frames.append(pd.DataFrame(data, index=time_index))
        full = pd.concat(frames, keys=np.arange(n_rows),
                         names=["row", "time"])
        full.attrs["origin"] = origin
        full.attrs["survival_at_origin"] = s_origin
        return full



class SurvfuncRight:
    """
    Estimation and inference for a survival function.

    The survival function S(t) = P(T > t) is the probability that an
    event time T is greater than t.

    This class currently only supports right censoring.

    Parameters
    ----------
    time : array_like
        An array of times (censoring times or event times)
    status : array_like
        Status at the event time, status==1 is the 'event'
        (e.g. death, failure), meaning that the event
        occurs at the given value in `time`; status==0
        indicates that censoring has occurred, meaning that
        the event occurs after the given value in `time`.
    entry : array_like, optional An array of entry times for handling
        left truncation (the subject is not in the risk set on or
        before the entry time)
    title : str
        Optional title used for plots and summary output.
    freq_weights : array_like
        Optional frequency weights
    exog : array_like
        Optional, if present used to account for violation of
        independent censoring.
    bw_factor : float
        Band-width multiplier for kernel-based estimation.  Only used
        if exog is provided.

    Attributes
    ----------
    surv_prob : array_like
        The estimated value of the survivor function at each time
        point in `surv_times`.
    surv_prob_se : array_like
        The standard errors for the values in `surv_prob`.  Not available
        if exog is provided.
    surv_times : array_like
        The points where the survival function changes.
    n_risk : array_like
        The number of subjects at risk just before each time value in
        `surv_times`.  Not available if exog is provided.
    n_events : array_like
        The number of events (e.g. deaths) that occur at each point
        in `surv_times`.  Not available if exog is provided.

    Notes
    -----
    If exog is None, the standard Kaplan-Meier estimator is used.  If
    exog is not None, a local estimate of the marginal survival
    function around each point is constructed, and these are then
    averaged.  This procedure gives an estimate of the marginal
    survival function that accounts for dependent censoring as long as
    the censoring becomes independent when conditioning on the
    covariates in exog.  See Zeng et al. (2004) for details.

    References
    ----------
    D. Zeng (2004).  Estimating marginal survival function by
    adjusting for dependent censoring using many covariates.  Annals
    of Statistics 32:4.
    https://arxiv.org/pdf/math/0409180.pdf
    """

    def __init__(self, time, status, entry=None, title=None,
                 freq_weights=None, exog=None, bw_factor=1.):

        _checkargs(time, status, entry, freq_weights, exog)
        time = self.time = np.asarray(time)
        status = self.status = np.asarray(status)
        if freq_weights is not None:
            freq_weights = self.freq_weights = np.asarray(freq_weights)

        if entry is not None:
            entry = self.entry = np.asarray(entry)

        if exog is not None:
            if entry is not None:
                raise ValueError("exog and entry cannot both be present")
            from ._kernel_estimates import _kernel_survfunc
            exog = self.exog = np.asarray(exog)
            nobs = exog.shape[0]
            kw = nobs**(-1/3.0) * bw_factor
            kfunc = lambda x: np.exp(-x**2 / kw**2).sum(1)
            x = _kernel_survfunc(time, status, exog, kfunc, freq_weights)
            self.surv_prob = x[0]
            self.surv_times = x[1]
            return

        x = _calc_survfunc_right(time, status, weights=freq_weights,
                                 entry=entry)

        self.surv_prob = x[0]
        self.surv_prob_se = x[1]
        self.surv_times = x[2]
        self.n_risk = x[4]
        self.n_events = x[5]
        self.title = "" if not title else title

    def plot(self, ax=None):
        """
        Plot the survival function.

        Examples
        --------
        Change the line color:

        >>> import statsmodels.api as sm
        >>> data = sm.datasets.get_rdataset("flchain", "survival").data
        >>> df = data.loc[data.sex == "F", :]
        >>> sf = sm.SurvfuncRight(df["futime"], df["death"])
        >>> fig = sf.plot()
        >>> ax = fig.get_axes()[0]
        >>> li = ax.get_lines()
        >>> li[0].set_color('purple')
        >>> li[1].set_color('purple')

        Do not show the censoring points:

        >>> fig = sf.plot()
        >>> ax = fig.get_axes()[0]
        >>> li = ax.get_lines()
        >>> li[1].set_visible(False)
        """

        return plot_survfunc(self, ax)

    def quantile(self, p):
        """
        Estimated quantile of a survival distribution.

        Parameters
        ----------
        p : float
            The probability point at which the quantile
            is determined.

        Returns the estimated quantile.
        """

        # SAS uses a strict inequality here.
        ii = np.flatnonzero(self.surv_prob < 1 - p)

        if len(ii) == 0:
            return np.nan

        return self.surv_times[ii[0]]

    def quantile_ci(self, p, alpha=0.05, method='cloglog'):
        """
        Returns a confidence interval for a survival quantile.

        Parameters
        ----------
        p : float
            The probability point for which a confidence interval is
            determined.
        alpha : float
            The confidence interval has nominal coverage probability
            1 - `alpha`.
        method : str
            Function to use for g-transformation, must be ...

        Returns
        -------
        lb : float
            The lower confidence limit.
        ub : float
            The upper confidence limit.

        Notes
        -----
        The confidence interval is obtained by inverting Z-tests.  The
        limits of the confidence interval will always be observed
        event times.

        References
        ----------
        The method is based on the approach used in SAS, documented here:

          http://support.sas.com/documentation/cdl/en/statug/68162/HTML/default/viewer.htm#statug_lifetest_details03.htm
        """

        tr = norm.ppf(1 - alpha / 2)

        method = method.lower()
        if method == "cloglog":
            g = lambda x: np.log(-np.log(x))
            gprime = lambda x: -1 / (x * np.log(x))
        elif method == "linear":
            g = lambda x: x
            gprime = lambda x: 1
        elif method == "log":
            g = np.log
            gprime = lambda x: 1 / x
        elif method == "logit":
            g = lambda x: np.log(x / (1 - x))
            gprime = lambda x: 1 / (x * (1 - x))
        elif method == "asinsqrt":
            g = lambda x: np.arcsin(np.sqrt(x))
            gprime = lambda x: 1 / (2 * np.sqrt(x) * np.sqrt(1 - x))
        else:
            raise ValueError("unknown method")

        r = g(self.surv_prob) - g(1 - p)
        r /= (gprime(self.surv_prob) * self.surv_prob_se)

        ii = np.flatnonzero(np.abs(r) <= tr)
        if len(ii) == 0:
            return np.nan, np.nan

        lb = self.surv_times[ii[0]]

        if ii[-1] == len(self.surv_times) - 1:
            ub = np.inf
        else:
            ub = self.surv_times[ii[-1] + 1]

        return lb, ub

    def summary(self):
        """
        Return a summary of the estimated survival function.

        The summary is a dataframe containing the unique event times,
        estimated survival function values, and related quantities.
        """

        df = pd.DataFrame(index=self.surv_times)
        df.index.name = "Time"
        df["Surv prob"] = self.surv_prob
        df["Surv prob SE"] = self.surv_prob_se
        df["num at risk"] = self.n_risk
        df["num events"] = self.n_events

        return df

    def simultaneous_cb(self, alpha=0.05, method="hw", transform="log"):
        """
        Returns a simultaneous confidence band for the survival function.

        Parameters
        ----------
        alpha : float
            `1 - alpha` is the desired simultaneous coverage
            probability for the confidence region.  Currently alpha
            must be set to 0.05, giving 95% simultaneous intervals.
        method : str
            The method used to produce the simultaneous confidence
            band.  Only the Hall-Wellner (hw) method is currently
            implemented.
        transform : str
            The used to produce the interval (note that the returned
            interval is on the survival probability scale regardless
            of which transform is used).  Only `log` and `arcsin` are
            implemented.

        Returns
        -------
        lcb : array_like
            The lower confidence limits corresponding to the points
            in `surv_times`.
        ucb : array_like
            The upper confidence limits corresponding to the points
            in `surv_times`.
        """

        method = method.lower()
        if method != "hw":
            msg = "only the Hall-Wellner (hw) method is implemented"
            raise ValueError(msg)

        if alpha != 0.05:
            raise ValueError("alpha must be set to 0.05")

        transform = transform.lower()
        s2 = self.surv_prob_se**2 / self.surv_prob**2
        nn = self.n_risk
        if transform == "log":
            denom = np.sqrt(nn) * np.log(self.surv_prob)
            theta = 1.3581 * (1 + nn * s2) / denom
            theta = np.exp(theta)
            lcb = self.surv_prob**(1/theta)
            ucb = self.surv_prob**theta
        elif transform == "arcsin":
            k = 1.3581
            k *= (1 + nn * s2) / (2 * np.sqrt(nn))
            k *= np.sqrt(self.surv_prob / (1 - self.surv_prob))
            f = np.arcsin(np.sqrt(self.surv_prob))
            v = np.clip(f - k, 0, np.inf)
            lcb = np.sin(v)**2
            v = np.clip(f + k, -np.inf, np.pi/2)
            ucb = np.sin(v)**2
        else:
            raise ValueError("Unknown transform")

        return lcb, ucb


def survdiff(time, status, group, weight_type=None, strata=None,
             entry=None, **kwargs):
    """
    Test for the equality of two survival distributions.

    Parameters
    ----------
    time : array_like
        The event or censoring times.
    status : array_like
        The censoring status variable, status=1 indicates that the
        event occurred, status=0 indicates that the observation was
        censored.
    group : array_like
        Indicators of the two groups
    weight_type : str
        The following weight types are implemented:
            None (default) : logrank test
            fh : Fleming-Harrington, weights by S^(fh_p),
                 requires exponent fh_p to be provided as keyword
                 argument; the weights are derived from S defined at
                 the previous event time, and the first weight is
                 always 1.
            gb : Gehan-Breslow, weights by the number at risk
            tw : Tarone-Ware, weights by the square root of the number
                 at risk
    strata : array_like
        Optional stratum indicators for a stratified test
    entry : array_like
        Entry times to handle left truncation. The subject is not in
        the risk set on or before the entry time.

    Returns
    -------
    chisq : The chi-square (1 degree of freedom) distributed test
            statistic value
    pvalue : The p-value for the chi^2 test
    """

    time = np.asarray(time)
    status = np.asarray(status)
    group = np.asarray(group)

    gr = np.unique(group)

    if strata is None:
        obs, var = _survdiff(time, status, group, weight_type, gr,
                             entry, **kwargs)
    else:
        strata = np.asarray(strata)
        stu = np.unique(strata)
        obs, var = 0., 0.
        for st in stu:
            # could be more efficient?
            ii = (strata == st)
            obs1, var1 = _survdiff(time[ii], status[ii], group[ii],
                                   weight_type, gr, entry, **kwargs)
            obs += obs1
            var += var1

    chisq = obs.dot(np.linalg.solve(var, obs))  # (O - E).T * V^(-1) * (O - E)
    pvalue = 1 - chi2.cdf(chisq, len(gr)-1)

    return chisq, pvalue


def _survdiff(time, status, group, weight_type, gr, entry=None,
              **kwargs):
    # logrank test for one stratum
    # calculations based on https://web.stanford.edu/~lutian/coursepdf/unit6.pdf
    # formula for variance better to take from https://web.stanford.edu/~lutian/coursepdf/survweek3.pdf

    # Get the unique times.
    if entry is None:
        utimes, rtimes = np.unique(time, return_inverse=True)
    else:
        utimes, rtimes = np.unique(np.concatenate((time, entry)),
                                   return_inverse=True)
        rtimes = rtimes[0:len(time)]

    # Split entry times by group if present (should use pandas groupby)
    tse = [(gr_i, None) for gr_i in gr]
    if entry is not None:
        for k, _ in enumerate(gr):
            ii = (group == gr[k])
            entry1 = entry[ii]
            tse[k] = (gr[k], entry1)

    # Event count and risk set size at each time point, per group and overall.
    # TODO: should use Pandas groupby
    nrisk, obsv = [], []
    ml = len(utimes)
    for g, entry0 in tse:

        mk = (group == g)
        n = np.bincount(rtimes, weights=mk, minlength=ml)

        ob = np.bincount(rtimes, weights=status*mk, minlength=ml)
        obsv.append(ob)

        if entry is not None:
            n = np.cumsum(n) - n
            rentry = np.searchsorted(utimes, entry0, side='left')
            n0 = np.bincount(rentry, minlength=ml)
            n0 = np.cumsum(n0) - n0
            nr = n0 - n
        else:
            nr = np.cumsum(n[::-1])[::-1]

        nrisk.append(nr)

    obs = sum(obsv)
    nrisk_tot = sum(nrisk)
    ix = np.flatnonzero(nrisk_tot > 1)

    weights = None
    if weight_type is not None:
        weight_type = weight_type.lower()
        if weight_type == "gb":
            weights = nrisk_tot
        elif weight_type == "tw":
            weights = np.sqrt(nrisk_tot)
        elif weight_type == "fh":
            if "fh_p" not in kwargs:
                msg = "weight_type type 'fh' requires specification of fh_p"
                raise ValueError(msg)
            fh_p = kwargs["fh_p"]
            # Calculate the survivor function directly to avoid the
            # overhead of creating a SurvfuncRight object
            sp = 1 - obs / nrisk_tot.astype(np.float64)
            sp = np.log(sp)
            sp = np.cumsum(sp)
            sp = np.exp(sp)
            weights = sp**fh_p
            weights = np.roll(weights, 1)
            weights[0] = 1
        else:
            raise ValueError("weight_type not implemented")

    dfs = len(gr) - 1
    r = np.vstack(nrisk) / np.clip(nrisk_tot, 1e-10, np.inf)[None, :]  # each line is timeseries of r's. line per group

    # The variance of event counts in each group.
    groups_oe = []
    groups_var = []

    var_denom = nrisk_tot - 1
    var_denom = np.clip(var_denom, 1e-10, np.inf)

    # use the first group as a reference
    for g in range(1, dfs+1):
        # Difference between observed and  expected number of events in the group #g
        oe = obsv[g] - r[g]*obs

        # build one row of the dfs x dfs variance matrix
        var_tensor_part = r[1:, :].T * (np.eye(1, dfs, g-1).ravel() - r[g, :, None])  # r*(1 - r) in multidim
        var_scalar_part = obs * (nrisk_tot - obs) / var_denom
        var = var_tensor_part * var_scalar_part[:, None]

        if weights is not None:
            oe = weights * oe
            var = (weights**2)[:, None] * var

        # sum over times and store
        groups_oe.append(oe[ix].sum())
        groups_var.append(var[ix].sum(axis=0))

    obs_vec = np.hstack(groups_oe)
    var_mat = np.vstack(groups_var)

    return obs_vec, var_mat


def plot_survfunc(survfuncs, ax=None):
    """
    Plot one or more survivor functions.

    Parameters
    ----------
    survfuncs : object or array_like
        A single SurvfuncRight object, or a list or SurvfuncRight
        objects that are plotted together.

    Returns
    -------
    A figure instance on which the plot was drawn.

    Examples
    --------
    Add a legend:

    >>> import statsmodels.api as sm
    >>> from statsmodels.duration.survfunc import plot_survfunc
    >>> data = sm.datasets.get_rdataset("flchain", "survival").data
    >>> df = data.loc[data.sex == "F", :]
    >>> sf0 = sm.SurvfuncRight(df["futime"], df["death"])
    >>> sf1 = sm.SurvfuncRight(3.0 * df["futime"], df["death"])
    >>> fig = plot_survfunc([sf0, sf1])
    >>> ax = fig.get_axes()[0]
    >>> ax.set_position([0.1, 0.1, 0.64, 0.8])
    >>> ha, lb = ax.get_legend_handles_labels()
    >>> leg = fig.legend((ha[0], ha[1]), (lb[0], lb[1]), loc='center right')

    Change the line colors:

    >>> fig = plot_survfunc([sf0, sf1])
    >>> ax = fig.get_axes()[0]
    >>> ax.set_position([0.1, 0.1, 0.64, 0.8])
    >>> ha, lb = ax.get_legend_handles_labels()
    >>> ha[0].set_color('purple')
    >>> ha[1].set_color('orange')
    """

    fig, ax = utils.create_mpl_ax(ax)

    # If we have only a single survival function to plot, put it into
    # a list.
    try:
        assert type(survfuncs[0]) is SurvfuncRight
    except:
        survfuncs = [survfuncs]

    for gx, sf in enumerate(survfuncs):

        # The estimated survival function does not include a point at
        # time 0, include it here for plotting.
        surv_times = np.concatenate(([0], sf.surv_times))
        surv_prob = np.concatenate(([1], sf.surv_prob))

        # If the final times are censoring times they are not included
        # in the survival function so we add them here
        mxt = max(sf.time)
        if mxt > surv_times[-1]:
            surv_times = np.concatenate((surv_times, [mxt]))
            surv_prob = np.concatenate((surv_prob, [surv_prob[-1]]))

        label = getattr(sf, "title", "Group %d" % (gx + 1))

        li, = ax.step(surv_times, surv_prob, '-', label=label, lw=2,
                      where='post')

        # Plot the censored points.
        ii = np.flatnonzero(np.logical_not(sf.status))
        ti = np.unique(sf.time[ii])
        jj = np.searchsorted(surv_times, ti) - 1
        sp = surv_prob[jj]
        ax.plot(ti, sp, '+', ms=12, color=li.get_color(),
                label=label + " points")

    ax.set_ylim(0, 1.01)

    return fig
