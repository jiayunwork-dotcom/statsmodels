"""
Created on Fri Dec 19 11:29:18 2014

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
from scipy import stats
import pandas as pd


# ---------------------------------------------------------------------------
# Shared preparation of prediction requests
#
# Exposure and offset appear in several prediction entry points: the model
# ``predict`` methods, ``Results.predict`` (which delegates to the model) and
# the various ``get_prediction`` implementations.  Keeping the source rules in
# one place makes the direct prediction and the inferential prediction agree.
#
# The rules implemented below are
#
# * if ``exog`` is not provided, the exposure and offset used in the fit are
#   the defaults;
# * if ``exog`` is provided, the fitted exposure and offset are dropped, even
#   when the supplied values are equal to the training design;
# * an explicitly provided ``exposure`` and ``offset`` each override their own
#   fitted value, the other one still follows the ``exog`` rule.
#
# Models store the log of the exposure internally (both GLM and the count
# models do), but the public argument is always the raw exposure.  The
# resolver below therefore returns the *internal* log exposure, so that no
# round trip through ``exp``/``log`` is needed in the prediction code.
# ---------------------------------------------------------------------------


def _validate_1d_real(value, nrows, name, positive=False):
    """Validate a scalar or a one dimensional prediction input.

    Scalars are returned unchanged, so that they broadcast over the
    prediction rows.  One dimensional arrays are used positionally (pandas
    indexes are ignored) and must have one element per prediction row.
    """
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued, got complex input")
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be real-valued") from exc

    if arr.ndim == 0:
        value_ = float(arr)
        if not np.isfinite(value_):
            raise ValueError(f"{name} must be finite, got {value_!r}")
        if positive and not value_ > 0:
            raise ValueError(f"{name} must be strictly positive, got {value_!r}")
        return value_

    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be a scalar or a 1-dimensional array, got "
            f"{arr.ndim} dimensions"
        )
    if arr.shape[0] != nrows:
        raise ValueError(
            f"{name} has {arr.shape[0]} values but {nrows} prediction rows "
            "were requested"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if positive and not np.all(arr > 0):
        raise ValueError(f"{name} must be strictly positive")
    return arr


def _validate_stored(value, nrows, name):
    """Validate an array stored on the model, without checking its sign.

    Stored exposures are in log space and might in principle be non-finite
    if a degenerate exposure was passed at fit time.  The prediction request
    is only required to be validated for inputs supplied by the caller, so
    stored values are only checked for shape.
    """
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim != 1:
        raise ValueError(f"stored {name} must be scalar or 1-dimensional")
    if arr.shape[0] != nrows:
        raise ValueError(
            f"stored {name} has {arr.shape[0]} values but {nrows} prediction "
            "rows were requested"
        )
    return arr


def as_2d_exog(model, exog):
    """Interpret a one dimensional ``exog`` as in results ``predict``.

    A one dimensional design is a single row when the model has several
    regressors, and a column when it has a single regressor.
    """
    arr = np.asarray(exog)
    if arr.ndim != 1:
        return arr
    model_exog = getattr(model, "exog", None)
    if model_exog is not None and np.ndim(model_exog) == 2:
        k_exog = model_exog.shape[1]
    else:  # pragma: no cover - models without a stored exog
        k_exog = 1
    if k_exog > 1:
        return arr[None, :]
    return arr[:, None]


def prediction_nrows(model, exog):
    """Number of prediction rows for a resolved design matrix.

    A one dimensional ``exog`` is interpreted the way ``np.dot`` would use
    it: as a single row when the model has more than one regressor, and as
    one column otherwise.  ``Results.predict`` always supplies a two
    dimensional design, so this only matters for direct ``model.predict``
    calls.
    """
    exog = np.asarray(exog)
    if exog.ndim == 2:
        return exog.shape[0]
    if exog.ndim != 1:
        raise ValueError("exog must be one or two dimensional")
    model_exog = getattr(model, "exog", None)
    if model_exog is not None and np.ndim(model_exog) == 2:
        k_exog = model_exog.shape[1]
        if k_exog > 1:
            return 1
    return exog.shape[0]


def _rowwise_linpred_var(exog, covb):
    """Variance of ``exog @ params`` for each row, shape (nrows,)."""
    return (exog * np.dot(covb, exog.T).T).sum(1)


def _predict_exog(exog, exog_provided):
    """Design matrix to pass on to ``model.predict``.

    When the caller did not supply a design matrix we pass ``None`` so that
    the model uses its own training design *and* its fitted exposure and
    offset.  Passing the training design explicitly would, by the source
    rules, invalidate the fitted exposure and offset.
    """
    return exog if exog_provided else None


class RowFactoredJacobian:
    """Jacobian of a row-wise prediction, factored as ``factor * exog``.

    Many count model predictions are of the form ``f(x_i @ beta + ...)``
    where the derivative with respect to ``beta`` is a scalar factor times
    the row of the design matrix.  Storing only the factor, shape
    ``(nrows, k)``, avoids materializing the full ``(nrows, k, p)``
    derivative tensor when there are ``k`` requested statistics such as
    probabilities for several counts.

    Parameters
    ----------
    factor : ndarray
        Shape ``(nrows,)`` for a single statistic or ``(nrows, k)`` for
        several statistics.  The derivative of the statistic for row ``i``
        and column ``j`` equals ``factor[i, j] * exog[i]``.
    exog : ndarray
        Design matrix used by the prediction, shape ``(nrows, p)``.
    """

    def __init__(self, factor, exog):
        self.factor = np.asarray(factor, dtype=np.float64)
        self.exog = exog

    @property
    def shape(self):
        return (self.exog.shape[0],) + self.factor.shape[1:] + (
            self.exog.shape[1],)

    def asarray(self):
        """Materialize the full Jacobian, used only for small problems."""
        return self.factor[..., None] * self.exog[:, None, :]

    def row_variance(self, covb):
        """Variance of each row statistic under ``covb``.

        ``g' cov g`` with ``g = factor * x`` reduces to
        ``factor**2 * x' cov x`` and therefore never forms the full tensor.
        """
        linpred_var = _rowwise_linpred_var(self.exog, covb)
        if self.factor.ndim == 1:
            return self.factor**2 * linpred_var
        return self.factor**2 * linpred_var[:, None]

    def aggregate_derivative(self, agg_weights, covb=None):
        """Derivative of the weighted row average.

        Shape ``(p,)`` for a single statistic and ``(k, p)`` for several
        statistics per row.
        """
        w = np.asarray(agg_weights, dtype=np.float64)
        nrows = self.exog.shape[0]
        if self.factor.ndim == 1:
            return np.dot(self.factor * w, self.exog) / nrows
        return np.dot((self.factor * w[:, None]).T, self.exog) / nrows


def resolve_linear_terms(model, exog,
                         exposure, offset, nrows=None):
    """Resolve a prediction request into log-exposure and offset terms.

    Returns the two quantities that are added to ``exog @ params``.  Stored
    values are used directly (no ``exp``/``log`` round trip), user supplied
    exposures are validated and logged.  A missing term is returned as
    ``0.0`` so that it broadcasts over the prediction rows.
    """
    exog_provided = exog is not None
    if exposure is None:
        stored_exposure = getattr(model, "exposure", None)
        if not exog_provided and stored_exposure is not None:
            log_exposure = _validate_stored(
                stored_exposure, nrows, "exposure"
            )
        else:
            log_exposure = 0.0
    else:
        raw_exposure = _validate_1d_real(
            exposure, nrows, "exposure", positive=True
        )
        log_exposure = np.log(raw_exposure)

    if offset is None:
        stored_offset = getattr(model, "offset", None)
        if not exog_provided and stored_offset is not None:
            offset = _validate_stored(stored_offset, nrows, "offset")
        else:
            offset = 0.0
    else:
        offset = _validate_1d_real(offset, nrows, "offset")

    return offset, log_exposure


def raw_exposure_from_log(log_exposure):
    """Convert a resolved log exposure to the raw exposure argument.

    ``0`` log exposure means "no exposure" and maps to ``None``.  Internal
    call sites of multi-part models keep the exposure in log space but must
    call the shared count-model ``predict`` with the raw exposure.
    """
    arr = np.asarray(log_exposure)
    if arr.ndim == 0 and arr == 0:
        return None
    if arr.ndim > 0 and not np.any(arr):
        return None
    return np.exp(log_exposure)


# this is similar to ContrastResults after t_test, partially copied, adjusted
class PredictionResultsBase:
    """Based class for get_prediction results
    """

    def __init__(self, predicted, var_pred, func=None, deriv=None,
                 df=None, dist=None, row_labels=None, **kwds):
        self.predicted = predicted
        self.var_pred = var_pred
        self.func = func
        self.deriv = deriv
        self.df = df
        self.row_labels = row_labels
        self.__dict__.update(kwds)

        if dist is None or dist == 'norm':
            self.dist = stats.norm
            self.dist_args = ()
        elif dist == 't':
            self.dist = stats.t
            self.dist_args = (self.df,)
        else:
            self.dist = dist
            self.dist_args = ()

    @property
    def se(self):
        return np.sqrt(self.var_pred)

    @property
    def tvalues(self):
        return self.predicted / self.se

    def t_test(self, value=0, alternative='two-sided'):
        '''z- or t-test for hypothesis that mean is equal to value

        Parameters
        ----------
        value : array_like
            value under the null hypothesis
        alternative : str
            'two-sided', 'larger', 'smaller'

        Returns
        -------
        stat : ndarray
            test statistic
        pvalue : ndarray
            p-value of the hypothesis test, the distribution is given by
            the attribute of the instance, specified in `__init__`. Default
            if not specified is the normal distribution.

        '''
        # assumes symmetric distribution
        stat = (self.predicted - value) / self.se

        if alternative in ['two-sided', '2-sided', '2s']:
            pvalue = self.dist.sf(np.abs(stat), *self.dist_args)*2
        elif alternative in ['larger', 'l']:
            pvalue = self.dist.sf(stat, *self.dist_args)
        elif alternative in ['smaller', 's']:
            pvalue = self.dist.cdf(stat, *self.dist_args)
        else:
            raise ValueError('invalid alternative')
        return stat, pvalue

    def _conf_int_generic(self, center, se, alpha, dist_args=None):
        """internal function to avoid code duplication
        """
        if dist_args is None:
            dist_args = ()

        q = self.dist.ppf(1 - alpha / 2., *dist_args)
        lower = center - q * se
        upper = center + q * se
        ci = np.column_stack((lower, upper))
        # if we want to stack at a new last axis, for lower.ndim > 1
        # np.concatenate((lower[..., None], upper[..., None]), axis=-1)
        return ci

    def conf_int(self, *, alpha=0.05, **kwds):
        """Confidence interval for the predicted value.

        Parameters
        ----------
        alpha : float, optional
            The significance level for the confidence interval.
            ie., The default `alpha` = .05 returns a 95% confidence interval.

        kwds : extra keyword arguments
            Ignored in base class, only for compatibility, consistent signature
            with subclasses

        Returns
        -------
        ci : ndarray, (k_constraints, 2)
            The array has the lower and the upper limit of the confidence
            interval in the columns.
        """

        ci = self._conf_int_generic(self.predicted, self.se, alpha,
                                    dist_args=self.dist_args)
        return ci

    def summary_frame(self, alpha=0.05):
        """Summary frame

        Parameters
        ----------
        alpha : float, optional
            The significance level for the confidence interval.
            ie., The default `alpha` = .05 returns a 95% confidence interval.

        Returns
        -------
        pandas DataFrame with columns 'predicted', 'se', 'ci_lower', 'ci_upper'
        """
        ci = self.conf_int(alpha=alpha)
        to_include = {}
        to_include['predicted'] = self.predicted
        to_include['se'] = self.se
        to_include['ci_lower'] = ci[:, 0]
        to_include['ci_upper'] = ci[:, 1]

        self.table = to_include
        # pandas dict does not handle 2d_array
        # data = np.column_stack(list(to_include.values()))
        # names = ....
        res = pd.DataFrame(to_include, index=self.row_labels,
                           columns=to_include.keys())
        return res


class PredictionResultsMonotonic(PredictionResultsBase):

    def __init__(self, predicted, var_pred, linpred=None, linpred_se=None,
                 func=None, deriv=None, df=None, dist=None, row_labels=None):
        # TODO: is var_resid used? drop from arguments?
        self.predicted = predicted
        self.var_pred = var_pred
        self.linpred = linpred
        self.linpred_se = linpred_se
        self.func = func
        self.deriv = deriv
        self.df = df
        self.row_labels = row_labels

        if dist is None or dist == 'norm':
            self.dist = stats.norm
            self.dist_args = ()
        elif dist == 't':
            self.dist = stats.t
            self.dist_args = (self.df,)
        else:
            self.dist = dist
            self.dist_args = ()

    def _conf_int_generic(self, center, se, alpha, dist_args=None):
        """internal function to avoid code duplication
        """
        if dist_args is None:
            dist_args = ()

        q = self.dist.ppf(1 - alpha / 2., *dist_args)
        lower = center - q * se
        upper = center + q * se
        ci = np.column_stack((lower, upper))
        # if we want to stack at a new last axis, for lower.ndim > 1
        # np.concatenate((lower[..., None], upper[..., None]), axis=-1)
        return ci

    def conf_int(self, method='endpoint', alpha=0.05, **kwds):
        """Confidence interval for the predicted value.

        This is currently only available for t and z tests.

        Parameters
        ----------
        method : {"endpoint", "delta"}
            Method for confidence interval, "m
            If method is "endpoint", then the confidence interval of the
            linear predictor is transformed by the prediction function.
            If method is "delta", then the delta-method is used. The confidence
            interval in this case might reach outside the range of the
            prediction, for example probabilities larger than one or smaller
            than zero.
        alpha : float, optional
            The significance level for the confidence interval.
            ie., The default `alpha` = .05 returns a 95% confidence interval.
        kwds : extra keyword arguments
            currently ignored, only for compatibility, consistent signature

        Returns
        -------
        ci : ndarray, (k_constraints, 2)
            The array has the lower and the upper limit of the confidence
            interval in the columns.
        """
        tmp = np.linspace(0, 1, 6)
        # TODO: drop check?
        is_linear = (self.func(tmp) == tmp).all()
        if method == 'endpoint' and not is_linear:
            ci_linear = self._conf_int_generic(self.linpred, self.linpred_se,
                                               alpha,
                                               dist_args=self.dist_args)
            ci = self.func(ci_linear)
        elif method == 'delta' or is_linear:
            ci = self._conf_int_generic(self.predicted, self.se, alpha,
                                        dist_args=self.dist_args)

        return ci


class PredictionResultsDelta(PredictionResultsBase):
    """Prediction results based on delta method
    """

    def __init__(self, results_delta, **kwds):

        predicted = results_delta.predicted()
        var_pred = results_delta.var()

        super().__init__(predicted, var_pred, **kwds)

    @classmethod
    def from_parts(cls, predicted, var_pred, df=None, dist=None,
                   row_labels=None, deriv=None, **kwds):
        """Construct directly from the predicted value and its variance.

        This is used by the shared inference layer after it has already
        computed the prediction and its derivative, and avoids the extra
        function evaluations of the ``NonlinearDeltaCov`` wrapper.
        """
        self = cls.__new__(cls)
        PredictionResultsBase.__init__(
            self, predicted, var_pred, df=df, dist=dist,
            row_labels=row_labels, deriv=deriv, **kwds)
        return self


class PredictionResultsMean(PredictionResultsBase):
    """Prediction results for GLM.

    This results class is used for backwards compatibility for
    `get_prediction` with GLM. The new PredictionResults classes dropped the
    `_mean` post fix in the attribute names.
    """

    def __init__(self, predicted_mean, var_pred_mean, var_resid=None,
                 df=None, dist=None, row_labels=None, linpred=None, link=None):
        # TODO: is var_resid used? drop from arguments?
        self.predicted = predicted_mean
        self.var_pred = var_pred_mean
        self.df = df
        self.var_resid = var_resid
        self.row_labels = row_labels
        self.linpred = linpred
        self.link = link

        if dist is None or dist == 'norm':
            self.dist = stats.norm
            self.dist_args = ()
        elif dist == 't':
            self.dist = stats.t
            self.dist_args = (self.df,)
        else:
            self.dist = dist
            self.dist_args = ()

    @property
    def predicted_mean(self):
        # alias for backwards compatibility
        return self.predicted

    @property
    def var_pred_mean(self):
        # alias for backwards compatibility
        return self.var_pred

    @property
    def se_mean(self):
        # alias for backwards compatibility
        return self.se

    def conf_int(self, method='endpoint', alpha=0.05, **kwds):
        """Confidence interval for the predicted value.

        This is currently only available for t and z tests.

        Parameters
        ----------
        method : {"endpoint", "delta"}
            Method for confidence interval, "m
            If method is "endpoint", then the confidence interval of the
            linear predictor is transformed by the prediction function.
            If method is "delta", then the delta-method is used. The confidence
            interval in this case might reach outside the range of the
            prediction, for example probabilities larger than one or smaller
            than zero.
        alpha : float, optional
            The significance level for the confidence interval.
            ie., The default `alpha` = .05 returns a 95% confidence interval.
        kwds : extra keyword arguments
            currently ignored, only for compatibility, consistent signature

        Returns
        -------
        ci : ndarray, (k_constraints, 2)
            The array has the lower and the upper limit of the confidence
            interval in the columns.
        """
        tmp = np.linspace(0, 1, 6)
        is_linear = (self.link.inverse(tmp) == tmp).all()
        if method == 'endpoint' and not is_linear:
            ci_linear = self.linpred.conf_int(alpha=alpha, obs=False)
            ci = self.link.inverse(ci_linear)
        elif method == 'delta' or is_linear:
            se = self.se_mean
            q = self.dist.ppf(1 - alpha / 2., *self.dist_args)
            lower = self.predicted_mean - q * se
            upper = self.predicted_mean + q * se
            ci = np.column_stack((lower, upper))
            # if we want to stack at a new last axis, for lower.ndim > 1
            # np.concatenate((lower[..., None], upper[..., None]), axis=-1)

        return ci

    def summary_frame(self, alpha=0.05):
        """Summary frame

        Parameters
        ----------
        alpha : float, optional
            The significance level for the confidence interval.
            ie., The default `alpha` = .05 returns a 95% confidence interval.

        Returns
        -------
        pandas DataFrame with columns
        'mean', 'mean_se', 'mean_ci_lower', 'mean_ci_upper'.
        """
        # TODO: finish and cleanup
        ci_mean = self.conf_int(alpha=alpha)
        to_include = {}
        to_include['mean'] = self.predicted_mean
        to_include['mean_se'] = self.se_mean
        to_include['mean_ci_lower'] = ci_mean[:, 0]
        to_include['mean_ci_upper'] = ci_mean[:, 1]

        self.table = to_include
        # pandas dict does not handle 2d_array
        # data = np.column_stack(list(to_include.values()))
        # names = ....
        res = pd.DataFrame(to_include, index=self.row_labels,
                           columns=to_include.keys())
        return res


def _get_exog_predict(self, exog=None, transform=True, row_labels=None):
    """Prepare or transform exog for prediction

    Parameters
    ----------
    exog : array_like, optional
        The values for which you want to predict.
    transform : bool, optional
        If the model was fit via a formula, do you want to pass
        exog through the formula. Default is True. E.g., if you fit
        a model y ~ log(x1) + log(x2), and transform is True, then
        you can pass a data structure that contains x1 and x2 in
        their original form. Otherwise, you'd need to log the data
        first.
    row_labels : list of str or None
        If row_lables are provided, then they will replace the generated
        labels.

    Returns
    -------
    exog : ndarray
        Prediction exog
    row_labels : list of str
        Labels or pandas index for rows of prediction
    exog_provided : bool
        Whether the caller supplied ``exog``.  This matters for the exposure
        and offset defaults: fitted values are only used when the caller did
        not supply a design matrix.
    """

    exog_provided = exog is not None

    # prepare exog and row_labels, based on base Results.predict
    if transform and hasattr(self.model, 'formula') and exog is not None:
        from patsy import dmatrix
        if isinstance(exog, pd.Series):
            exog = pd.DataFrame(exog)
        exog = dmatrix(self.model.data.design_info, exog)

    if exog is not None:
        if row_labels is None:
            row_labels = getattr(exog, 'index', None)
            if callable(row_labels):
                row_labels = None

        exog = np.asarray(exog)
        if exog.ndim == 1 and (self.model.exog.ndim == 1 or
                               self.model.exog.shape[1] == 1):
            exog = exog[:, None]
        exog = np.atleast_2d(exog)  # needed in count model shape[1]
    else:
        exog = self.model.exog

        if row_labels is None:
            row_labels = getattr(self.model.data, 'row_labels', None)
    return exog, row_labels, exog_provided


def get_prediction_glm(self, exog=None, transform=True,
                       row_labels=None, linpred=None, link=None,
                       pred_kwds=None):
    """
    Compute prediction results for GLM compatible models.

    Parameters
    ----------
    exog : array_like, optional
        The values for which you want to predict.
    transform : bool, optional
        If the model was fit via a formula, do you want to pass
        exog through the formula. Default is True. E.g., if you fit
        a model y ~ log(x1) + log(x2), and transform is True, then
        you can pass a data structure that contains x1 and x2 in
        their original form. Otherwise, you'd need to log the data
        first.
    row_labels : list of str or None
        If row_lables are provided, then they will replace the generated
        labels.
    linpred : linear prediction instance
        Instance of linear prediction results used for confidence intervals
        based on endpoint transformation.
    link : instance of link function
        If no link function is provided, then the `model.family.link` is used.
    pred_kwds : dict
        Some models can take additional keyword arguments, such as offset or
        additional exog in multi-part models. See the predict method of the
        model for the details.

    Returns
    -------
    prediction_results : generalized_linear_model.PredictionResults
        The prediction results instance contains prediction and prediction
        variance and can on demand calculate confidence intervals and summary
        tables for the prediction of the mean and of new observations.
    """

    # prepare exog and row_labels, based on base Results.predict
    exog, row_labels, exog_provided = _get_exog_predict(
        self,
        exog=exog,
        transform=transform,
        row_labels=row_labels,
        )

    predicted_mean = self.model.predict(
        self.params, _predict_exog(exog, exog_provided), **pred_kwds)

    covb = self.cov_params()

    link_deriv = self.model.family.link.inverse_deriv(linpred.predicted_mean)
    var_pred_mean = link_deriv**2 * _rowwise_linpred_var(exog, covb)
    var_resid = self.scale  # self.mse_resid / weights

    # TODO: check that we have correct scale, Refactor scale #???
    # special case for now:
    if self.cov_type == 'fixed scale':
        var_resid = self.cov_kwds['scale']

    dist = ['norm', 't'][self.use_t]
    return PredictionResultsMean(
        predicted_mean, var_pred_mean, var_resid,
        df=self.df_resid, dist=dist,
        row_labels=row_labels, linpred=linpred, link=link)


def get_prediction_linear(self, exog=None, transform=True,
                          row_labels=None, pred_kwds=None, index=None):
    """
    Compute prediction results for linear prediction.

    Parameters
    ----------
    exog : array_like, optional
        The values for which you want to predict.
    transform : bool, optional
        If the model was fit via a formula, do you want to pass
        exog through the formula. Default is True. E.g., if you fit
        a model y ~ log(x1) + log(x2), and transform is True, then
        you can pass a data structure that contains x1 and x2 in
        their original form. Otherwise, you'd need to log the data
        first.
    row_labels : list of str or None
        If row_lables are provided, then they will replace the generated
        labels.
    pred_kwargs :
        Some models can take additional keyword arguments, such as offset or
        additional exog in multi-part models.
        See the predict method of the model for the details.
    index : slice or array-index
        Is used to select rows and columns of cov_params, if the prediction
        function only depends on a subset of parameters.

    Returns
    -------
    prediction_results : PredictionResults
        The prediction results instance contains prediction and prediction
        variance and can on demand calculate confidence intervals and summary
        tables for the prediction.
    """

    # prepare exog and row_labels, based on base Results.predict
    exog, row_labels, exog_provided = _get_exog_predict(
        self,
        exog=exog,
        transform=transform,
        row_labels=row_labels,
        )

    k1 = exog.shape[1]
    if len(self.params > k1):
        # TODO: we allow endpoint transformation only for the first link
        index = np.arange(k1)
    else:
        index = None
    # get linear prediction and standard errors
    covb = self.cov_params(column=index)
    var_pred = _rowwise_linpred_var(exog, covb)
    pred_kwds_linear = dict(pred_kwds or {})
    pred_kwds_linear["which"] = "linear"
    predicted = self.model.predict(
        self.params, _predict_exog(exog, exog_provided), **pred_kwds_linear)

    dist = ['norm', 't'][self.use_t]
    res = PredictionResultsBase(predicted, var_pred,
                                df=self.df_resid, dist=dist,
                                row_labels=row_labels
                                )
    return res


def get_prediction_monotonic(self, exog=None, transform=True,
                             row_labels=None, link=None,
                             pred_kwds=None, index=None):
    """
    Compute prediction results when endpoint transformation is valid.

    Parameters
    ----------
    exog : array_like, optional
        The values for which you want to predict.
    transform : bool, optional
        If the model was fit via a formula, do you want to pass
        exog through the formula. Default is True. E.g., if you fit
        a model y ~ log(x1) + log(x2), and transform is True, then
        you can pass a data structure that contains x1 and x2 in
        their original form. Otherwise, you'd need to log the data
        first.
    row_labels : list of str or None
        If row_lables are provided, then they will replace the generated
        labels.
    link : instance of link function
        If no link function is provided, then the ``mmodel.family.link` is
        used.
    pred_kwargs :
        Some models can take additional keyword arguments, such as offset or
        additional exog in multi-part models.
        See the predict method of the model for the details.
    index : slice or array-index
        Is used to select rows and columns of cov_params, if the prediction
        function only depends on a subset of parameters.

    Returns
    -------
    prediction_results : PredictionResults
        The prediction results instance contains prediction and prediction
        variance and can on demand calculate confidence intervals and summary
        tables for the prediction.
    """

    # prepare exog and row_labels, based on base Results.predict
    exog, row_labels, exog_provided = _get_exog_predict(
        self,
        exog=exog,
        transform=transform,
        row_labels=row_labels,
        )

    if link is None:
        link = self.model.family.link

    func_deriv = link.inverse_deriv

    # get linear prediction and standard errors
    covb = self.cov_params(column=index)
    linpred_var = _rowwise_linpred_var(exog, covb)
    pred_exog = _predict_exog(exog, exog_provided)
    pred_kwds_linear = dict(pred_kwds or {})
    pred_kwds_linear["which"] = "linear"
    linpred = self.model.predict(self.params, pred_exog, **pred_kwds_linear)

    predicted = self.model.predict(
        self.params, pred_exog, **(pred_kwds or {}))
    link_deriv = func_deriv(linpred)
    var_pred = link_deriv**2 * linpred_var

    dist = ['norm', 't'][self.use_t]
    res = PredictionResultsMonotonic(predicted, var_pred,
                                     df=self.df_resid, dist=dist,
                                     row_labels=row_labels, linpred=linpred,
                                     linpred_se=np.sqrt(linpred_var),
                                     func=link.inverse, deriv=func_deriv)
    return res


def get_prediction_delta(
        self,
        exog=None,
        which="mean",
        average=False,
        agg_weights=None,
        transform=True,
        row_labels=None,
        pred_kwds=None
        ):
    """
    compute prediction results

    Parameters
    ----------
    exog : array_like, optional
        The values for which you want to predict.
    which : str
        The statistic that is prediction. Which statistics are available
        depends on the model.predict method.
    average : bool
        If average is True, then the mean prediction is computed, that is,
        predictions are computed for individual exog and then them mean over
        observation is used.
        If average is False, then the results are the predictions for all
        observations, i.e. same length as ``exog``.
    agg_weights : ndarray, optional
        Aggregation weights, only used if average is True.
        The weights are not normalized.
    transform : bool, optional
        If the model was fit via a formula, do you want to pass
        exog through the formula. Default is True. E.g., if you fit
        a model y ~ log(x1) + log(x2), and transform is True, then
        you can pass a data structure that contains x1 and x2 in
        their original form. Otherwise, you'd need to log the data
        first.
    row_labels : list of str or None
        If row_lables are provided, then they will replace the generated
        labels.
    pred_kwargs :
        Some models can take additional keyword arguments, such as offset or
        additional exog in multi-part models.
        See the predict method of the model for the details.

    Returns
    -------
    prediction_results : generalized_linear_model.PredictionResults
        The prediction results instance contains prediction and prediction
        variance and can on demand calculate confidence intervals and summary
        tables for the prediction of the mean and of new observations.
    """

    # prepare exog and row_labels, based on base Results.predict
    exog, row_labels, exog_provided = _get_exog_predict(
        self,
        exog=exog,
        transform=transform,
        row_labels=row_labels,
        )

    pred_kwds = {} if pred_kwds is None else dict(pred_kwds)
    # ``which`` is passed explicitly so it must not also be in the keywords
    pred_kwds.pop("which", None)

    if agg_weights is None:
        agg_weights = np.array(1.)
    else:
        agg_weights = np.asarray(agg_weights, dtype=np.float64)
        if agg_weights.ndim > 1:
            raise ValueError("agg_weights must be one dimensional")
        if (agg_weights.ndim == 1 and
                agg_weights.shape[0] not in (1, exog.shape[0])):
            raise ValueError(
                "agg_weights must be a scalar or have one value per "
                "prediction row"
            )
    if agg_weights.ndim == 0:
        agg_weights = np.full(exog.shape[0], float(agg_weights))
    elif agg_weights.shape[0] == 1 and exog.shape[0] != 1:
        agg_weights = np.full(exog.shape[0], float(agg_weights[0]))

    params = np.asarray(self.params, dtype=np.float64)
    covb = self.cov_params()

    # Predicted statistic and its derivative with respect to the parameters.
    # The derivative is provided by the model when it is available, which
    # avoids repeated numerical differentiation of the prediction function.
    pred_exog = _predict_exog(exog, exog_provided)
    predicted = self.model.predict(params, pred_exog, which=which,
                                   **pred_kwds)
    deriv = _predict_derivatives(
        self.model, params, pred_exog, which, pred_kwds
    )
    if deriv is None or (average and not
                         isinstance(deriv, RowFactoredJacobian)):
        if average:
            # differentiate the weighted average directly, so that the full
            # per-row derivative tensor is never stored for the aggregation
            deriv = _numeric_derivatives_average(
                self.model, params, pred_exog, which, pred_kwds,
                np.shape(predicted), agg_weights)
        else:
            deriv = _numeric_derivatives(
                self.model, params, pred_exog, which, pred_kwds,
                predicted.shape
            )

    if average:
        # Each row is weighted and then divided by the number of rows, the
        # aggregation weights are not normalized.  The variance comes from
        # the joint variation of the parameters, not from averaging the
        # per-row variances.
        if isinstance(deriv, RowFactoredJacobian):
            agg_deriv = deriv.aggregate_derivative(agg_weights)
            if deriv.factor.ndim == 1:
                predicted = (predicted * agg_weights).mean()
            else:
                predicted = (predicted * agg_weights[:, None]).mean(axis=0)
        else:
            # deriv is already the derivative of the weighted average
            pred_arr = np.asarray(predicted)
            if pred_arr.ndim <= 1:
                predicted = (pred_arr * agg_weights).mean()
            else:
                predicted = (pred_arr * agg_weights[:, None]).mean(axis=0)
            agg_deriv = deriv
        var_pred = (np.dot(agg_deriv, covb) * agg_deriv).sum(-1)
        stored_deriv = agg_deriv
    else:
        if isinstance(deriv, RowFactoredJacobian):
            var_pred = deriv.row_variance(covb)
        else:
            var_pred = (np.dot(deriv, covb) * deriv).sum(-1)
        stored_deriv = deriv

    res = PredictionResultsDelta.from_parts(
        predicted, var_pred,
        df=self.df_resid, dist=['norm', 't'][self.use_t],
        # the delta results historically do not carry row labels
        row_labels=None, deriv=stored_deriv
    )
    res.predicted, res.var_pred = _squeeze_prediction(
        res.predicted, res.var_pred)
    return res


def _squeeze_prediction(predicted, var_pred):
    """Match the shape convention of the old delta method results.

    A single-row prediction with several requested statistics was returned
    with the observation axis squeezed, e.g. shape ``(k,)`` instead of
    ``(1, k)``.  Keeping that convention avoids changing the public shape
    for formula models and scalar exog rows.
    """
    predicted = np.asarray(predicted)
    var_pred = np.asarray(var_pred)
    if predicted.ndim > 1:
        predicted = predicted.squeeze()
        var_pred = var_pred.squeeze()
    return predicted, var_pred


def _numeric_derivatives(model, params, exog, which, pred_kwds, shape):
    """Fallback numerical Jacobian of the prediction function.

    Used only when the model does not provide an analytic derivative.  The
    prediction is flattened to its requested statistics, so the result has
    ``shape + (n_params,)`` and the intermediate derivative array is
    O(k * p).  No observation covariance and no full derivative tensor for
    the aggregation are formed.
    """
    from statsmodels.tools.numdiff import approx_fprime_cs, approx_fprime

    shape = tuple(shape) if isinstance(shape, tuple) else tuple(np.shape(shape))
    n_stats = int(np.prod(shape))
    n_params = params.shape[0]

    def f(p):
        out = model.predict(p, exog, which=which, **pred_kwds)
        return np.asarray(out).reshape(n_stats)

    try:
        jac = approx_fprime_cs(params, f)
    except TypeError:  # some distributions do not support complex input
        jac = approx_fprime(params, f)
    # both numdiff helpers put the parameter axis last
    return np.asarray(jac).reshape(shape + (n_params,))


def _numeric_derivatives_average(model, params, exog, which, pred_kwds,
                                 shape, agg_weights):
    """Numerical Jacobian of the weighted row average.

    The average is computed inside the differentiated function, so the
    result is O(k * p) and no per-row derivative tensor is stored.
    """
    from statsmodels.tools.numdiff import approx_fprime_cs, approx_fprime

    shape = tuple(shape) if isinstance(shape, tuple) else tuple(np.shape(shape))
    n_stats = int(np.prod(shape))
    n_params = params.shape[0]
    w = np.asarray(agg_weights, dtype=np.float64)

    def f(p):
        out = model.predict(p, exog, which=which, **pred_kwds)
        out = np.asarray(out).reshape(n_stats)
        if n_stats == 1:
            return np.array([(out * w).mean()])
        return (out.reshape(shape[0], -1) * w[:, None]).mean(axis=0)

    try:
        jac = approx_fprime_cs(params, f)
    except TypeError:  # some distributions do not support complex input
        jac = approx_fprime(params, f)
    return np.asarray(jac).reshape(-1, n_params)


def _predict_derivatives(model, params, exog, which, pred_kwds):
    """Return the analytic Jacobian of ``predict`` if the model has one.

    The model method is responsible for the interpretation of ``which`` and
    of the extra prediction keywords (for example ``exog_infl`` or
    ``y_values``).  It returns an array with one row per observation and the
    parameter numbers in the last axis.
    """
    deriv_func = getattr(model, "_predict_deriv", None)
    if deriv_func is None:
        return None
    try:
        return deriv_func(params, exog, which=which, **pred_kwds)
    except NotImplementedError:
        return None


def get_prediction(self, exog=None, transform=True, which="mean",
                   row_labels=None, average=False, agg_weights=None,
                   pred_kwds=None):
    """
    Compute prediction results when endpoint transformation is valid.

    Parameters
    ----------
    exog : array_like, optional
        The values for which you want to predict.
    transform : bool, optional
        If the model was fit via a formula, do you want to pass
        exog through the formula. Default is True. E.g., if you fit
        a model y ~ log(x1) + log(x2), and transform is True, then
        you can pass a data structure that contains x1 and x2 in
        their original form. Otherwise, you'd need to log the data
        first.
    which : str
        Which statistic is to be predicted. Default is "mean".
        The available statistics and options depend on the model.
        see the model.predict docstring
    linear : bool
        Linear has been replaced by the `which` keyword and will be
        deprecated.
        If linear is True, then `which` is ignored and the linear
        prediction is returned.
    row_labels : list of str or None
        If row_lables are provided, then they will replace the generated
        labels.
    average : bool
        If average is True, then the mean prediction is computed, that is,
        predictions are computed for individual exog and then the average
        over observation is used.
        If average is False, then the results are the predictions for all
        observations, i.e. same length as ``exog``.
    agg_weights : ndarray, optional
        Aggregation weights, only used if average is True.
        The weights are not normalized.
    **kwargs :
        Some models can take additional keyword arguments, such as offset,
        exposure or additional exog in multi-part models like zero inflated
        models.
        See the predict method of the model for the details.

    Returns
    -------
    prediction_results : PredictionResults
        The prediction results instance contains prediction and prediction
        variance and can on demand calculate confidence intervals and
        summary dataframe for the prediction.

    Notes
    -----
    Status: new in 0.14, experimental
    """
    use_endpoint = getattr(self.model, "_use_endpoint", True)

    if which == "linear" and not average:
        res = get_prediction_linear(
            self,
            exog=exog,
            transform=transform,
            row_labels=row_labels,
            pred_kwds=pred_kwds,
            )

    elif (which == "mean")and (use_endpoint is True) and (average is False):
        # endpoint transformation
        k1 = self.model.exog.shape[1]
        if len(self.params > k1):
            # TODO: we allow endpoint transformation only for the first link
            index = np.arange(k1)
        else:
            index = None

        pred_kwds["which"] = which
        # TODO: add link or ilink to all link based models (except zi
        link = getattr(self.model, "link", None)
        if link is None:
            # GLM
            if hasattr(self.model, "family"):
                link = getattr(self.model.family, "link", None)
        if link is None:
            # defaulting to log link for count models
            import warnings
            warnings.warn("using default log-link in get_prediction")
            from statsmodels.genmod.families import links
            link = links.Log()
        res = get_prediction_monotonic(
            self,
            exog=exog,
            transform=transform,
            row_labels=row_labels,
            link=link,
            pred_kwds=pred_kwds,
            index=index,
            )

    else:
        # which is not mean or linear, or we need averaging
        res = get_prediction_delta(
            self,
            exog=exog,
            which=which,
            average=average,
            agg_weights=agg_weights,
            pred_kwds=pred_kwds,
            )

    return res


def params_transform_univariate(params, cov_params, link=None, transform=None,
                                row_labels=None):
    """
    results for univariate, nonlinear, monotonicaly transformed parameters

    This provides transformed values, standard errors and confidence interval
    for transformations of parameters, for example in calculating rates with
    `exp(params)` in the case of Poisson or other models with exponential
    mean function.
    """

    from statsmodels.genmod.families import links
    if link is None and transform is None:
        link = links.Log()

    if row_labels is None and hasattr(params, 'index'):
        row_labels = params.index

    params = np.asarray(params)

    predicted_mean = link.inverse(params)
    link_deriv = link.inverse_deriv(params)
    var_pred_mean = link_deriv**2 * np.diag(cov_params)
    # TODO: do we want covariance also, or just var/se

    dist = stats.norm

    # TODO: need ci for linear prediction, method of `lin_pred
    linpred = PredictionResultsMean(
        params, np.diag(cov_params), dist=dist,
        row_labels=row_labels, link=links.Identity())

    res = PredictionResultsMean(
        predicted_mean, var_pred_mean, dist=dist,
        row_labels=row_labels, linpred=linpred, link=link)

    return res
