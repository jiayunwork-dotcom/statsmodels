.. currentmodule:: statsmodels.regression.linear_model


.. _regression:

Linear Regression
=================

Linear models with independently and identically distributed errors, and for
errors with heteroscedasticity or autocorrelation. This module allows
estimation by ordinary least squares (OLS), weighted least squares (WLS),
generalized least squares (GLS), and feasible generalized least squares with
autocorrelated AR(p) errors.

See `Module Reference`_ for commands and arguments.

Examples
--------

.. ipython:: python

    # Load modules and data
    import numpy as np
    import statsmodels.api as sm
    spector_data = sm.datasets.spector.load()
    spector_data.exog = sm.add_constant(spector_data.exog, prepend=False)

    # Fit and summarize OLS model
    mod = sm.OLS(spector_data.endog, spector_data.exog)
    res = mod.fit()
    print(res.summary())

Detailed examples can be found here:


* `OLS <examples/notebooks/generated/ols.ipynb>`_
* `WLS <examples/notebooks/generated/wls.ipynb>`_
* `GLS <examples/notebooks/generated/gls.ipynb>`_
* `Recursive LS <examples/notebooks/generated/recursive_ls.ipynb>`_
* `Rolling LS <examples/notebooks/generated/rolling_ls.ipynb>`_

.. _regression_cluster_bootstrap:

Inference for related observations: cluster bootstrap
-----------------------------------------------------

Regression tables often contain several related rows per school, store, or
subject.  For such data, ``OLS.fit()`` results can be calibrated with a
*pairs cluster bootstrap*: whole clusters are resampled with replacement,
the model is refit on each resample, and the empirical distribution of the
refitted coefficients replaces the t or normal reference of ``t_test`` and
``get_prediction``.  The same calibration drives linear contrasts
(including simultaneous, family-wise intervals that preserve the
dependence between contrasts) and joint confidence bands for conditional
mean predictions.

.. ipython:: python

    import numpy as np
    import statsmodels.api as sm

    rng = np.random.default_rng(20261117)
    sizes = [8, 5, 12, 7, 9, 6, 10, 8, 11, 7]
    rows, groups = [], []
    for g, s in enumerate(sizes):
        x = rng.normal(size=s)
        u = rng.normal()                      # cluster-level shock
        y = 1.0 + 0.5 * x + u + rng.normal(scale=0.5, size=s)
        rows.append(np.column_stack([np.ones(s), x, y]))
        groups += [f"school{g}"] * s
    data = np.vstack(rows)
    res = sm.OLS(data[:, 2], data[:, :2]).fit()

    cal = res.get_cluster_bootstrap(groups, n_resamples=999, random_state=0)
    print(cal)

    # linear contrasts on the empirical reference
    tt = res.t_test([[0, 1]], inference=cal)
    print(tt.summary_frame())
    # a simultaneous family of contrasts
    ts = res.t_test([[1, 0], [0, 1]], inference=cal, simultaneous=True)
    print(ts.conf_int(0.1))
    # joint confidence band for conditional means
    pred = res.get_prediction(data[:5, :2], inference=cal)
    print(pred.summary_frame(alpha=0.1, simultaneous=True))

The calibration is bound to the fitted results instance that produced it;
using it with any other fit raises an error.  Cluster labels are given in
the physical row order of the modeling input *before* any rows were dropped
by the formula or ``missing='drop'`` handling, and the same rows are dropped
from the labels.  Invalid resamples (for example rank-deficient resampled
designs) are retried a bounded number of times and reported through
``cal.n_valid``, ``cal.n_failed`` and ``cal.failure_reasons``.

The resampling unit, estimand, centering, scaling, finite-sample tail
probabilities, quantile rule, and simultaneity definition are documented in
detail in :class:`ClusterBootstrapCalibration`.  In short: the bootstrap
distribution of ``beta* - beta_hat`` approximates the sampling distribution
of ``beta_hat - beta``; two-sided p-values are
``(1 + #{|T*| >= |t_obs|}) / (B + 1)`` and never zero; intervals invert
these p-values through order statistics of the studentized bootstrap
statistics; simultaneous families use the family-wise maximum statistic,
so no Bonferroni or independence approximation is needed.  The method
requires independent clusters and is asymptotic in the number of clusters;
with few clusters, strongly unequal cluster sizes, or systematically
different cluster-level error variances, coverage can deviate from the
nominal level.

Technical Documentation
-----------------------

The statistical model is assumed to be

 :math:`Y = X\beta + \epsilon`,  where :math:`\epsilon\sim N\left(0,\Sigma\right).`

Depending on the properties of :math:`\Sigma`, we have currently four classes available:

* GLS : generalized least squares for arbitrary covariance :math:`\Sigma`
* OLS : ordinary least squares for i.i.d. errors :math:`\Sigma=\textbf{I}`
* WLS : weighted least squares for heteroskedastic errors :math:`\text{diag}\left  (\Sigma\right)`
* GLSAR : feasible generalized least squares with autocorrelated AR(p) errors
  :math:`\Sigma=\Sigma\left(\rho\right)`

All regression models define the same methods and follow the same structure,
and can be used in a similar fashion. Some of them contain additional model
specific methods and attributes.

GLS is the superclass of the other regression classes except for RecursiveLS,
RollingWLS and RollingOLS.

.. Class hierachy: TODO

.. yule_walker is not a full model class, but a function that estimate the
.. parameters of a univariate autoregressive process, AR(p). It is used in GLSAR,
.. but it can also be used independently of any models. yule_walker only
.. calculates the estimates and the standard deviation of the lag parameters but
.. not the additional regression statistics. We hope to include yule-walker in
.. future in a separate univariate time series class. A similar result can be
.. obtained with GLSAR if only the constant is included as regressors. In this
.. case the parameter estimates of the lag estimates are not reported, however
.. additional statistics, for example aic, become available.

References
^^^^^^^^^^

General reference for regression models:

* D.C. Montgomery and E.A. Peck. "Introduction to Linear Regression Analysis." 2nd. Ed., Wiley, 1992.

Econometrics references for regression models:

* R.Davidson and J.G. MacKinnon. "Econometric Theory and Methods," Oxford, 2004.
* W.Green. "Econometric Analysis," 5th ed., Pearson, 2003.

.. toctree::
..   :maxdepth: 1
..
..   regression_techn1

Attributes
^^^^^^^^^^

The following is more verbose description of the attributes which is mostly
common to all regression classes

pinv_wexog : array
    The `p` x `n` Moore-Penrose pseudoinverse of the whitened design matrix.
    It is approximately equal to
    :math:`\left(X^{T}\Sigma^{-1}X\right)^{-1}X^{T}\Psi`, where
    :math:`\Psi` is defined such that :math:`\Psi\Psi^{T}=\Sigma^{-1}`.
cholsimgainv : array
    The `n` x `n` upper triangular matrix :math:`\Psi^{T}` that satisfies
    :math:`\Psi\Psi^{T}=\Sigma^{-1}`.
df_model : float
    The model degrees of freedom. This is equal to `p` - 1, where `p` is the
    number of regressors. Note that the intercept is not counted as using a
    degree of freedom here.
df_resid : float
    The residual degrees of freedom. This is equal `n - p` where `n` is the
    number of observations and `p` is the number of parameters. Note that the
    intercept is counted as using a degree of freedom here.
llf : float
    The value of the likelihood function of the fitted model.
nobs : float
    The number of observations `n`
normalized_cov_params : array
    A `p` x `p` array equal to :math:`(X^{T}\Sigma^{-1}X)^{-1}`.
sigma : array
    The `n` x `n` covariance matrix of the error terms:
    :math:`\epsilon\sim N\left(0,\Sigma\right)`.
wexog : array
    The whitened design matrix :math:`\Psi^{T}X`.
wendog : array
    The whitened response variable :math:`\Psi^{T}Y`.

Module Reference
----------------

.. module:: statsmodels.regression.linear_model
   :synopsis: Least squares linear models

Model Classes
^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   OLS
   GLS
   WLS
   GLSAR
   yule_walker
   burg

.. module:: statsmodels.regression.quantile_regression
   :synopsis: Quantile regression

.. currentmodule:: statsmodels.regression.quantile_regression

.. autosummary::
   :toctree: generated/

   QuantReg

.. module:: statsmodels.regression.recursive_ls
   :synopsis: Recursive least squares using the Kalman Filter

.. currentmodule:: statsmodels.regression.recursive_ls

.. autosummary::
   :toctree: generated/

   RecursiveLS

.. module:: statsmodels.regression.rolling
   :synopsis: Rolling (moving) least squares

.. currentmodule:: statsmodels.regression.rolling

.. autosummary::
   :toctree: generated/

   RollingWLS
   RollingOLS

.. module:: statsmodels.regression.process_regression
   :synopsis: Process regression

.. currentmodule:: statsmodels.regression.process_regression

.. autosummary::
   :toctree: generated/

   GaussianCovariance
   ProcessMLE

.. module:: statsmodels.regression.dimred
   :synopsis: Dimension reduction methods

.. currentmodule:: statsmodels.regression.dimred

.. autosummary::
   :toctree: generated/

    SlicedInverseReg
    PrincipalHessianDirections
    SlicedAverageVarianceEstimation


Results Classes
^^^^^^^^^^^^^^^

Fitting a linear regression model returns a results class. OLS has a
specific results class with some additional methods compared to the
results class of the other linear models.

.. currentmodule:: statsmodels.regression.linear_model

.. autosummary::
   :toctree: generated/

   RegressionResults
   OLSResults
   PredictionResults
   ClusterBootstrapCalibration

.. currentmodule:: statsmodels.base.elastic_net

.. autosummary::
   :toctree: generated/

    RegularizedResults

.. currentmodule:: statsmodels.regression.quantile_regression

.. autosummary::
   :toctree: generated/

   QuantRegResults

.. currentmodule:: statsmodels.regression.recursive_ls

.. autosummary::
   :toctree: generated/

   RecursiveLSResults

.. currentmodule:: statsmodels.regression.rolling

.. autosummary::
   :toctree: generated/

   RollingRegressionResults

.. currentmodule:: statsmodels.regression.process_regression

.. autosummary::
   :toctree: generated/

   ProcessMLEResults

.. currentmodule:: statsmodels.regression.dimred

.. autosummary::
   :toctree: generated/

   DimReductionResults
