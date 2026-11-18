.. module:: statsmodels.imputation.mice
   :synopsis: Multiple imputation for missing data

.. currentmodule:: statsmodels.imputation.mice

.. _imputation:


Multiple Imputation with Chained Equations
==========================================

The MICE module allows most statsmodels models to be fit to a dataset
with missing values on the independent and/or dependent variables, and
provides rigorous standard errors for the fitted parameters.  The
basic idea is to treat each variable with missing values as the
dependent variable in a regression, with some or all of the remaining
variables as its predictors.  The MICE procedure cycles through these
models, fitting each in turn, then uses a procedure called "predictive
mean matching" (PMM) to generate random draws from the predictive
distributions determined by the fitted models.  These random draws
become the imputed values for one imputed data set.

By default, each variable with missing variables is modeled using a
linear regression with main effects for all other variables in the
data set.  Note that even when the imputation model is linear, the PMM
procedure preserves the domain of each variable.  Thus, for example,
if all observed values for a given variable are positive, all imputed
values for the variable will always be positive.  The user also has
the option to specify which model is used to produce imputed values
for each variable.

.. code


Classes
-------

.. currentmodule:: statsmodels.imputation.mice

.. autosummary::
   :toctree: generated/

   MICE
   MICEData

.. currentmodule:: statsmodels.imputation.bayes_mi

.. autosummary::
   :toctree: generated/

   MI
   BayesGaussMI


Finite-sample inference
-----------------------

The pooled results use a normal (z / chi-square) reference distribution
by default.  When only a few imputations are available the
between-imputation variance is itself estimated imprecisely, so a
finite-sample calibration is more appropriate.  Passing
``inference="small_sample"`` (a keyword-only argument of
:meth:`MICE.fit`, :meth:`MICE.combine` and :meth:`MI.fit`) enables:

* **Scalar contrasts** (each individual coefficient and any single
  linear restriction) use the Barnard--Rubin adjusted degrees of
  freedom (Barnard and Rubin, 1999).  Every contrast receives its own
  denominator degrees of freedom because different contrasts can carry
  different amounts of missing information.  ``params``, ``bse``,
  ``pvalues``, ``conf_int`` and ``summary`` are all aligned with the
  unit-contrast ``t_test``, and the parameter table reports the degrees
  of freedom actually used.

* **Joint contrasts** (several restrictions tested together, via
  ``wald_test`` / ``f_test``) use the Li--Raghunathan--Rubin pooled
  Wald rule, often called the D1 statistic (Li, Raghunathan and Rubin,
  1991).  The F statistic uses the effective rank of the restrictions
  as its numerator degrees of freedom and a denominator derived from
  the joint relative increase in variance.  A rank-one joint test is
  exactly the corresponding scalar test squared (``F = t**2``) with the
  same p-value and denominator degrees of freedom.

Why D1 rather than D2?
~~~~~~~~~~~~~~~~~~~~~~

Two moment-based combining rules for joint hypotheses are available in
the multiple-imputation literature.  D1 pools the parameters and their
covariance with Rubin's rules and forms one Wald quadratic form on the
pooled contrast covariance.  D2 instead averages the per-imputation
completed-data Wald statistics and compares the average with an adjusted
chi-square/F reference.

D1 is used here because it works from the same pooled covariance as the
scalar contrasts, so a rank-one joint test coincides exactly with the
Barnard--Rubin t test; it preserves the between-imputation covariance
structure and supports correlated restrictions and non-zero targets
``q``.  D2 needs only the per-imputation statistics but discards that
covariance structure, is typically conservative and poorly calibrated
with few imputations, and does not reduce to the scalar test at rank
one.  Averaging p-values is not a valid version of either rule and is
not used.  D1 assumes approximately normal completed-data estimators and
a positive-definite average within-imputation covariance on the
effective constraint space; its F calibration is a finite-sample
approximation, and its only extra cost is inverting the small
contrast-sized pooled covariance rather than re-fitting any model.

The Rubin point estimate and total covariance are identical in both
modes; only the reference distribution and the degrees of freedom
change.  The method names and bibliographic references are attached to
the test objects (``_mi_diagnostics``) and shown in the summary.

The small-sample mode currently requires at least two non-robust, full
column rank OLS fits of the same analysis model, each with positive
residual degrees of freedom and finite real inputs.  Parameters are
aligned by name across imputations.  When the residual degrees of
freedom are not identical (or are unavailable) across fits, supply the
common complete-data degrees of freedom explicitly with
``df_complete``; ``df_complete=np.inf`` selects the large-sample
normal/chi-square limit.  Supplying ``cov_p`` / ``invcov``, overriding
the degrees of freedom, or setting ``use_t=False`` is rejected in this
mode so that the between-imputation uncertainty cannot be bypassed.

Term-level tests of formula models remain available through
``wald_test_terms`` (with ``skip_single``, ``combine_terms`` and
``extra_constraints``); the table reports each term's effective
constraint rank and its own denominator degrees of freedom.

References
~~~~~~~~~~

Barnard, J. and Rubin, D.B. (1999).  Small-sample degrees of freedom
with multiple imputation.  *Biometrika* 86(4), 948-955.

Li, K.-H., Raghunathan, T.E. and Rubin, D.B. (1991).  Large-sample
significance levels from multiply imputed data using moment-based
statistics and an F reference distribution.  *JASA* 86(416),
1065-1073.

Rubin, D.B. (1987).  *Multiple Imputation for Nonresponse in Surveys*.
Wiley.

van Buuren, S. (2018).  *Flexible Imputation of Missing Data*, 2nd ed.
Chapman & Hall/CRC.


Implementation Details
----------------------

Internally, this function uses
`pandas.isnull <https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html#working-with-missing-data>`_.
Anything that returns True from this function will be treated as missing data.
