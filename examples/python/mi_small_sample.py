"""
Finite-sample inference with multiple imputation
=================================================

The default multiple-imputation summary uses a normal (z / chi-square)
approximation.  With only a few imputations the between-imputation
uncertainty is itself estimated imprecisely, so a finite-sample
calibration is preferable.  Passing ``inference="small_sample"`` to
:meth:`MICE.fit` / :meth:`MI.fit` switches the pooled results to:

* scalar (one-restriction) tests and confidence intervals based on the
  Barnard--Rubin adjusted degrees of freedom (Barnard & Rubin, 1999);
* joint (multi-restriction) F tests based on the
  Li--Raghunathan--Rubin pooled Wald rule ("D1"; Li, Raghunathan &
  Rubin, 1991).

The Rubin point estimates and total covariance are identical in both
modes; only the reference distribution and the per-contrast degrees of
freedom change.  Each contrast gets its own denominator degrees of
freedom because each can carry a different amount of missing
information.

The example below uses the Bayesian Gaussian imputer
(:class:`~statsmodels.imputation.bayes_mi.BayesGaussMI`) with the
:class:`~statsmodels.imputation.bayes_mi.MI` driver; the same keyword
works with :class:`~statsmodels.imputation.mice.MICE`.
"""
import numpy as np
import statsmodels.api as sm
from statsmodels.imputation.bayes_mi import BayesGaussMI, MI

# ----------------------------------------------------------------------
# Data with values missing completely at random.
# ----------------------------------------------------------------------
rng = np.random.RandomState(414)
n = 300
x = rng.normal(size=(n, 4))
x[rng.choice(n, 40, replace=False), 0] = np.nan
x[rng.choice(n, 30, replace=False), 1] = np.nan
x[rng.choice(n, 25, replace=False), 2] = np.nan


def model_args(data):
    # Regress the first column on an intercept and the other three.
    return data[:, 0], sm.add_constant(data[:, 1:])


imputer = BayesGaussMI(x.copy())
mi = MI(imputer, sm.OLS, model_args, burn=100, nrep=20, skip=10)

# ----------------------------------------------------------------------
# Finite-sample results.
# ----------------------------------------------------------------------
results = mi.fit(inference="small_sample")

print("Rubin point estimates:\n", results.params)
print("\nPer-parameter Barnard--Rubin denominator df:\n",
      results.df_denom_params)
print("\n95% confidence intervals (t based):\n", results.conf_int())

# Scalar tests: numeric matrix, (R, q) tuple, or name-based string.
t_named = results.t_test("x1 = 0")
print("\nScalar t test 'x1 = 0': t = %.3f, df = %.2f, p = %.4g"
      % (np.squeeze(t_named.statistic), np.squeeze(t_named.df_denom),
         np.squeeze(t_named.pvalue)))
print(t_named.summary_frame())

# Joint test of two coefficients at once.
R = np.array([[0, 1, 0, 0],
              [0, 0, 1, 0]])
f_test = results.f_test(R)
print("\nJoint F test: F = %.3f, df_num = %d, df_denom = %.2f, p = %.4g"
      % (float(np.squeeze(f_test.statistic)), f_test.df_num,
         f_test.df_denom, float(np.squeeze(f_test.pvalue))))
# Method and bibliographic reference are available as diagnostics.
print("Joint method:", f_test._mi_diagnostics["method"])
print("Reference:", f_test._mi_diagnostics["references"])

# A rank-one joint test is exactly the corresponding scalar t test
# squared, with the same p-value and denominator df.
f_one = results.f_test(np.array([[0, 1, 0, 0]]))
t_one = results.t_test(np.array([0, 1, 0, 0]))
assert np.isclose(float(np.squeeze(f_one.statistic)),
                  float(np.squeeze(t_one.statistic)) ** 2)

# ----------------------------------------------------------------------
# Formula models: term-level finite-sample tests.
# ----------------------------------------------------------------------
# r.wald_test_terms(skip_single=..., combine_terms=...,
#                   extra_constraints=...) reuses the formula design
# information and reports, for every term, its effective constraint
# rank, F statistic, p-value and the term-specific denominator df.

# ----------------------------------------------------------------------
# Methods, assumptions and limitations
# ----------------------------------------------------------------------
# Joint method choice.  Two moment-based combining rules appear in the
# literature.  D1 (used here) pools the parameters and covariance with
# Rubin's rules and forms one Wald quadratic form on the pooled
# contrast covariance; D2 would average the per-imputation Wald
# statistics.  D1 is preferred because it reuses the same pooled
# covariance as the scalar tests (so rank one gives F = t**2 exactly),
# keeps the between-imputation covariance structure, handles correlated
# restrictions and non-zero q, and costs only an inversion of the small
# contrast-sized covariance.  D2 discards that covariance structure, is
# conservative with few imputations, and does not collapse to the scalar
# test at rank one.  Averaging p-values is not a valid substitute for
# either rule.
#
# Scalar: Barnard--Rubin (1999) combines the classic Rubin
#   (m-1)(1 + 1/r)^2 term with a complete-data observed-df adjustment,
#   where r is the contrast-specific relative increase in variance.
#   It assumes the completed-data estimator is approximately normal
#   with a t reference on the analysis-model residual df.
# Joint:  LRR/D1 (1991) builds a Wald quadratic form from the pooled
#   Rubin covariance of the joint contrast and calibrates it against an
#   F with an effective numerator rank and a denominator df derived
#   from the joint relative increase in variance.  It needs a positive
#   definite average within-imputation covariance on the (effective)
#   constraint space.
# Both require at least two non-robust, full-rank OLS fits of the same
# model on every completed data set, with positive residual df.  Robust
# covariance, custom cov_p/invcov and df overrides are rejected so that
# the between-imputation variance cannot be silently bypassed.
