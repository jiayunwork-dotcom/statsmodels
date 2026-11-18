:orphan:

====================
Release 0.14.6 (dev)
====================

Release summary
===============

This release adds finite-sample multiple-imputation inference to the
:mod:`statsmodels.imputation` analysis chains.

The Highlights
==============

Multiple imputation now supports a finite-sample inference mode in
addition to the default normal approximation.  Pass
``inference="small_sample"`` (a keyword-only argument of
``MICE.fit``, ``MICE.combine`` and ``MI.fit``) to obtain:

- Barnard--Rubin adjusted degrees of freedom for scalar linear
  contrasts, including the individual parameter tests and confidence
  intervals.  Each contrast uses its own denominator degrees of
  freedom because contrasts can carry different amounts of missing
  information.
- Li--Raghunathan--Rubin ("D1") pooled Wald F tests for joint linear
  hypotheses through ``wald_test`` and ``f_test``.  A rank-one joint
  test is exactly the corresponding scalar test squared, with the same
  p-value and denominator degrees of freedom.
- Formula ``wald_test_terms`` integration reporting each term's
  effective constraint rank and term-specific denominator degrees of
  freedom.

The Rubin point estimates and total covariance are unchanged from the
legacy normal-approximation results; only the reference distribution
and degrees of freedom differ.  Parameters are aligned by name across
imputations, incompatible fits raise a located ``ValueError``, and the
inference object retains only compact pooled summaries rather than the
completed data sets or full regression objects.

The small-sample mode currently requires at least two non-robust, full
column rank OLS fits of the same analysis model, each with positive
residual degrees of freedom and finite real inputs.  Robust
covariance, custom covariance or inverse-covariance overrides, degrees
of freedom overrides and ``use_t=False`` are explicitly rejected in
this mode.

The previously incomplete linear-contrast metadata on the legacy
multiple-imputation result objects has also been repaired so that
``t_test`` / ``f_test`` / ``wald_test`` resolve parameter names and
residual degrees of freedom.

References
----------

Barnard, J. and Rubin, D.B. (1999).  Small-sample degrees of freedom
with multiple imputation.  *Biometrika* 86(4), 948-955.

Li, K.-H., Raghunathan, T.E. and Rubin, D.B. (1991).  Large-sample
significance levels from multiply imputed data using moment-based
statistics and an F reference distribution.  *JASA* 86(416),
1065-1073.
