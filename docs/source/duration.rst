.. module:: statsmodels.duration
   :synopsis: Models for durations

.. currentmodule:: statsmodels.duration


.. _duration:

Methods for Survival and Duration Analysis
==========================================

:mod:`statsmodels.duration` implements several standard methods for
working with censored data.  These methods are most commonly used when
the data consist of durations between an origin time point and the
time at which some event of interest occurred.  A typical example is a
medical study in which the origin is the time at which a subject is
diagnosed with some condition, and the event of interest is death (or
disease progression, recovery, etc.).

Currently only right-censoring is handled.  Right censoring occurs
when we know that an event occurred after a given time `t`, but we do
not know the exact event time.

Survival function estimation and inference
------------------------------------------

The :class:`statsmodels.api.SurvfuncRight` class can be used to
estimate a survival function using data that may be right censored.
``SurvfuncRight`` implements several inference procedures including
confidence intervals for survival distribution quantiles, pointwise
and simultaneous confidence bands for the survival function, and
plotting procedures.  The ``duration.survdiff`` function provides
testing procedures for comparing survival distributions.

Here we create a ``SurvfuncRight`` object using data from the
`flchain` study, which is available through the R datasets repository.
We fit the survival distribution only for the female subjects.


.. ipython:: python
   :okwarning:

   import statsmodels.api as sm

   data = sm.datasets.get_rdataset("flchain", "survival").data
   df = data.loc[data.sex == "F", :]
   sf = sm.SurvfuncRight(df["futime"], df["death"])

The main features of the fitted survival distribution can be seen by
calling the ``summary`` method:

.. ipython:: python
   :okwarning:

    sf.summary().head()

We can obtain point estimates and confidence intervals for quantiles
of the survival distribution.  Since only around 30% of the subjects
died during this study, we can only estimate quantiles below the 0.3
probability point:

.. ipython:: python
   :okwarning:

    sf.quantile(0.25)
    sf.quantile_ci(0.25)

To plot a single survival function, call the ``plot`` method:

.. ipython:: python
   :okwarning:

    @savefig duration_survival_plot1.png
    sf.plot()

Since this is a large dataset with a lot of censoring, we may wish
to not plot the censoring symbols:

.. ipython:: python
   :okwarning:

    fig = sf.plot()
    ax = fig.get_axes()[0]
    pt = ax.get_lines()[1]
    @savefig duration_survival_nocensor_plot.png
    pt.set_visible(False)

We can also add a 95% simultaneous confidence band to the plot.
Typically these bands only plotted for central part of the
distribution.

.. ipython:: python
   :okwarning:

    fig = sf.plot()
    lcb, ucb = sf.simultaneous_cb()
    ax = fig.get_axes()[0]
    ax.fill_between(sf.surv_times, lcb, ucb, color='lightgrey')
    ax.set_xlim(365, 365*10)
    ax.set_ylim(0.7, 1)
    ax.set_ylabel("Proportion alive")
    @savefig duration_survival_95ci_plot.png
    ax.set_xlabel("Days since enrollment")

Here we plot survival functions for two groups (females and males) on
the same axes:

.. ipython:: python
   :okwarning:

   import matplotlib.pyplot as plt

   gb = data.groupby("sex")
   ax = plt.axes()
   sexes = []
   for g in gb:
       sexes.append(g[0])
       sf = sm.SurvfuncRight(g[1]["futime"], g[1]["death"])
       sf.plot(ax)
   li = ax.get_lines()
   li[1].set_visible(False)
   li[3].set_visible(False)
   plt.figlegend((li[0], li[2]), sexes, loc="center right")
   plt.ylim(0.6, 1)
   ax.set_ylabel("Proportion alive")
   @savefig duration_survival_bysex_plot.png
   ax.set_xlabel("Days since enrollment")

We can formally compare two survival distributions with ``survdiff``,
which implements several standard nonparametric procedures.  The
default procedure is the logrank test:

.. ipython:: python
   :okwarning:

    stat, pv = sm.duration.survdiff(data.futime, data.death, data.sex)

Here are some of the other testing procedures implemented by survdiff:

.. ipython:: python
   :okwarning:

    # Fleming-Harrington with p=1, i.e. weight by pooled survival time
    stat, pv = sm.duration.survdiff(data.futime, data.death, data.sex, weight_type='fh', fh_p=1)

    # Gehan-Breslow, weight by number at risk
    stat, pv = sm.duration.survdiff(data.futime, data.death, data.sex, weight_type='gb')

    # Tarone-Ware, weight by the square root of the number at risk
    stat, pv = sm.duration.survdiff(data.futime, data.death, data.sex, weight_type='tw')


Regression methods
------------------

Proportional hazard regression models ("Cox models") are a regression
technique for censored data.  They allow variation in the time to an
event to be explained in terms of covariates, similar to what is done
in a linear or generalized linear regression model.  These models
express the covariate effects in terms of "hazard ratios", meaning the
the hazard (instantaneous event rate) is multiplied by a given factor
depending on the value of the covariates.


.. ipython:: python
   :okexcept:
   :okwarning:

   import statsmodels.api as sm
   import statsmodels.formula.api as smf

   data = sm.datasets.get_rdataset("flchain", "survival").data
   del data["chapter"]
   data = data.dropna()
   data["lam"] = data["lambda"]
   data["female"] = (data["sex"] == "F").astype(int)
   data["year"] = data["sample.yr"] - min(data["sample.yr"])
   status = data["death"].values
   mod = smf.phreg("futime ~ 0 + age + female + creatinine + np.sqrt(kappa) + np.sqrt(lam) + year + mgus", data, status=status, ties="efron")
   rslt = mod.fit()
   print(rslt.summary())


Competing risks and interval failure probabilities
---------------------------------------------------

A device (or patient) may leave the risk set through more than one
cause.  Fitting one :class:`PHReg` model per failure cause is not
enough on its own: the separate single-outcome survival curves do not
account for the competing causes and cannot be interpreted as the
marginal probability of each failure.  ``CumIncidenceRight`` combines
the cause-specific Cox fits into proper cumulative incidence
probabilities, and answers *interval* questions of the form
"conditional on running failure-free up to an inspection time
``origin``, what is the probability of failing from each cause, or of
still running, by each future query time?".

The cause-specific models use the usual analysis table with one row
per subject, the same exit time, entry time, strata, numeric design
matrix and offset in every model; a subject that fails from one cause
is right censored at its exit time in all the other cause models.
Efron or Breslow tie handling, delayed entry, stratification and
offsets are all supported.  Only unregularized fits are accepted.

.. ipython:: python
   :okexcept:
   :okwarning:

   import numpy as np
   from statsmodels.duration.hazard_regression import PHReg
   from statsmodels.duration.survfunc import CumIncidenceRight

   rng = np.random.default_rng(0)
   n = 200
   x = rng.normal(size=(n, 2))
   t1 = -np.log(rng.uniform(size=n)) / np.exp(x @ [0.5, -0.3])
   t2 = -np.log(rng.uniform(size=n)) / np.exp(x @ [-0.2, 0.4]) * 2
   tc = rng.uniform(0.5, 6, size=n)
   time = np.minimum(np.minimum(t1, t2), tc)
   status = np.where(t1 <= np.minimum(t2, tc), 1,
                     np.where(t2 <= tc, 2, 0))
   fit_motor = PHReg(time, x, (status == 1).astype(float)).fit()
   fit_sensor = PHReg(time, x, (status == 2).astype(float)).fit()
   ci = CumIncidenceRight.from_phreg(
       {"motor": fit_motor, "sensor": fit_sensor})

The query takes a finite non-negative ``origin`` that is assumed to
have been survived failure-free, and a non-decreasing sequence of
future query times (repeated times and times equal to the origin are
allowed).  In fitted mode it also takes the prediction design matrix,
one stratum per row and optional prediction offsets.

.. ipython:: python
   :okexcept:
   :okwarning:

   query = ci.predict(1.0, np.array([1.0, 2.0, 3.0, 5.0]),
                      exog=x[:3], strata=np.zeros(3, dtype=int))
   query

Each row sums to one.  Cause columns are the conditional probability
of failure from that cause within ``(origin, time]``; the
``"__failure_free__"`` column is ``P(T > time | T > origin)``.  Cause
probabilities are non-decreasing and the failure-free probability is
non-increasing in time.  Splitting an interval at an intermediate
inspection reproduces the whole-interval probabilities:

.. math::

   P_j(t_2) = P_j(t_1) + S(t_1)\,P_j(t_2 \mid t_1), \qquad
   S(t_2) = S(t_1)\,S(t_2 \mid t_1).

Post-event hazard convention
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``PHReg.baseline_cumulative_hazard`` and the ``surv`` / ``cumhaz``
predictions default to the historical left-continuous convention, in
which the value at an event time excludes the hazard jump at that
time.  Passing ``post_event=True`` selects the right-continuous,
post-event convention used throughout the competing-risk calculations:
the cumulative hazard is zero before the first event, includes the
jump at each event time, and carries the final jump forward.  With a
single cause, the competing-risk failure-free probability reduces
exactly to the ratio of post-event PHReg survival predictions
``S(t) / S(origin)``.

Same-day competing events
^^^^^^^^^^^^^^^^^^^^^^^^^^

When several causes have hazard jumps at the same time, the survival
probability lost on that day is apportioned across causes in
proportion to their scaled hazard jumps.  Equal jumps therefore
receive equal probability, a cause with a zero jump receives no
probability, and the apportioned masses plus the remaining survival
sum exactly to one.  This is the standard Aalen-Johansen rule and
implicitly assumes that, within a day, the tied cause-specific
hazards act as independent constant hazards over infinitesimal
orderings of the tied events.  If event times are coarsened (for
example recorded only to the day when the true ordering is finer),
the within-day split is not separately identifiable from the data;
recording times on a finer grid changes which events share a day and
hence changes the apportionment, though the total daily event mass
is unaffected.

Empirical curves
^^^^^^^^^^^^^^^^

A ``CumIncidenceRight`` built directly from raw ``time`` and
``status`` arrays (the empirical Aalen-Johansen estimator) supports
the same ``predict`` query.  This first version supports only the
unweighted estimator without ``exog`` adjustment; frequency-weighted
or kernel-adjusted curves raise an error when queried.  A query whose
origin has no remaining survivable probability mass is also rejected,
because the requested conditional probabilities are then undefined.

.. ipython:: python
   :okexcept:
   :okwarning:

   from statsmodels.duration.survfunc import CumIncidenceRight as CIR
   emp = CIR(time, status)
   emp.predict(0.0, np.array([1.0, 3.0, 6.0]))


See :ref:`statsmodels-examples` for more detailed examples.


There are some notebook examples on the Wiki:
`Wiki notebooks for PHReg and Survival Analysis <https://github.com/statsmodels/statsmodels/wiki/Examples#survival-analysis>`_


.. todo::

   Technical Documentation

References
^^^^^^^^^^

References for Cox proportional hazards regression model::

    T Therneau (1996). Extending the Cox model. Technical report.
    http://www.mayo.edu/research/documents/biostat-58pdf/DOC-10027288

    G Rodriguez (2005). Non-parametric estimation in survival models.
    http://data.princeton.edu/pop509/NonParametricSurvival.pdf

    B Gillespie (2006). Checking the assumptions in the Cox proportional
    hazards model.
    http://www.mwsug.org/proceedings/2006/stats/MWSUG-2006-SD08.pdf


Module Reference
----------------

.. module:: statsmodels.duration.survfunc
   :synopsis: Models for Survival Analysis

.. currentmodule:: statsmodels.duration.survfunc

The class for working with survival distributions is:

.. autosummary::
   :toctree: generated/

   SurvfuncRight
   CumIncidenceRight

.. module:: statsmodels.duration.hazard_regression
   :synopsis: Proportional hazards model for Survival Analysis

.. currentmodule:: statsmodels.duration.hazard_regression

The proportional hazards regression model class is:

.. autosummary::
   :toctree: generated/

   PHReg

The proportional hazards regression result class is:

.. autosummary::
   :toctree: generated/

   PHRegResults

The primary helper class is:

.. autosummary::
   :toctree: generated/

   rv_discrete_float
