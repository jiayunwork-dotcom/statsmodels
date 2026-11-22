"""Tests for exposure and offset handling in count model prediction.

The tests cover the model ``predict`` methods, the results ``predict``
methods and ``get_prediction`` for GLM with a Poisson family and for the
discrete ``Poisson`` model.

All expected values are computed independently of the code under test, from
the design matrix, an independent Poisson maximum likelihood fit and the
Poisson pmf.  The formulas are written out explicitly so that a bug in the
shared prediction layer cannot leak into the expectations.
"""

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy import stats

import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Poisson
from statsmodels.genmod.generalized_linear_model import GLM
import statsmodels.formula.api as smf


RTOL = 1e-7
ATOL = 1e-10

# Artificial data used to reproduce the original inconsistency.
X_BASE = np.column_stack([np.ones(12), np.arange(12.) / 5])
Y_BASE = np.array([1, 3, 2, 5, 4, 8, 7, 13, 15, 19, 22, 32], dtype=float)
EXPOSURE_BASE = np.linspace(1, 3, 12)
OFFSET_BASE = np.linspace(-.5, .5, 12)

# Independent Poisson MLE for the data above; computed with a plain Newton
# iteration on the log-likelihood (see the module docstring).
PARAMS_BASE = np.array([0.8893011815, 0.4243088657])
COVB_BASE = np.array([[0.0767775536, -0.0421688135],
                      [-0.0421688135, 0.0257174713]])


def _fit_glm(exposure=EXPOSURE_BASE, offset=OFFSET_BASE, **kwds):
    return GLM(Y_BASE, X_BASE, family=sm.families.Poisson(),
               exposure=exposure, offset=offset).fit(**kwds)


def _fit_poisson(exposure=EXPOSURE_BASE, offset=OFFSET_BASE, **kwds):
    return Poisson(Y_BASE, X_BASE, exposure=exposure,
                   offset=offset).fit(disp=0, **kwds)


def _base_terms(offset=OFFSET_BASE, exposure=EXPOSURE_BASE):
    eta = (X_BASE @ PARAMS_BASE + np.asarray(offset) +
           np.log(np.asarray(exposure)))
    return eta, np.exp(eta)


def _linpred_se(X, covb):
    return np.sqrt(np.einsum("ij,jk,ik->i", X, covb, X))


# ---------------------------------------------------------------------------
# Reference quantities computed from the independent fit
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def glm_results():
    return _fit_glm()


@pytest.fixture(scope="module")
def poisson_results():
    return _fit_poisson()


def test_reference_fit_matches():
    # the fitted coefficients and covariance must match the independent fit
    res = _fit_poisson()
    assert_allclose(res.params, PARAMS_BASE, rtol=RTOL, atol=ATOL)
    assert_allclose(res.cov_params(), COVB_BASE, rtol=1e-6, atol=1e-9)


# ---------------------------------------------------------------------------
# Default sources: fitted exposure and offset are used
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", ["glm_results", "poisson_results"])
@pytest.mark.parametrize("which", ["mean", "linear"])
def test_default_sources(request, fixture, which):
    res = request.getfixturevalue(fixture)
    eta, mu = _base_terms()
    expected = mu if which == "mean" else eta
    expected_se = (mu if which == "mean" else 1.) * _linpred_se(
        X_BASE, res.cov_params())

    # model.predict, results.predict and get_prediction all agree
    model_pred = res.model.predict(res.params, which=which)
    res_pred = res.predict(which=which)
    assert_allclose(model_pred, expected, rtol=RTOL, atol=ATOL)
    assert_allclose(res_pred, expected, rtol=RTOL, atol=ATOL)

    if which == "mean":
        pred = res.get_prediction(which="mean")
    else:
        pred = res.get_prediction(which="linear")
    assert_allclose(pred.predicted, expected, rtol=RTOL, atol=ATOL)
    assert_allclose(pred.se, expected_se, rtol=RTOL, atol=ATOL)
    assert_allclose(pred.summary_frame()["predicted"], expected,
                    rtol=RTOL, atol=ATOL)


def test_glm_legacy_get_prediction(glm_results):
    # the backwards compatible which=None result must use the same sources
    eta, mu = _base_terms()
    res = glm_results
    pred = res.get_prediction()
    se_mean = mu * _linpred_se(X_BASE, res.cov_params())
    se_lin = _linpred_se(X_BASE, res.cov_params())

    assert_allclose(pred.predicted_mean, mu, rtol=RTOL, atol=ATOL)
    assert_allclose(pred.var_pred_mean, se_mean**2, rtol=RTOL, atol=ATOL)
    assert_allclose(pred.linpred.predicted_mean, eta, rtol=RTOL, atol=ATOL)
    assert_allclose(pred.linpred.se_mean, se_lin, rtol=RTOL, atol=ATOL)
    frame = pred.summary_frame()
    assert_allclose(frame["mean"], mu, rtol=RTOL, atol=ATOL)
    assert_allclose(frame["mean_se"], se_mean, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# Explicit exog drops the fitted exposure and offset
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", ["glm_results", "poisson_results"])
def test_explicit_training_exog_drops_fitted_terms(request, fixture):
    res = request.getfixturevalue(fixture)
    mu = np.exp(X_BASE @ PARAMS_BASE)
    se = mu * _linpred_se(X_BASE, res.cov_params())

    for pred in [
        res.model.predict(res.params, exog=X_BASE),
        res.predict(exog=X_BASE),
        res.get_prediction(exog=X_BASE).predicted,
        res.get_prediction(which="mean", exog=X_BASE).predicted,
    ]:
        assert_allclose(pred, mu, rtol=RTOL, atol=ATOL)

    pred = res.get_prediction(exog=X_BASE)
    assert_allclose(pred.se, se, rtol=RTOL, atol=ATOL)
    # passing the training data explicitly gives the old, wrong-with-defaults
    # value and must differ from the default prediction
    assert not np.allclose(mu, _base_terms()[1])


@pytest.mark.parametrize("fixture", ["glm_results", "poisson_results"])
def test_explicit_new_exog_linear_drops_terms(request, fixture):
    res = request.getfixturevalue(fixture)
    new_exog = X_BASE[:5] + np.array([0., .1])
    eta = new_exog @ PARAMS_BASE
    mu = np.exp(eta)
    pred = res.get_prediction(exog=new_exog, which="linear")
    assert_allclose(pred.predicted, eta, rtol=RTOL, atol=ATOL)
    pred_mean = res.get_prediction(exog=new_exog)
    assert_allclose(pred_mean.predicted, mu, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# Single term overrides
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", ["glm_results", "poisson_results"])
def test_single_override_keeps_other_fitted(request, fixture):
    res = request.getfixturevalue(fixture)
    # explicit exposure, fitted offset remains because exog is not supplied.
    # The explicit exposure *replaces* the fitted exposure, it does not
    # multiply it.
    c = 2.5
    eta = X_BASE @ PARAMS_BASE + OFFSET_BASE + np.log(c)
    mu = np.exp(eta)
    for pred in [
        res.model.predict(res.params, exposure=c),
        res.predict(exposure=c),
        res.get_prediction(exposure=c).predicted,
    ]:
        assert_allclose(pred, mu, rtol=RTOL, atol=ATOL)

    # explicit offset, fitted exposure remains
    offset = OFFSET_BASE + 0.3
    eta = X_BASE @ PARAMS_BASE + offset + np.log(EXPOSURE_BASE)
    mu = np.exp(eta)
    assert_allclose(res.predict(offset=offset), mu, rtol=RTOL, atol=ATOL)

    # both explicit
    eta = X_BASE @ PARAMS_BASE + np.log(2.0) + 0.25
    mu = np.exp(eta)
    pred = res.predict(exposure=2.0, offset=0.25)
    assert_allclose(pred, mu, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("fixture", ["glm_results", "poisson_results"])
def test_explicit_exog_and_single_override(request, fixture):
    # exog supplied invalidates the fitted offset; exposure then must be
    # given explicitly to get the same value as the default prediction
    res = request.getfixturevalue(fixture)
    eta = X_BASE @ PARAMS_BASE + OFFSET_BASE + np.log(EXPOSURE_BASE)
    mu = np.exp(eta)
    # Note: offset must be passed (with exog) because exog resets it.
    pred = res.predict(exog=X_BASE, exposure=EXPOSURE_BASE,
                       offset=OFFSET_BASE)
    assert_allclose(pred, mu, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# Units: model storage must not change the public scaling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", ["glm_results", "poisson_results"])
def test_exposure_scaling(request, fixture):
    res = request.getfixturevalue(fixture)
    c = 2.5
    base = res.get_prediction()
    scaled = res.get_prediction(exposure=c * EXPOSURE_BASE)
    assert_allclose(scaled.predicted, c * base.predicted, rtol=RTOL, atol=ATOL)
    assert_allclose(scaled.se, c * base.se, rtol=RTOL, atol=ATOL)

    base_lin = res.get_prediction(which="linear")
    scaled_lin = res.get_prediction(which="linear", exposure=c * EXPOSURE_BASE)
    assert_allclose(scaled_lin.predicted, base_lin.predicted + np.log(c),
                    rtol=RTOL, atol=ATOL)
    assert_allclose(scaled_lin.se, base_lin.se, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("fixture", ["glm_results", "poisson_results"])
def test_exposure_equals_offset(request, fixture):
    # with explicit exog no fitted terms apply, so an equal amount of
    # exposure can be written as exposure=c or as offset=log(c)
    res = request.getfixturevalue(fixture)
    c = 1.7
    pe = res.get_prediction(exog=X_BASE, exposure=np.full(12, c))
    po = res.get_prediction(exog=X_BASE, offset=np.full(12, np.log(c)))
    assert_allclose(pe.predicted, po.predicted, rtol=RTOL, atol=ATOL)
    assert_allclose(pe.se, po.se, rtol=RTOL, atol=ATOL)


def test_model_storage_units():
    # GLM stores log exposure, discrete Poisson stores log exposure too, but
    # predict must still accept raw exposure and produce equal results.
    glm = _fit_glm()
    poisson = _fit_poisson()
    assert_allclose(glm.model.exposure, np.log(EXPOSURE_BASE), rtol=1e-13)
    assert_allclose(poisson.model.exposure, np.log(EXPOSURE_BASE), rtol=1e-13)
    assert_allclose(glm.predict(), poisson.predict(), rtol=RTOL, atol=ATOL)
    assert_allclose(glm.predict(exposure=np.ones(12)),
                    poisson.predict(exposure=np.ones(12)),
                    rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# Averaging
# ---------------------------------------------------------------------------


AGG_WEIGHTS = np.linspace(0.5, 2, 12)
AGG_WEIGHTS = AGG_WEIGHTS / AGG_WEIGHTS.mean()


@pytest.mark.parametrize("fixture", ["glm_results", "poisson_results"])
@pytest.mark.parametrize("which", ["mean", "linear"])
def test_average_prediction(request, fixture, which):
    res = request.getfixturevalue(fixture)
    eta, mu = _base_terms()
    if which == "mean":
        row = mu
        # derivative of the mean is mu * x; the average is weighted by weights
        # that are not normalized, then divided by the number of rows.
        g = ((row * AGG_WEIGHTS)[:, None] * X_BASE).mean(axis=0)
        expected = (row * AGG_WEIGHTS).mean()
    else:
        row = eta
        g = (AGG_WEIGHTS[:, None] * X_BASE).mean(axis=0)
        expected = (row * AGG_WEIGHTS).mean()

    pred = res.get_prediction(which=which, average=True,
                              agg_weights=AGG_WEIGHTS)
    se = np.sqrt(g @ res.cov_params() @ g)
    assert_allclose(pred.predicted, expected, rtol=RTOL, atol=ATOL)
    assert_allclose(pred.se, se, rtol=RTOL, atol=ATOL)
    # the uncertainty is not the mean of the per-row standard errors
    assert not np.allclose(pred.se, res.get_prediction(which=which).se.mean())

    ci = pred.conf_int()
    z = stats.norm.ppf(0.975)
    assert_allclose(ci[0, 0], expected - z * se, rtol=RTOL, atol=ATOL)
    assert_allclose(ci[0, 1], expected + z * se, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("fixture", ["glm_results", "poisson_results"])
def test_average_constant_weight(request, fixture):
    # constant aggregation weights: not normalized, so result is scaled
    res = request.getfixturevalue(fixture)
    _, mu = _base_terms()
    pred = res.get_prediction(which="mean", average=True,
                              agg_weights=np.full(12, 1.25))
    assert_allclose(pred.predicted, 1.25 * mu.mean(), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("fixture", ["glm_results", "poisson_results"])
def test_average_reorder_and_duplicate(request, fixture):
    res = request.getfixturevalue(fixture)
    _, mu = _base_terms()
    pred = res.get_prediction(which="mean", average=True,
                              agg_weights=AGG_WEIGHTS)
    idx = np.array([5, 2, 9, 0, 11, 4, 7, 1, 8, 3, 10, 6])

    permuted = res.get_prediction(
        exog=X_BASE[idx], which="mean", average=True,
        agg_weights=AGG_WEIGHTS[idx], exposure=EXPOSURE_BASE[idx],
        offset=OFFSET_BASE[idx])
    assert_allclose(permuted.predicted, pred.predicted, rtol=RTOL, atol=ATOL)
    assert_allclose(permuted.se, pred.se, rtol=RTOL, atol=ATOL)

    # duplicating rows with the same weights does not change the average,
    # because the aggregation is a weighted sum divided by the row count
    dup_exog = np.vstack([X_BASE, X_BASE])
    duplicated = res.get_prediction(
        exog=dup_exog, which="mean", average=True,
        agg_weights=np.tile(AGG_WEIGHTS, 2),
        exposure=np.tile(EXPOSURE_BASE, 2),
        offset=np.tile(OFFSET_BASE, 2))
    assert_allclose(duplicated.predicted, pred.predicted, rtol=RTOL, atol=ATOL)
    assert_allclose(duplicated.se, pred.se, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("fixture", ["glm_results", "poisson_results"])
def test_average_matches_subset_mean(request, fixture):
    # equal weights on a subset equal the plain mean over that subset
    res = request.getfixturevalue(fixture)
    n = 12
    aw = np.zeros(n)
    aw[:5] = 1
    aw /= aw.mean()
    pred = res.get_prediction(which="mean", average=True, agg_weights=aw)
    subset = res.predict(exog=X_BASE[:5], exposure=EXPOSURE_BASE[:5],
                         offset=OFFSET_BASE[:5])
    assert_allclose(pred.predicted, subset.mean(), rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# Poisson probabilities
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("average", [False, True])
def test_prob_prediction(average):
    res = _fit_poisson()
    counts = np.arange(3)
    _, mu = _base_terms()
    probs = stats.poisson._pmf(counts[None, :], mu[:, None])
    # independently computed with the scipy distribution function
    probs_scipy = stats.poisson.pmf(counts[None, :], mu[:, None])
    assert_allclose(probs, probs_scipy, rtol=RTOL, atol=ATOL)

    grad = (probs[:, :, None] * (counts[None, :] - mu[:, None])[:, :, None] *
            X_BASE[:, None, :])

    kwds = {"agg_weights": AGG_WEIGHTS} if average else {}
    pred = res.get_prediction(which="prob", y_values=counts,
                              average=average, **kwds)
    if average:
        expected = (probs * AGG_WEIGHTS[:, None]).mean(axis=0)
        g = (grad * AGG_WEIGHTS[:, None, None]).mean(axis=0)
        se = np.sqrt(np.einsum("kp,pq,kq->k", g, res.cov_params(), g))
    else:
        expected = probs
        var = np.einsum("nkp,pq,nkq->nk", grad, res.cov_params(), grad)
        se = np.sqrt(var)
    assert_allclose(pred.predicted, expected, rtol=RTOL, atol=ATOL)
    assert_allclose(pred.se, se, rtol=RTOL, atol=ATOL)


def test_prob_direct_matches_get_prediction():
    res = _fit_poisson()
    counts = np.arange(4)
    direct = res.model.predict(res.params, which="prob", y_values=counts)
    pred = res.get_prediction(which="prob", y_values=counts)
    assert_allclose(pred.predicted, direct, rtol=RTOL, atol=ATOL)


def test_prob_zero_count_and_large_mean():
    # a sparse mean and a large mean, both cross checked with the scipy pmf
    res = _fit_poisson()
    counts = np.array([0, 1, 5])
    _, mu = _base_terms()
    probs = stats.poisson.pmf(counts[None, :], mu[:, None])
    pred = res.get_prediction(exog=X_BASE, exposure=EXPOSURE_BASE,
                              offset=OFFSET_BASE, which="prob",
                              y_values=counts)
    assert_allclose(pred.predicted, probs, rtol=RTOL, atol=ATOL)
    assert np.all(pred.se[0] > 0)

    # large mean: increase the exposure dramatically
    big = res.get_prediction(which="prob", y_values=counts,
                             exposure=50 * EXPOSURE_BASE)
    big_probs = stats.poisson.pmf(counts[None, :],
                                  (mu * 50)[:, None])
    assert_allclose(big.predicted, big_probs, rtol=1e-6, atol=1e-12)


def test_prob_uses_resolved_exposure():
    # probability predictions with an explicit exposure must use it.
    res = _fit_poisson()
    counts = np.arange(3)
    _, mu = _base_terms()
    scale = 2.
    pred = res.get_prediction(which="prob", y_values=counts, exposure=scale)
    eta = X_BASE @ PARAMS_BASE + OFFSET_BASE + np.log(scale)
    expected = stats.poisson.pmf(counts[None, :], np.exp(eta)[:, None])
    assert_allclose(pred.predicted, expected, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------


BAD_EXPOSURE = [0, -1.0, np.nan, np.inf, 1j, np.ones((2, 2)), np.ones(3)]
BAD_OFFSET = [np.nan, np.inf, 1j, np.ones((2, 2)), np.ones(3)]


@pytest.mark.parametrize("fixture", ["glm_results", "poisson_results"])
@pytest.mark.parametrize("bad", BAD_EXPOSURE)
def test_invalid_exposure(request, fixture, bad):
    res = request.getfixturevalue(fixture)
    calls = [
        lambda: res.model.predict(res.params, exposure=bad),
        lambda: res.predict(exposure=bad),
        lambda: res.get_prediction(exposure=bad),
        lambda: res.get_prediction(which="mean", exposure=bad),
        lambda: res.get_prediction(which="linear", exposure=bad),
        lambda: res.get_prediction(which="mean", average=True,
                                   exposure=bad),
    ]
    for call in calls:
        with pytest.raises(ValueError):
            call()
    # a failed request must not break later calls
    assert_allclose(res.predict(), _base_terms()[1], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("fixture", ["glm_results", "poisson_results"])
@pytest.mark.parametrize("bad", BAD_OFFSET)
def test_invalid_offset(request, fixture, bad):
    res = request.getfixturevalue(fixture)
    calls = [
        lambda: res.model.predict(res.params, offset=bad),
        lambda: res.predict(offset=bad),
        lambda: res.get_prediction(offset=bad),
        lambda: res.get_prediction(which="mean", offset=bad),
        lambda: res.get_prediction(which="linear", offset=bad),
    ]
    for call in calls:
        with pytest.raises(ValueError):
            call()


def test_invalid_prob_exposure():
    res = _fit_poisson()
    with pytest.raises(ValueError):
        res.get_prediction(which="prob", exposure=0)
    with pytest.raises(ValueError):
        res.get_prediction(which="prob", exposure=np.ones(3))


def test_valid_offset_signs():
    # offset may be zero or negative; exposure may not
    res = _fit_poisson()
    _, mu = _base_terms(offset=np.zeros(12), exposure=EXPOSURE_BASE)
    assert_allclose(res.predict(offset=0.), mu, rtol=RTOL, atol=ATOL)
    offset = -np.abs(OFFSET_BASE) - 0.5
    mu = np.exp(X_BASE @ PARAMS_BASE + offset + np.log(EXPOSURE_BASE))
    assert_allclose(res.predict(offset=offset), mu, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# Scalar and one dimensional inputs, positional use
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", ["glm_results", "poisson_results"])
def test_scalar_broadcast(request, fixture):
    res = request.getfixturevalue(fixture)
    c = 1.5
    eta = X_BASE @ PARAMS_BASE + np.log(c) + OFFSET_BASE
    mu = np.exp(eta)
    new_exog = X_BASE[:4]
    pred = res.get_prediction(exog=new_exog, exposure=c)
    mu_sub = np.exp(new_exog @ PARAMS_BASE + np.log(c))
    assert_allclose(pred.predicted, mu_sub, rtol=RTOL, atol=ATOL)

    # scalar exposure combined with fitted offset
    pred = res.get_prediction(exposure=c)
    assert_allclose(pred.predicted, mu, rtol=RTOL, atol=ATOL)


def test_array_used_positionally_not_by_index():
    # pandas indexes must not be used to align exposure/offset
    res = _fit_poisson()
    exog = pd.DataFrame(X_BASE[:4], index=[10, 11, 12, 13])
    exposure = pd.Series(EXPOSURE_BASE[:4], index=[3, 2, 1, 0])
    predicted = res.get_prediction(exog=exog, exposure=exposure,
                                   offset=pd.Series(OFFSET_BASE[:4],
                                                    index=[9, 8, 7, 6]))
    expected = np.exp(X_BASE[:4] @ PARAMS_BASE + np.log(EXPOSURE_BASE[:4]) +
                      OFFSET_BASE[:4])
    assert_allclose(predicted.predicted, expected, rtol=RTOL, atol=ATOL)


def test_scalar_direct_predict_rows():
    # a scalar exposure is broadcast over the prediction rows
    res = _fit_poisson()
    pred = res.predict(exog=X_BASE[:3], exposure=2.0)
    expected = np.exp(X_BASE[:3] @ PARAMS_BASE + np.log(2.0))
    assert_allclose(pred, expected, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# The same request must agree across direct prediction, attributes and the
# summary frame; the variance must correspond to the exposure used.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", ["glm_results", "poisson_results"])
@pytest.mark.parametrize("which", ["mean", "linear"])
def test_request_consistency(request, fixture, which):
    res = request.getfixturevalue(fixture)
    kwds = {"exposure": 1.7 * EXPOSURE_BASE, "offset": OFFSET_BASE + 0.1}
    if which == "mean":
        direct = res.predict(which="mean", **kwds)
    else:
        direct = res.predict(which="linear", **kwds)
    pred = res.get_prediction(which=which, **kwds)
    frame = pred.summary_frame()
    assert_allclose(pred.predicted, direct, rtol=RTOL, atol=ATOL)
    assert_allclose(frame["predicted"], direct, rtol=RTOL, atol=ATOL)
    assert_allclose(frame["se"], pred.se, rtol=1e-14)
    # for the mean the variance depends on the exposure through the link
    # derivative, so it must differ from a prediction without it.  For the
    # linear prediction the variance only depends on the design matrix.
    no_exposure = res.get_prediction(which=which, exog=X_BASE)
    same = np.allclose(pred.var_pred, no_exposure.var_pred)
    assert same == (which == "linear")


def test_glm_legacy_request_consistency(glm_results):
    res = glm_results
    kwds = {"exposure": 1.7 * EXPOSURE_BASE, "offset": OFFSET_BASE + 0.1}
    pred = res.get_prediction(**kwds)
    eta = X_BASE @ PARAMS_BASE + np.log(1.7 * EXPOSURE_BASE) + OFFSET_BASE + .1
    mu = np.exp(eta)
    assert_allclose(pred.predicted_mean, mu, rtol=RTOL, atol=ATOL)
    assert_allclose(pred.summary_frame()["mean"], mu, rtol=RTOL, atol=ATOL)
    # the linear prediction variance corresponds to the same design
    lin = pred.linpred
    assert_allclose(lin.predicted_mean, eta, rtol=RTOL, atol=ATOL)
    assert_allclose(pred.predicted_mean, res.predict(**kwds), rtol=RTOL,
                    atol=ATOL)


# ---------------------------------------------------------------------------
# Derivative bookkeeping: no observation covariance or (n, k, p) tensor
# ---------------------------------------------------------------------------


def test_prob_derivative_is_row_factored():
    from statsmodels.base._prediction_inference import RowFactoredJacobian

    n = 300
    rs = np.random.RandomState(0)
    X = np.column_stack([np.ones(n), rs.randn(n, 2)])
    exposure = rs.uniform(0.5, 2, n)
    offset = rs.normal(0, 0.1, n)
    y = rs.poisson(np.exp(X @ np.array([0.3, 0.2, -0.1]) + offset) * exposure)
    res = Poisson(y, X, exposure=exposure, offset=offset).fit(disp=0)

    k = 5
    pred = res.get_prediction(which="prob", y_values=np.arange(k))
    assert isinstance(pred.deriv, RowFactoredJacobian)
    assert pred.deriv.factor.shape == (n, k)
    assert pred.predicted.shape == (n, k)
    # materializing the full tensor would be the only place it exists, and
    # the result object does not keep it
    full = pred.deriv.asarray()
    assert full.shape == (n, k, 3)

    avg = res.get_prediction(which="prob", y_values=np.arange(k),
                             average=True)
    assert np.shape(avg.deriv) == (k, 3)


def test_aggregate_derivative_memory_shape():
    # for the aggregated prediction no (n, k, p) derivative is retained
    from statsmodels.base._prediction_inference import RowFactoredJacobian

    n = 400
    rs = np.random.RandomState(1)
    X = np.column_stack([np.ones(n), rs.randn(n, 4)])
    exposure = rs.uniform(0.5, 2, n)
    y = rs.poisson(np.exp(X @ np.array([0.2, 0.1, -0.1, 0.05, 0.0])) *
                   exposure)
    res = Poisson(y, X, exposure=exposure).fit(disp=0)
    k = 6
    avg = res.get_prediction(which="prob", y_values=np.arange(k),
                             average=True)
    assert np.shape(avg.predicted) == (k,)
    assert np.shape(avg.deriv) == (k, 5)

    # averaging an analytic row-factored derivative does not form the tensor
    row = res.get_prediction(which="prob", y_values=np.arange(k))
    assert isinstance(row.deriv, RowFactoredJacobian)
    agg = row.deriv.aggregate_derivative(np.ones(n))
    assert agg.shape == (k, 5)


# ---------------------------------------------------------------------------
# Formula models
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def formula_data():
    rs = np.random.RandomState(1)
    n = 30
    df = pd.DataFrame({
        "x": np.linspace(0, 2, n),
        "g": np.repeat(["a", "b", "c"], 10),
        "exposure": np.linspace(1, 2, n),
        "offset": np.linspace(-.2, .2, n),
    })
    df["y"] = rs.poisson(
        np.exp(0.5 + 0.3 * df["x"] + df["offset"]) * df["exposure"])
    return df


@pytest.mark.parametrize("formula", ["glm", "poisson"])
def test_formula_transform(formula_data, formula):
    df = formula_data
    if formula == "glm":
        mod = smf.glm("y ~ x + g", data=df, family=sm.families.Poisson(),
                      exposure=df.exposure.values, offset=df.offset.values)
    else:
        mod = smf.poisson("y ~ x + g", data=df,
                          exposure=df.exposure.values,
                          offset=df.offset.values)
    res = mod.fit(disp=0)
    new = df.iloc[:6][["x", "g"]].copy()
    pred_true = res.get_prediction(new, exposure=df.exposure.values[:6],
                                   offset=df.offset.values[:6])
    pred_false = res.get_prediction(res.model.exog[:6], transform=False,
                                    exposure=df.exposure.values[:6],
                                    offset=df.offset.values[:6])
    assert_allclose(pred_true.predicted, pred_false.predicted,
                    rtol=RTOL, atol=ATOL)
    assert_allclose(pred_true.se, pred_false.se, rtol=RTOL, atol=ATOL)

    # formula labels, model state and params are unchanged
    assert res.model.formula == "y ~ x + g"
    assert isinstance(res.model.exog, np.ndarray)
    params_before = np.array(res.params)
    res.get_prediction(new)
    assert_allclose(np.array(res.params), params_before, rtol=1e-14)


# ---------------------------------------------------------------------------
# GLM and discrete Poisson agree
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cov_type", ["nonrobust", "HC0"])
def test_glm_poisson_agree(cov_type):
    glm = _fit_glm(cov_type=cov_type)
    poisson = _fit_poisson(cov_type=cov_type)
    assert_allclose(glm.params, poisson.params, rtol=1e-8, atol=1e-10)
    assert_allclose(glm.cov_params(), poisson.cov_params(), rtol=1e-6,
                    atol=1e-9)

    for which in ["mean", "linear"]:
        pg = glm.get_prediction(which=which)
        pp = poisson.get_prediction(which=which)
        assert_allclose(pg.predicted, pp.predicted, rtol=RTOL, atol=ATOL)
        assert_allclose(pg.se, pp.se, rtol=1e-6, atol=1e-9)

    pg = glm.get_prediction(which="mean", average=True,
                            agg_weights=AGG_WEIGHTS)
    pp = poisson.get_prediction(which="mean", average=True,
                                agg_weights=AGG_WEIGHTS)
    assert_allclose(pg.predicted, pp.predicted, rtol=RTOL, atol=ATOL)
    assert_allclose(pg.se, pp.se, rtol=1e-6, atol=1e-9)

    # the prediction variance must be propagated with the covariance of the
    # results instance, including the non-default cov_type
    _, mu = _base_terms()
    covb = glm.cov_params()
    expected_se = mu * _linpred_se(X_BASE, covb)
    assert_allclose(glm.get_prediction(which="mean").se, expected_se,
                    rtol=RTOL, atol=ATOL)
