"""Resource experiment for Poisson prediction with exposure and offset.

Design (fixed): p = 6 parameters, k = 8 requested counts (probabilities),
and n in {2000, 20000}.  Exposure and offset are both present and vary
across rows.

For each n the script records, for the per-row probability prediction and
for the aggregated (average=True) probability prediction:

* the number of calls to the prediction function (model.predict) that come
  from the shared inference layer, i.e. the derivative work
* the number of derivative helper invocations
* the peak traced memory during the call
* the shapes of the arrays retained by the returned results object

The growth source is reported explicitly.  The expected behaviour is

* per-row probabilities keep a row-factored Jacobian of shape (n, k) and
  a predicted array of shape (n, k), so extra memory is O(n*k) and not
  O(n*k*p) and no n x n observation covariance is ever formed;
* the aggregated probabilities differentiate the weighted average directly,
  so the retained derivative is (k, p) and no (n, k, p) tensor is kept.

Usage:
    python run_resource_experiment.py [--out resource_results.json]
"""

import argparse
import json
import resource
import tracemalloc

import numpy as np

import statsmodels.api as sm
import statsmodels.base._prediction_inference as pinference


N_PARAMS = 6
N_COUNTS = 8
NS = (2000, 20000)
DATA_SEED = 20261122


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def make_data(n, seed=DATA_SEED):
    gen = np.random.default_rng(seed)
    x = gen.standard_normal((n, N_PARAMS - 1))
    exog = np.column_stack([np.ones(n), x])
    beta = np.arange(1.0, N_PARAMS + 1.0) / 10.0
    exposure = gen.uniform(0.5, 2.0, n)
    offset = gen.normal(0.0, 0.1, n)
    mu = np.exp(exog @ beta + offset) * exposure
    y = gen.poisson(mu)
    return exog, y, exposure, offset


class Counters:
    def __init__(self):
        self.reset()

    def reset(self):
        self.predict_calls = 0
        self.deriv_helper_calls = 0
        self.numeric_calls = 0
        self.row_factored_calls = 0


def _deriv_shapes(deriv):
    if isinstance(deriv, pinference.RowFactoredJacobian):
        return {"type": "RowFactoredJacobian",
                "factor": list(deriv.factor.shape),
                "exog": list(deriv.exog.shape)}
    return {"type": type(deriv).__name__,
            "shape": list(np.shape(deriv))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="resource_results.json")
    args = parser.parse_args()

    counters = Counters()
    original_predict_deriv = pinference._predict_derivatives
    original_numeric = pinference._numeric_derivatives
    original_numeric_avg = pinference._numeric_derivatives_average

    def counting_predict_deriv(*a, **k):
        counters.deriv_helper_calls += 1
        out = original_predict_deriv(*a, **k)
        if isinstance(out, pinference.RowFactoredJacobian):
            counters.row_factored_calls += 1
        return out

    def counting_numeric(*a, **k):
        counters.numeric_calls += 1
        return original_numeric(*a, **k)

    def counting_numeric_avg(*a, **k):
        counters.numeric_calls += 1
        return original_numeric_avg(*a, **k)

    pinference._predict_derivatives = counting_predict_deriv
    pinference._numeric_derivatives = counting_numeric
    pinference._numeric_derivatives_average = counting_numeric_avg

    report = {"n_params": N_PARAMS, "n_counts": N_COUNTS,
              "n_values": list(NS), "seed": DATA_SEED, "cases": []}

    try:
        for n in NS:
            exog, y, exposure, offset = make_data(n)
            res = sm.Poisson(y, exog, exposure=exposure,
                             offset=offset).fit(disp=0)
            model = res.model
            y_values = np.arange(N_COUNTS)

            original_model_predict = model.predict

            def counting_model_predict(*a, **k):
                counters.predict_calls += 1
                return original_model_predict(*a, **k)

            model.predict = counting_model_predict
            try:
                for average in (False, True):
                    counters.reset()
                    rss0 = peak_rss_mb()
                    tracemalloc.start()
                    pred = res.get_prediction(which="prob",
                                              y_values=y_values,
                                              average=average)
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    rss1 = peak_rss_mb()
                    case = {
                        "n_obs": n,
                        "average": average,
                        "predict_calls": counters.predict_calls,
                        "predict_deriv_calls": counters.deriv_helper_calls,
                        "row_factored_calls": counters.row_factored_calls,
                        "numeric_deriv_calls": counters.numeric_calls,
                        "peak_traced_mb": peak / 1e6,
                        "rss_before_mb": rss0,
                        "rss_after_mb": rss1,
                        "predicted_shape": list(np.shape(pred.predicted)),
                        "se_shape": list(np.shape(pred.se)),
                        "deriv": _deriv_shapes(pred.deriv),
                    }
                    report["cases"].append(case)
                    print(json.dumps(case), flush=True)
            finally:
                model.predict = original_model_predict
    finally:
        pinference._predict_derivatives = original_predict_deriv
        pinference._numeric_derivatives = original_numeric
        pinference._numeric_derivatives_average = original_numeric_avg

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
