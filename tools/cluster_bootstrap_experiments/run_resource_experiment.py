"""
Resource experiment: fit counts, memory, and runtime of cluster bootstrap
calibration and queries.

Design (fixed):
  * n = 20000 rows, p = 6 parameters (intercept + 5 regressors),
    G = 500 clusters of 40 rows, B = 999 requested resamples
  * prediction grids with m = 2000 and m = 20000 rows
  * batch_size in {8, 32}

Recorded per phase:
  * number of OLS refits (instrumented via a counter on the module's
    resample-fit function): calibration must perform exactly n_valid
    refits and queries must perform zero
  * peak RSS (ru_maxrss) before/after each phase
  * wall time
  * shapes of the objects and arrays held by the calibration

The calibration stores O(n*p + B*p + G) and queries must not materialize
the B x m prediction matrix: peak memory during prediction queries must
stay roughly flat between m = 2000 and m = 20000 (a B*m matrix would
cost ~160 MB at m = 20000 and ~1.6 GB at B*m for m = 200000).

Usage:
    python run_resource_experiment.py [--out resource_results.json]
"""

import argparse
import json
import resource
import time

import numpy as np

import statsmodels.api as sm
import statsmodels.regression._cluster_bootstrap as cb_module

N_OBS = 20000
N_PARAMS = 6
N_CLUSTERS = 500
CLUSTER_SIZE = N_OBS // N_CLUSTERS
N_RESAMPLES = 999
GRIDS = (2000, 20000)
BATCH_SIZES = (8, 32)
DATA_SEED = 20261119
CALIBRATION_SEED = 81520263


def peak_rss_mb():
    # ru_maxrss is a high-water mark in KiB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="resource_results.json")
    args = parser.parse_args()

    report = {
        "n_obs": N_OBS, "n_params": N_PARAMS, "n_clusters": N_CLUSTERS,
        "n_resamples": N_RESAMPLES, "data_seed": DATA_SEED,
        "calibration_seed": CALIBRATION_SEED, "phases": [],
    }

    gen = np.random.default_rng(DATA_SEED)
    x = gen.standard_normal((N_OBS, N_PARAMS - 1))
    exog = np.column_stack([np.ones(N_OBS), x])
    beta = np.arange(1.0, N_PARAMS + 1.0) / 2.0
    groups = np.repeat(np.arange(N_CLUSTERS), CLUSTER_SIZE)
    u = gen.standard_normal(N_CLUSTERS)
    e = gen.standard_normal(N_OBS) * 0.5
    y = exog @ beta + u[groups] + e

    res = sm.OLS(y, exog).fit()

    # instrument the resample fit to count OLS refits
    fit_counter = {"n": 0}
    original_fit = cb_module._fit_resampled

    def counting_fit(*a, **k):
        fit_counter["n"] += 1
        return original_fit(*a, **k)

    cb_module._fit_resampled = counting_fit
    try:
        for batch_size in BATCH_SIZES:
            fit_counter["n"] = 0
            rss0 = peak_rss_mb()
            t0 = time.perf_counter()
            cal = res.get_cluster_bootstrap(
                groups, n_resamples=N_RESAMPLES,
                random_state=CALIBRATION_SEED, batch_size=batch_size)
            t1 = time.perf_counter()
            rss1 = peak_rss_mb()
            fits_after_cal = fit_counter["n"]
            phase = {
                "phase": "calibration", "batch_size": batch_size,
                "seconds": t1 - t0, "rss_before_mb": rss0,
                "rss_after_mb": rss1, "ols_refits": fits_after_cal,
                "n_valid": cal.n_valid, "n_failed": cal.n_failed,
                "boot_params_shape": list(cal.boot_params.shape),
                "params_shape": list(cal.params.shape),
                "cluster_sizes_shape": list(cal.cluster_sizes.shape),
            }
            report["phases"].append(phase)
            print(f"calibration batch_size={batch_size}: "
                  f"{phase['seconds']:.2f}s, refits={fits_after_cal}, "
                  f"rss {rss0:.0f}->{rss1:.0f} MB", flush=True)

            for m in GRIDS:
                grid_gen = np.random.default_rng(DATA_SEED + m)
                a_new = np.column_stack(
                    [np.ones(m)]
                    + [grid_gen.standard_normal(m)
                       for _ in range(N_PARAMS - 1)])
                for simultaneous in (False, True):
                    fit_counter["n"] = 0
                    rss_q0 = peak_rss_mb()
                    tq0 = time.perf_counter()
                    pred = res.get_prediction(a_new, inference=cal)
                    ci = pred.conf_int(alpha=0.05,
                                       simultaneous=simultaneous)
                    frame = pred.summary_frame(
                        alpha=0.05, simultaneous=simultaneous)
                    tq1 = time.perf_counter()
                    rss_q1 = peak_rss_mb()
                    phase = {
                        "phase": "prediction_query",
                        "batch_size": batch_size, "m": m,
                        "simultaneous": simultaneous,
                        "seconds": tq1 - tq0,
                        "rss_before_mb": rss_q0, "rss_after_mb": rss_q1,
                        "ols_refits": fit_counter["n"],
                        "ci_shape": list(ci.shape),
                        "frame_shape": list(frame.shape),
                    }
                    report["phases"].append(phase)
                    print(f"  query m={m} simul={simultaneous} "
                          f"bs={batch_size}: {phase['seconds']:.2f}s, "
                          f"refits={fit_counter['n']}, "
                          f"rss {rss_q0:.0f}->{rss_q1:.0f} MB", flush=True)
                # contrast query on the same design for reference
                fit_counter["n"] = 0
                tq0 = time.perf_counter()
                tt = res.t_test(a_new[:50], inference=cal,
                                simultaneous=True)
                _ = tt.conf_int(0.05)
                tq1 = time.perf_counter()
                report["phases"].append({
                    "phase": "contrast_query", "batch_size": batch_size,
                    "m": 50, "seconds": tq1 - tq0,
                    "ols_refits": fit_counter["n"]})
                print(f"  contrast m=50 bs={batch_size}: "
                      f"{tq1 - tq0:.2f}s, refits={fit_counter['n']}",
                      flush=True)

        # batch_size must not change any output
        cal8 = res.get_cluster_bootstrap(groups, n_resamples=N_RESAMPLES,
                                         random_state=CALIBRATION_SEED,
                                         batch_size=8)
        cal32 = res.get_cluster_bootstrap(groups, n_resamples=N_RESAMPLES,
                                          random_state=CALIBRATION_SEED,
                                          batch_size=32)
        same_boot = bool(np.array_equal(cal8.boot_params,
                                        cal32.boot_params))
        m = GRIDS[-1]
        grid_gen = np.random.default_rng(DATA_SEED + m)
        a_new = np.column_stack(
            [np.ones(m)]
            + [grid_gen.standard_normal(m) for _ in range(N_PARAMS - 1)])
        ci8 = res.get_prediction(a_new, inference=cal8).conf_int(
            alpha=0.05, simultaneous=True)
        ci32 = res.get_prediction(a_new, inference=cal32).conf_int(
            alpha=0.05, simultaneous=True)
        same_ci = bool(np.array_equal(ci8, ci32))
        report["batch_size_invariance"] = {
            "boot_params_bitwise_equal": same_boot,
            "joint_ci_bitwise_equal": same_ci,
        }
        print(f"batch_size invariance: boot={same_boot}, joint_ci={same_ci}")
    finally:
        cb_module._fit_resampled = original_fit

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["batch_size_invariance"], indent=2))


if __name__ == "__main__":
    main()
