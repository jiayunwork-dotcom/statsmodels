"""
Coverage experiment for the pairs cluster bootstrap joint mean band.

Design (fixed in advance, see the master seeds below):
  * 400 independent synthetic datasets
  * each dataset: 40 independent clusters of 8 rows (n = 320)
  * y = 1 + 0.5 * x + u_g + e, x iid N(0, 1), u_g iid N(0, 1),
    e iid N(0, 0.25)
  * per dataset: one pairs cluster bootstrap calibration with
    n_resamples = 999 (all valid in this design)
  * estimand: 90% simultaneous (family-wise) confidence band for the
    conditional mean at the five design points x in {-2, -1, 0, 1, 2}
    (rows [1, x]); success = all five true means 1 + 0.5*x are inside
    the joint band
  * acceptance: family coverage in [0.84, 0.96]

Data seeds and calibration seeds are drawn from two separate, fixed
master seeds below; they do not depend on any result.

Usage:
    python run_coverage_experiment.py [--n-datasets 400] [--out coverage_results.json]
"""

import argparse
import json
import time

import numpy as np

import statsmodels.api as sm

# pre-registered master seeds (data and calibration streams are separate)
DATA_MASTER_SEED = 20261117
CALIBRATION_MASTER_SEED = 81520261
N_DATASETS = 400
N_CLUSTERS = 40
CLUSTER_SIZE = 8
BETA = np.array([1.0, 0.5])
ERROR_SD = 0.5  # N(0, 0.25) errors
N_RESAMPLES = 999
ALPHA = 0.10  # 90% joint band
X_GRID = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
ACCEPT = (0.84, 0.96)


def make_dataset(gen):
    x = gen.standard_normal(N_CLUSTERS * CLUSTER_SIZE)
    u = gen.standard_normal(N_CLUSTERS)
    e = gen.standard_normal(N_CLUSTERS * CLUSTER_SIZE) * ERROR_SD
    groups = np.repeat(np.arange(N_CLUSTERS), CLUSTER_SIZE)
    y = BETA[0] + BETA[1] * x + u[groups] + e
    exog = np.column_stack([np.ones_like(x), x])
    return y, exog, groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-datasets", type=int, default=N_DATASETS)
    parser.add_argument("--out", default="coverage_results.json")
    args = parser.parse_args()

    n_datasets = args.n_datasets
    data_children = np.random.SeedSequence(DATA_MASTER_SEED).spawn(n_datasets)
    calib_seeds = np.random.default_rng(CALIBRATION_MASTER_SEED).integers(
        0, 2 ** 32 - 1, size=n_datasets)

    design = np.column_stack([np.ones_like(X_GRID), X_GRID])
    true_mean = design @ BETA

    covered_joint = np.zeros(n_datasets, dtype=bool)
    covered_pointwise = np.zeros((n_datasets, len(X_GRID)), dtype=bool)
    n_valid_min = N_RESAMPLES
    t_start = time.perf_counter()
    for i in range(n_datasets):
        gen = np.random.default_rng(data_children[i])
        y, exog, groups = make_dataset(gen)
        res = sm.OLS(y, exog).fit()
        cal = res.get_cluster_bootstrap(
            groups, n_resamples=N_RESAMPLES,
            random_state=int(calib_seeds[i]))
        if cal.n_valid < N_RESAMPLES:
            raise RuntimeError(
                f"dataset {i}: only {cal.n_valid} valid resamples")
        n_valid_min = min(n_valid_min, cal.n_valid)
        # joint 90% band for the five conditional means
        tt = res.t_test(design, inference=cal, simultaneous=True)
        ci_joint = tt.conf_int(ALPHA)
        covered_joint[i] = np.all(
            (ci_joint[:, 0] <= true_mean) & (true_mean <= ci_joint[:, 1]))
        # pointwise 90% intervals (secondary statistic)
        tp = res.t_test(design, inference=cal, simultaneous=False)
        ci_pw = tp.conf_int(ALPHA)
        covered_pointwise[i] = (
            (ci_pw[:, 0] <= true_mean) & (true_mean <= ci_pw[:, 1]))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n_datasets} datasets done, "
                  f"running joint coverage = {covered_joint[:i + 1].mean():.3f}",
                  flush=True)
    elapsed = time.perf_counter() - t_start

    cov_joint = float(covered_joint.mean())
    cov_pw = float(covered_pointwise.mean())
    # Wilson score interval for the joint coverage
    z = 1.959963984540054
    n = n_datasets
    denom = 1 + z * z / n
    center = (cov_joint + z * z / (2 * n)) / denom
    half = z * np.sqrt(cov_joint * (1 - cov_joint) / n
                       + z * z / (4 * n * n)) / denom
    result = {
        "experiment": "joint_mean_band_coverage",
        "n_datasets": n_datasets,
        "n_clusters": N_CLUSTERS,
        "cluster_size": CLUSTER_SIZE,
        "n_resamples": N_RESAMPLES,
        "min_valid_resamples": n_valid_min,
        "alpha": ALPHA,
        "x_grid": X_GRID.tolist(),
        "data_master_seed": DATA_MASTER_SEED,
        "calibration_master_seed": CALIBRATION_MASTER_SEED,
        "coverage_joint": cov_joint,
        "coverage_joint_wilson95": [center - half, center + half],
        "coverage_pointwise_mean": cov_pw,
        "acceptance_interval": list(ACCEPT),
        "accepted": bool(ACCEPT[0] <= cov_joint <= ACCEPT[1]),
        "elapsed_seconds": elapsed,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"joint 90% band coverage = {cov_joint:.3f} "
          f"(acceptance [{ACCEPT[0]}, {ACCEPT[1]}]) -> "
          f"{'PASS' if result['accepted'] else 'FAIL'}")


if __name__ == "__main__":
    main()
