"""
Diagnostic experiments: where does the pairs cluster bootstrap degrade?

The pairs cluster bootstrap is asymptotic in the number of clusters G and
treats the observed clusters as draws from a common cluster population.
These scenarios probe the limits stated in the documentation:

  D1  baseline:            G = 40, equal size 8, homoskedastic errors
  D2  unequal sizes:       G = 40, sizes ramping from 2 to 44
  D3  heteroskedastic clusters: G = 40, size 8, within-cluster error sd
                           rising from 0.5 to 4.0 across clusters
  D4  few clusters:        G = 6, equal size 8
  D5  few + unequal:       G = 6, sizes [4, 8, 12, 16, 20, 24]

For each scenario, 200 datasets are generated with the same model as the
main coverage experiment (y = 1 + 0.5*x + u_g + e) and the family coverage
of the 90% joint mean band at x in {-2, -1, 0, 1, 2} is recorded.  These
are diagnostics, not acceptance tests: deviations document the method's
limits; no claim is made that nominal coverage holds for arbitrary
six-cluster or strongly heterogeneous designs.

Usage:
    python run_diagnostics_experiment.py [--n-datasets 200]

"""

import argparse
import json
import time

import numpy as np

import statsmodels.api as sm

DATA_MASTER_SEED = 20261118
CALIBRATION_MASTER_SEED = 81520262
N_DATASETS = 200
BETA = np.array([1.0, 0.5])
N_RESAMPLES = 999
ALPHA = 0.10
X_GRID = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

SCENARIOS = {
    "D1_baseline": {"sizes": [8] * 40, "sd_scale": None},
    "D2_unequal_sizes": {"sizes": [2 + (44 - 2) * g // 39 for g in range(40)],
                         "sd_scale": None},
    "D3_heteroskedastic": {"sizes": [8] * 40,
                           "sd_scale": "ramp"},
    "D4_few_clusters": {"sizes": [8] * 6, "sd_scale": None},
    "D5_few_unequal": {"sizes": [4, 8, 12, 16, 20, 24], "sd_scale": None},
}


def make_dataset(gen, sizes, sd_scale):
    n_clusters = len(sizes)
    groups = np.repeat(np.arange(n_clusters), sizes)
    n = len(groups)
    x = gen.standard_normal(n)
    u = gen.standard_normal(n_clusters)
    if sd_scale == "ramp":
        sd_g = 0.5 * 8.0 ** (np.arange(n_clusters) / (n_clusters - 1.0))
    else:
        sd_g = np.full(n_clusters, 0.5)
    e = gen.standard_normal(n) * sd_g[groups]
    y = BETA[0] + BETA[1] * x + u[groups] + e
    exog = np.column_stack([np.ones_like(x), x])
    return y, exog, groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-datasets", type=int, default=N_DATASETS)
    parser.add_argument("--out", default="diagnostics_results.json")
    args = parser.parse_args()
    n_datasets = args.n_datasets

    design = np.column_stack([np.ones_like(X_GRID), X_GRID])
    true_mean = design @ BETA

    data_children = np.random.SeedSequence(DATA_MASTER_SEED).spawn(
        len(SCENARIOS) * n_datasets)
    calib_seeds = np.random.default_rng(CALIBRATION_MASTER_SEED).integers(
        0, 2 ** 32 - 1, size=len(SCENARIOS) * n_datasets)

    results = {}
    t_start = time.perf_counter()
    for s_idx, (name, spec) in enumerate(SCENARIOS.items()):
        covered = np.zeros(n_datasets, dtype=bool)
        min_valid = N_RESAMPLES
        for i in range(n_datasets):
            k = s_idx * n_datasets + i
            gen = np.random.default_rng(data_children[k])
            y, exog, groups = make_dataset(gen, spec["sizes"],
                                           spec["sd_scale"])
            res = sm.OLS(y, exog).fit()
            cal = res.get_cluster_bootstrap(
                groups, n_resamples=N_RESAMPLES,
                random_state=int(calib_seeds[k]))
            min_valid = min(min_valid, cal.n_valid)
            ci = res.t_test(design, inference=cal,
                            simultaneous=True).conf_int(ALPHA)
            covered[i] = np.all(
                (ci[:, 0] <= true_mean) & (true_mean <= ci[:, 1]))
        results[name] = {
            "coverage_joint": float(covered.mean()),
            "n_datasets": n_datasets,
            "n_clusters": len(spec["sizes"]),
            "sizes": spec["sizes"] if len(spec["sizes"]) <= 8 else "ramp",
            "sd_scale": spec["sd_scale"],
            "min_valid_resamples": min_valid,
        }
        print(f"{name}: joint 90% coverage = {covered.mean():.3f}",
              flush=True)
    elapsed = time.perf_counter() - t_start
    out = {
        "experiment": "diagnostics",
        "alpha": ALPHA,
        "n_resamples": N_RESAMPLES,
        "data_master_seed": DATA_MASTER_SEED,
        "calibration_master_seed": CALIBRATION_MASTER_SEED,
        "scenarios": results,
        "elapsed_seconds": elapsed,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
