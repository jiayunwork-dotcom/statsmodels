# Cluster bootstrap experiments

Reproducible experiments for the pairs cluster bootstrap inference
implemented in `statsmodels/regression/_cluster_bootstrap.py`
(`RegressionResults.get_cluster_bootstrap`, `t_test(..., inference=...)`,
`get_prediction(..., inference=...)`).

All seeds are fixed in the scripts before any result is computed; data
seeds and calibration seeds come from two separate master seeds.

## How to run

From this directory, with the statsmodels source tree on `PYTHONPATH`:

```bash
export PYTHONPATH=/workspace
python run_coverage_experiment.py          # ~10 s, writes coverage_results.json
python run_diagnostics_experiment.py       # ~20 s, writes diagnostics_results.json
python run_resource_experiment.py          # ~15 s, writes resource_results.json
```

Each script accepts `--n-datasets` (coverage/diagnostics) and `--out`.

## 1. Coverage experiment (`run_coverage_experiment.py`)

Design (pre-registered): 400 independent datasets; each with 40 independent
clusters of 8 rows; `y = 1 + 0.5*x + u_g + e` with `x ~ N(0,1)`,
`u_g ~ N(0,1)`, `e ~ N(0,0.25)`; 999 valid resamples per dataset
(`min_valid_resamples = 999`); 90% simultaneous confidence band for the
conditional mean at `x in {-2,-1,0,1,2}`; success = all five true means
inside the joint band.  Master seeds: data `20261117`, calibration
`81520261`.

Raw result (`coverage_results.json`, run on this tree):

| quantity | value |
|---|---|
| joint 90% band coverage (400 datasets) | **0.8575** |
| Wilson 95% interval | [0.820, 0.888] |
| mean pointwise 90% interval coverage | 0.8845 |
| acceptance interval | [0.84, 0.96] -> **PASS** |

The joint band is slightly conservative-low against the 0.90 nominal level,
as expected for a cluster bootstrap at G = 40; it lies inside the
pre-registered acceptance interval.

## 2. Diagnostics (`run_diagnostics_experiment.py`)

Same estimand (90% joint mean band at five points), 200 datasets per
scenario, seeds: data `20261118`, calibration `81520262`.

| scenario | G | coverage |
|---|---|---|
| D1 baseline (equal size 8, homoskedastic) | 40 | 0.865 |
| D2 unequal sizes (2..44 ramp) | 40 | 0.845 |
| D3 within-cluster heteroskedasticity (sd 0.5..4) | 40 | 0.880 |
| D4 few clusters (equal size 8) | 6 | 0.725 |
| D5 few clusters + unequal sizes | 6 | 0.725 |

Interpretation (documents the method's limits; not an acceptance test):

* At G = 40 the joint band is mildly anti-conservative (0.845-0.880 vs
  0.90 nominal); unequal cluster sizes and cluster-level heteroskedasticity
  move the coverage by a few points in either direction.
* At G = 6 (the minimum supported) coverage drops to about 0.73: the
  empirical cluster distribution is a poor stand-in for the cluster
  sampling distribution.  The method is asymptotic in G; six-cluster
  designs are not expected to attain nominal coverage.

## 3. Resource experiment (`run_resource_experiment.py`)

n = 20000, p = 6, G = 500 (40 rows/cluster), B = 999 requested resamples;
prediction grids m = 2000 and m = 20000; batch_size 8 and 32.  OLS refits
are counted by instrumenting the module's resample-fit function; peak RSS
is `ru_maxrss`.

Raw result (`resource_results.json`):

| phase | batch | m | time | OLS refits | peak RSS before->after |
|---|---|---|---|---|---|
| calibration | 8 | - | 0.28 s | 999 (= n_valid) | 164 -> 168 MB |
| calibration | 32 | - | 0.26 s | 999 | 172 -> 173 MB |
| prediction query | 8 | 2000 | 0.07 s | 0 | 168 -> 168 MB |
| prediction query | 8 | 20000 | 0.73 s | 0 | 169 -> 172 MB |
| prediction query | 32 | 2000 | 0.07 s | 0 | 173 -> 173 MB |
| prediction query | 32 | 20000 | 0.68 s | 0 | 173 -> 173 MB |
| contrast query | 8/32 | 50 | <0.01 s | 0 | - |

Held by the calibration: `boot_params` (999, 6), `params` (6,),
`cluster_sizes` (500,) - i.e. O(B*p + p + G); no B copies of y/X, no B
stored models.

Findings:

* Exactly `n_valid` refits at calibration; **zero** refits during any
  query (t_test, conf_int, summary_frame, repeated queries).
* Peak RSS is flat between m = 2000 and m = 20000: no B x m prediction
  matrix is materialized (a full B x m matrix at m = 20000 would cost
  ~160 MB; none appears).
* `batch_size` 8 vs 32: `boot_params` and joint confidence bands are
  bitwise identical; only workspace chunking changes.
