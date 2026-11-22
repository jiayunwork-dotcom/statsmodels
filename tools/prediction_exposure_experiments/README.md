# Exposure/offset prediction resource experiment

This directory contains the resource experiment referenced by the
exposure/offset prediction fix.

Run from the repository root (or with `PYTHONPATH` pointing at it):

```
python tools/prediction_exposure_experiments/run_resource_experiment.py \
    --out resource_results.json
```

The script fixes `p = 6` parameters and `k = 8` requested counts, and runs
`n = 2000` and `n = 20000`.  For the per-row and the aggregated probability
prediction it records the number of calls into the shared prediction layer,
the number of derivative helper invocations, the peak traced memory and the
shapes of the arrays retained by the result object.

Recorded results (Python 3.11, NumPy 2.2.6, SciPy 1.14.1, pandas 2.2.3):

| n     | average | predict calls | derivative calls | peak MB | predicted shape  | retained derivative        |
|-------|---------|---------------|------------------|---------|------------------|----------------------------|
| 2000  | False   | 2             | 1                | 0.61    | (2000, 8)        | RowFactoredJacobian (2000, 8) factor |
| 2000  | True    | 2             | 1                | 0.55    | (8,)             | ndarray (8, 6)             |
| 20000 | False   | 2             | 1                | 5.51    | (20000, 8)       | RowFactoredJacobian (20000, 8) factor |
| 20000 | True    | 2             | 1                | 4.29    | (8,)             | ndarray (8, 6)             |

Per-row probabilities: two `model.predict` calls (one for the statistic, one
without an extra derivative evaluation needed because the analytic derivative
is used) and one derivative call.  Extra memory is dominated by the
`(n, k)` predicted array and the `(n, k)` row factor, i.e. `O(n*k)`.  The full
`(n, k, p)` derivative tensor is never materialized and no `n x n`
observation covariance is formed.

Aggregated probabilities: the weighted average is differentiated directly /
the row-factored derivative is contracted with the design, so only a `(k, p)`
derivative is retained.  Peak memory does not grow with `n` through the
derivative path.
