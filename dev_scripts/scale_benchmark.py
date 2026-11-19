"""
Memory/time scaling evidence for CumIncidenceRight fitted queries.

The implementation never builds an R x M x K array (prediction rows x
merged event times x causes).  Per query row it only allocates arrays
of size M or K x M on the shared stratum grid; the only R-scaled
allocation is the final result table of R x len(times) rows.

This script reports, for increasing sample size with sparse staggered
event dates:
  * M  - merged event grid size (distinct event times in the stratum)
  * the size an R*M*K float64 array *would* have, for contrast
  * measured peak heap growth during the query
  * wall time
"""
import time
import tracemalloc
import numpy as np
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import CumIncidenceRight


def build(rng, n, n_causes):
    x = rng.normal(size=(n, 2))
    tt = rng.uniform(0.1, 100.0, n)        # continuous, sparse dates
    cause = rng.integers(1, n_causes + 1, n)
    censor = rng.uniform(size=n) < 0.3
    raw_status = np.where(censor, 0, cause)
    fits = {k: PHReg(tt, x, (raw_status == k).astype(float)).fit()
            for k in range(1, n_causes + 1)}
    return CumIncidenceRight.from_phreg(fits), x


def main():
    rng = np.random.default_rng(0)
    K = 4
    header = (f"{'n':>7} {'R':>6} {'M':>6} {'R*M*K floats':>14} "
              f"{'R*M*K MB':>9} {'peak MB':>9} {'out MB':>8} "
              f"{'sec':>7}")
    print(header)
    rows = []
    for n in (500, 2000, 8000):
        ci, x = build(rng, n, K)
        R = n
        M = len(ci.times)
        qtimes = np.array([10., 20., 40., 40., 70., 100.])  # 6 incl dup
        st = np.zeros(R, dtype=int)
        # Warm up lazy caches; measure only the query call below.
        ci.predict(5.0, qtimes[:1], exog=x[:2], strata=st[:2])
        tracemalloc.start()
        t0 = time.perf_counter()
        q = ci.predict(5.0, qtimes, exog=x, strata=st)
        dt = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        forbidden_floats = R * M * K
        forbidden_bytes = forbidden_floats * 8
        forbidden_mb = forbidden_bytes / 1e6
        # Output table size (the only allocation that scales with R).
        out_bytes = R * len(qtimes) * (K + 1) * 8
        out_mb = out_bytes / 1e6
        print(f"{n:>7} {R:>6} {M:>6} {forbidden_floats:>14,d} "
              f"{forbidden_mb:>9.2f} {peak/1e6:>9.2f} {out_mb:>8.2f} "
              f"{dt:>7.3f}")
        per_row_grid_bytes = M * K * 8
        peak_per_row = peak / R
        # A handful of M*K temporaries per row plus the output row;
        # allow a generous constant factor, but it must not scale with
        # R as an R*M*K array would.
        assert peak_per_row < 60 * per_row_grid_bytes, (
            "per-row memory grows with R; an R*M*K array may be held")
        assert q.shape == (R * len(qtimes), K + 1)
        rows.append((R, M, dt, peak))
    # Per-row peak should be roughly constant as R grows.
    per_row = [pk / r for (r, _, _, pk) in rows]
    print("\npeak bytes per query row:", np.round(per_row, 1))
    assert max(per_row) / min(per_row) < 2.0, (
        "per-row peak memory is not stable as R grows")
    # Wall time should scale roughly linearly in R for fixed M/K.
    (r0, _, t0s, _), (r1, _, t1s, _), (r2, _, t2s, _) = rows
    print("\nsec per 1000 query rows:",
          np.round([t0s / r0 * 1000, t1s / r1 * 1000,
                    t2s / r2 * 1000], 4))


if __name__ == "__main__":
    main()
