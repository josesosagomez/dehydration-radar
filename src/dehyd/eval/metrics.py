"""Exp A scoring and uncertainty — pure functions over already-computed predictions.

No fitting, no fold construction, no I/O. Two roles for this module:

  * `subject_balanced_mae` is BOTH the inner-CV selection metric and the per-subject
    headline (subject-balanced, never pooled, so a subject with more eligible sessions
    cannot dominate). The frozen leakage suite re-exports it from here via
    `reference_procedure.py`, and pins it to 5.5 on a deliberately unequal fixture — so
    its behaviour must stay byte-identical to the milestone-1 definition.

  * The subject-cluster bootstrap is a self-contained **BCa** implementation with a
    percentile fallback, matching `StatsConfig` (B=10000, subject resample unit, BCa →
    percentile when unstable, undefined-metric skip-and-count with a >5% unreliable
    flag). We do NOT use `scipy.stats.bootstrap`: scipy is pinned <1.17 and we need the
    fallback/skip bookkeeping under our own control. `scipy.stats` is used only for the
    normal quantile/CDF (BCa) and the Wilcoxon signed-rank test.

The seed-collapse rules (StatsConfig) are metric-type-aware and live at the call sites:
  * additive per-subject metrics (per-subject MAE): average each subject's per-seed
    values into ONE value per subject, then `subject_cluster_bootstrap` over those;
  * pooled/nonlinear metrics (RMSE, pooled r): within each resample recompute the metric
    per seed on the resampled data, then average across seeds — `subject_cluster_bootstrap_pooled`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats as _sps

# ------------------------------------------------------------------- point metrics


def subject_balanced_mae(subjects, y_true, y_pred) -> float:
    """Mean over subjects of each subject's mean |error| across its sessions.

    Subject-balanced, NOT pooled. Byte-compatible with the milestone-1 definition the
    frozen leakage suite pins to 5.5 (fixture subjects=[1,1,1,1,1,2,2], y_pred all-ones
    then all-tens, y_true=0 → mean(1, 10) = 5.5, not the pooled 25/7).
    """
    subjects = np.asarray(subjects)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    per_subject = [
        np.abs(y_true[subjects == s] - y_pred[subjects == s]).mean()
        for s in sorted(set(subjects.tolist()))
    ]
    return float(np.mean(per_subject))


def session_rmse(y_true, y_pred) -> float:
    """Pooled session-level RMSE. Always defined for non-empty input."""
    d = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(d**2)))


def pooled_pearson_r(y_true, y_pred) -> float:
    """Pooled predicted-vs-actual Pearson r. NaN on zero-variance input (the caller
    skips-and-counts such resamples), so it never raises inside the bootstrap loop."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.size < 2 or yt.std() == 0.0 or yp.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(yt, yp)[0, 1])


def per_subject_pearson_r(subjects, y_true, y_pred, *, min_sessions: int = 3) -> dict[int, float]:
    """Per-subject predicted-vs-actual r for subjects with >= min_sessions sessions.

    Descriptive only (no CI, no headline). A subject with fewer than min_sessions
    sessions (StatsConfig.per_subject_pearson_r_min_sessions = 3) is omitted; a subject
    with zero variance gets NaN (reported, not dropped silently).
    """
    subjects = np.asarray(subjects)
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    out: dict[int, float] = {}
    for s in sorted(set(subjects.tolist())):
        mask = subjects == s
        if int(mask.sum()) >= min_sessions:
            out[int(s)] = pooled_pearson_r(yt[mask], yp[mask])
    return out


# --------------------------------------------------------------- cluster bootstrap


@dataclass(frozen=True)
class BootstrapCI:
    """A subject-cluster bootstrap confidence interval and its provenance.

    `method` is the interval actually used ("bca" or "percentile" — the fallback fires
    when BCa is unstable and is RECORDED, per StatsConfig.ci_fallback). `n_skipped`
    counts resamples whose metric was undefined (pooled r on zero-variance draws);
    `unreliable` is True when that exceeds StatsConfig.undefined_metric_skip_threshold_pct.
    """

    point: float
    low: float
    high: float
    method: str
    n_eval: int
    n_skipped: int
    unreliable: bool


def _percentile_interval(boot: np.ndarray, level: float) -> tuple[float, float]:
    finite = np.asarray(boot, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:  # every resample was skipped (e.g. pooled r on constant truth)
        return float("nan"), float("nan")
    alpha = (1.0 - level) / 2.0
    lo = float(np.percentile(finite, 100.0 * alpha))
    hi = float(np.percentile(finite, 100.0 * (1.0 - alpha)))
    return lo, hi


def _bca_interval(theta_hat, boot, jack, level) -> tuple[float, float] | None:
    """BCa endpoints, or None if the bias/acceleration terms are degenerate.

    Degeneracy (all bootstrap replicates on one side of theta_hat, or a zero jackknife
    spread — both common at small N_eval) makes z0 or a undefined; returning None signals
    the caller to fall back to the percentile interval and record that.
    """
    boot = np.asarray(boot, dtype=float)
    boot = boot[np.isfinite(boot)]
    if boot.size == 0:
        return None
    prop = float(np.mean(boot < theta_hat))
    if prop <= 0.0 or prop >= 1.0:
        return None
    z0 = float(_sps.norm.ppf(prop))

    jack = np.asarray(jack, dtype=float)
    jack = jack[np.isfinite(jack)]
    if jack.size < 2:
        return None
    jbar = jack.mean()
    diff = jbar - jack
    denom = 6.0 * (np.sum(diff**2) ** 1.5)
    if denom == 0.0:
        return None
    a = float(np.sum(diff**3) / denom)

    alpha = (1.0 - level) / 2.0
    z_lo = float(_sps.norm.ppf(alpha))
    z_hi = float(_sps.norm.ppf(1.0 - alpha))

    def adjust(z):
        return float(_sps.norm.cdf(z0 + (z0 + z) / (1.0 - a * (z0 + z))))

    p_lo = 100.0 * adjust(z_lo)
    p_hi = 100.0 * adjust(z_hi)
    if not (math.isfinite(p_lo) and math.isfinite(p_hi)):
        return None
    lo = float(np.nanpercentile(boot, p_lo))
    hi = float(np.nanpercentile(boot, p_hi))
    return lo, hi


def _finalize(theta_hat, boot, jack, *, level, method, n_eval, n_skipped, skip_threshold_pct):
    used = method
    interval = None
    if method == "bca":
        interval = _bca_interval(theta_hat, boot, jack, level)
        if interval is None:
            used = "percentile"
    if interval is None:
        interval = _percentile_interval(boot, level)
    unreliable = (n_skipped / max(len(boot), 1)) * 100.0 > skip_threshold_pct
    return BootstrapCI(
        point=float(theta_hat),
        low=interval[0],
        high=interval[1],
        method=used,
        n_eval=n_eval,
        n_skipped=n_skipped,
        unreliable=unreliable,
    )


def subject_cluster_bootstrap(
    per_subject_values,
    *,
    b: int = 10000,
    level: float = 0.95,
    rng_seed: int,
    method: str = "bca",
    skip_threshold_pct: float = 5.0,
) -> BootstrapCI:
    """Cluster bootstrap of the MEAN of one already-collapsed value per subject.

    For additive per-subject metrics (per-subject MAE, or per-subject radar-minus-baseline
    differences): resample the N subjects' scalars with replacement B times, take the mean
    each time, and form a BCa (fallback percentile) interval around the observed mean.
    """
    vals = np.asarray(per_subject_values, dtype=float)
    n = vals.size
    if n == 0:
        raise ValueError("subject_cluster_bootstrap got no subject values")
    theta_hat = float(np.mean(vals))
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, n, size=(b, n))
    boot = vals[idx].mean(axis=1)
    jack = np.array([np.delete(vals, i).mean() for i in range(n)]) if n >= 2 else np.array([theta_hat])
    return _finalize(
        theta_hat, boot, jack, level=level, method=method,
        n_eval=n, n_skipped=0, skip_threshold_pct=skip_threshold_pct,
    )


def subject_cluster_bootstrap_pooled(
    subjects,
    y_true,
    y_pred_by_seed,
    metric_fn,
    *,
    b: int = 10000,
    level: float = 0.95,
    rng_seed: int,
    method: str = "bca",
    skip_threshold_pct: float = 5.0,
) -> BootstrapCI:
    """Cluster bootstrap of a POOLED metric (RMSE, pooled r) with the pooled seed-collapse.

    `y_pred_by_seed` is (n_seeds, n_sessions); `y_true` is (n_sessions,). Within each
    resample of SUBJECTS (carrying all of a subject's sessions, with multiplicity) the
    metric is recomputed **per seed** and averaged across the finite seeds — one value per
    resample. A resample with no finite seed value is skipped and counted; if that exceeds
    `skip_threshold_pct` the CI is flagged unreliable. The point estimate is the same
    seed-averaged metric on the original subjects (each once).
    """
    subjects = np.asarray(subjects)
    y_true = np.asarray(y_true, dtype=float)
    y_pred_by_seed = np.atleast_2d(np.asarray(y_pred_by_seed, dtype=float))
    n_seeds = y_pred_by_seed.shape[0]

    subj_ids = sorted(set(subjects.tolist()))
    rows_by_subject = {s: np.flatnonzero(subjects == s) for s in subj_ids}
    n = len(subj_ids)

    def metric_over(chosen_subjects) -> float:
        idx = np.concatenate([rows_by_subject[s] for s in chosen_subjects])
        yt = y_true[idx]
        per_seed = [metric_fn(yt, y_pred_by_seed[k, idx]) for k in range(n_seeds)]
        per_seed = [v for v in per_seed if math.isfinite(v)]
        return float(np.mean(per_seed)) if per_seed else float("nan")

    theta_hat = metric_over(subj_ids)
    rng = np.random.default_rng(rng_seed)
    boot = np.empty(b, dtype=float)
    n_skipped = 0
    for i in range(b):
        chosen = [subj_ids[j] for j in rng.integers(0, n, size=n)]
        v = metric_over(chosen)
        if not math.isfinite(v):
            n_skipped += 1
        boot[i] = v
    jack = (
        np.array([metric_over([s for s in subj_ids if s != s0]) for s0 in subj_ids])
        if n >= 2
        else np.array([theta_hat])
    )
    return _finalize(
        theta_hat, boot, jack, level=level, method=method,
        n_eval=n, n_skipped=n_skipped, skip_threshold_pct=skip_threshold_pct,
    )


def mean_difference_ci(
    per_subject_differences, *, b: int = 10000, level: float = 0.95, rng_seed: int, method: str = "bca"
) -> BootstrapCI:
    """Cluster-bootstrap CI on the mean per-subject (radar − baseline) difference.

    Just `subject_cluster_bootstrap` over the per-subject difference values — the
    baseline comparison's interval companion to the Wilcoxon test below.
    """
    return subject_cluster_bootstrap(
        per_subject_differences, b=b, level=level, rng_seed=rng_seed, method=method
    )


def wilcoxon_signed_rank(differences) -> tuple[float, float]:
    """Wilcoxon signed-rank (statistic, p) over the per-subject differences.

    The paired radar-vs-baseline test (StatsConfig.paired_test). All-zero differences
    make the test undefined; we return (nan, nan) rather than raise so the caller records
    it. scipy.stats.wilcoxon is stable within the <1.17 pin.
    """
    diffs = np.asarray(differences, dtype=float)
    if diffs.size == 0 or np.allclose(diffs, 0.0):
        return float("nan"), float("nan")
    res = _sps.wilcoxon(diffs)
    return float(res.statistic), float(res.pvalue)
