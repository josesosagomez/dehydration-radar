"""Exp A/C scoring and uncertainty — pure functions over already-computed predictions.

No fitting, no fold construction, no I/O. Three roles for this module:

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

  * Exp C's four ordinal pure functions (`class_unit_mae`, `adjacent_accuracy`,
    `quadratic_weighted_kappa`, `confusion_counts`) score the 5-class S0-S4 secondary task.
    They never raise: `quadratic_weighted_kappa`'s undefinedness trigger is decided by the
    actual zero-expected-disagreement denominator, not by a class-count pre-check (O-M9-8),
    so the frozen per-fold MAE fallback can consume it directly.
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


def _cluster_bootstrap_over_rows(
    subjects,
    theta_of_rows,
    *,
    b: int = 10000,
    level: float = 0.95,
    rng_seed: int,
    method: str = "bca",
    skip_threshold_pct: float = 5.0,
) -> BootstrapCI:
    """Shared subject-cluster bootstrap machinery, extracted from what was previously
    `subject_cluster_bootstrap_pooled`'s body: resample SUBJECTS with replacement (b draws
    of n subjects, all of a subject's rows travel together with multiplicity), evaluate
    `theta_of_rows(row_index_array)` per resample, jackknife leave-one-subject-out for BCa,
    percentile fallback, skip-and-count. Internal helper — not exported. RNG draw order
    (`rng.integers(0, n, size=n)` called once per replicate, `b` replicates in a loop, never
    vectorised) is preserved bit-for-bit from the pre-extraction implementation.
    """
    subjects = np.asarray(subjects)
    subj_ids = sorted(set(subjects.tolist()))
    rows_by_subject = {s: np.flatnonzero(subjects == s) for s in subj_ids}
    n = len(subj_ids)

    def metric_over(chosen_subjects) -> float:
        idx = np.concatenate([rows_by_subject[s] for s in chosen_subjects])
        return theta_of_rows(idx)

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
    seed-averaged metric on the original subjects (each once). A thin wrapper over
    `_cluster_bootstrap_over_rows` — the bootstrap loop itself lives there.
    """
    subjects = np.asarray(subjects)
    y_true = np.asarray(y_true, dtype=float)
    y_pred_by_seed = np.atleast_2d(np.asarray(y_pred_by_seed, dtype=float))
    n_seeds = y_pred_by_seed.shape[0]

    def theta_of_rows(idx) -> float:
        yt = y_true[idx]
        per_seed = [metric_fn(yt, y_pred_by_seed[k, idx]) for k in range(n_seeds)]
        per_seed = [v for v in per_seed if math.isfinite(v)]
        return float(np.mean(per_seed)) if per_seed else float("nan")

    return _cluster_bootstrap_over_rows(
        subjects, theta_of_rows, b=b, level=level, rng_seed=rng_seed, method=method,
        skip_threshold_pct=skip_threshold_pct,
    )


def session_weighted_bootstrap(
    subjects,
    session_idx,
    y_true,
    y_pred_by_seed,
    *,
    y_pred_reference=None,
    b: int = 10000,
    level: float = 0.95,
    rng_seed: int,
    method: str = "bca",
    skip_threshold_pct: float = 5.0,
) -> BootstrapCI:
    """Exp B's PRIMARY CI (A-M8-1): the subject-cluster bootstrap of the session-weighted,
    equal-weight-per-session aggregate residual MAE (`equal_session_residual_mae`).

    Within each subject-resample, recompute the per-session residual MAEs on the resampled
    rows and average with EQUAL WEIGHT per session, per seed, then average across seeds ->
    one scalar per resample (the POOLED/nonlinear seed-collapse rule, since this is an
    average of averages, not an average of per-subject values). `y_pred_reference` (the
    session-mean baseline; identically zero on the residual scale by construction), when
    given, makes the bootstrapped quantity `aggregate(radar) - aggregate(reference)`
    directly, evaluated on the SAME resampled rows for both, so it is a genuinely paired
    difference bootstrap, not two independent ones.

    A-M8-2: a resample whose rows do not cover every session present in the FULL input
    (checked once per resample, before any per-seed work, since session coverage does not
    depend on `y_pred_by_seed`) is undefined for the four-session aggregate — skipped and
    counted via the existing NaN/`math.isfinite` machinery in `_cluster_bootstrap_over_rows`,
    never silently averaged over the surviving sessions.
    """
    subjects = np.asarray(subjects)
    session_idx = np.asarray(session_idx)
    y_true = np.asarray(y_true, dtype=float)
    y_pred_by_seed = np.atleast_2d(np.asarray(y_pred_by_seed, dtype=float))
    n_seeds = y_pred_by_seed.shape[0]
    y_pred_reference = None if y_pred_reference is None else np.asarray(y_pred_reference, dtype=float)
    expected_sessions = set(session_idx.tolist())

    def theta_of_rows(idx) -> float:
        if set(session_idx[idx].tolist()) != expected_sessions:
            return float("nan")   # A-M8-2: empty-session replicate, skip-and-count
        sidx = session_idx[idx]
        yt = y_true[idx]
        per_seed = []
        for k in range(n_seeds):
            agg = equal_session_residual_mae(None, yt, y_pred_by_seed[k, idx], sidx)
            if y_pred_reference is not None:
                agg -= equal_session_residual_mae(None, yt, y_pred_reference[idx], sidx)
            per_seed.append(agg)
        per_seed = [v for v in per_seed if math.isfinite(v)]
        return float(np.mean(per_seed)) if per_seed else float("nan")

    return _cluster_bootstrap_over_rows(
        subjects, theta_of_rows, b=b, level=level, rng_seed=rng_seed, method=method,
        skip_threshold_pct=skip_threshold_pct,
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


# ---------------------------------------------------------------------- Exp B objective


def per_session_residual_mae(session_idx, y_true, y_pred) -> dict[int, float]:
    """{session index: residual MAE over that session's rows}.

    The shared building block of BOTH Exp B's inner-CV objective and its reported
    per-session breakdown -- one definition, so selection and reporting cannot drift. A
    session with no rows in this call is simply absent from the returned dict.
    """
    session_idx = np.asarray(session_idx)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out: dict[int, float] = {}
    for s in sorted(set(session_idx.tolist())):
        mask = session_idx == s
        out[int(s)] = float(np.abs(y_true[mask] - y_pred[mask]).mean())
    return out


def equal_session_residual_mae(subjects, y_true, y_pred, session_idx) -> float:
    """Exp B's inner-CV selection objective and per-fold aggregate building block.

    The mean of `per_session_residual_mae`'s values, EQUAL WEIGHT PER SESSION over
    sessions present in THIS call's rows -- deliberately NOT subject-weighted (that is the
    separate paired-test estimand, computed elsewhere). `subjects` is accepted and unused
    so the signature matches the harness `score_fn` hook uniformly:
    `score_fn(subjects, y_true, y_pred, session_idx) -> float`. NaN on empty input.

    RUN-LEVEL VIABILITY: this function silently averages over whatever sessions are
    present in its arguments -- correct for scoring one fold's rows, where per-fold
    session drops are expected and already logged. It must NOT be used, uncritically, to
    compute the run-level primary point estimate: if a session is absent from the GLOBAL
    out-of-fold matrix, a naive call over the whole matrix would silently report a
    three-session mean labelled as the four-session primary. `summarize_exp_b` is the sole
    caller responsible for checking global per-session N_eval > 0 for all of S1-S4 BEFORE
    treating its output as the primary aggregate.
    """
    per_session = per_session_residual_mae(session_idx, y_true, y_pred)
    if not per_session:
        return float("nan")
    return float(np.mean(list(per_session.values())))


def holm_adjusted(p_values, *, family_size: int | None = None) -> list[float]:
    """Holm-Bonferroni step-down adjustment, RETURNED IN INPUT ORDER.

    `family_size` defaults to `len(p_values)` but the caller pins it to
    `StatsConfig.holm_family_expb_per_session = 4` so a session missing from a given run
    (a shrunk `len(p_values)`) cannot weaken the pre-registered correction. NaN inputs pass
    through as NaN at their original position -- they occupy a family-size slot
    (conservative: the multiplier budget spent on them is not handed to the finite
    p-values) but are excluded from the step-down ranking of the finite ones.
    """
    p_values = [float(p) for p in p_values]
    m0 = len(p_values) if family_size is None else family_size
    order = sorted(
        (i for i in range(len(p_values)) if not math.isnan(p_values[i])),
        key=lambda i: p_values[i],
    )
    adjusted = [float("nan")] * len(p_values)
    running_max = 0.0
    for rank, i in enumerate(order):
        multiplier = m0 - rank
        val = min(1.0, multiplier * p_values[i])
        running_max = max(running_max, val)
        adjusted[i] = running_max
    return adjusted


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


# ------------------------------------------------------- Exp C ordinal metrics (Milestone 9)


def class_unit_mae(y_class_true, y_class_pred) -> float:
    """Pooled mean |predicted - true| in class units -- Exp C's frozen inner objective
    (`:766-769`). Deliberately POOLED, not subject-balanced (that is Exp A's convention):
    `:1199-1204` classifies class-unit MAE among the pooled/nonlinear metrics. NaN on empty."""
    yt = np.asarray(y_class_true, dtype=float)
    yp = np.asarray(y_class_pred, dtype=float)
    if yt.size == 0:
        return float("nan")
    return float(np.mean(np.abs(yt - yp)))


def adjacent_accuracy(y_class_true, y_class_pred) -> float:
    """Pooled fraction of predictions within one class of the truth. NaN on empty."""
    yt = np.asarray(y_class_true, dtype=float)
    yp = np.asarray(y_class_pred, dtype=float)
    if yt.size == 0:
        return float("nan")
    return float(np.mean(np.abs(yt - yp) <= 1))


def confusion_counts(y_class_true, y_class_pred, *, n_classes: int = 5) -> np.ndarray:
    """`n_classes` x `n_classes` integer counts; ROWS = true class, columns = predicted class."""
    yt = np.asarray(y_class_true, dtype=int)
    yp = np.asarray(y_class_pred, dtype=int)
    counts = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(counts, (yt, yp), 1)
    return counts


def quadratic_weighted_kappa(y_class_true, y_class_pred, *, n_classes: int = 5) -> float:
    """Cohen's kappa with quadratic weights over the fixed `n_classes` x `n_classes` grid
    (weights `(i-j)^2/(K-1)^2`, expected counts from the marginal outer product `r_i*c_j/n`).

    O-M9-8 (decision 8a, owner-approved 2026-07-30): undefinedness is decided by the actual
    denominator, NOT by a class-count pre-check. Returns NaN iff the input is empty or the
    expected disagreement `sum(w_ij * E_ij)` is exactly 0 -- which on this fixed grid happens
    only when both marginals concentrate on the SAME single class. A single-class truth (or
    predicted) side alone does not trigger NaN as long as the other side varies: e.g. true
    all-S0 against a varying predictor has nonzero expected disagreement and kappa=0 for an
    uninformative predictor. Matches `sklearn.metrics.cohen_kappa_score(...,
    weights="quadratic", labels=[0..n_classes-1])` exactly, including its (8b)-would-skip
    cases; never raises, so the frozen per-fold MAE fallback can consume it directly.
    """
    yt = np.asarray(y_class_true, dtype=int)
    yp = np.asarray(y_class_pred, dtype=int)
    if yt.size == 0:
        return float("nan")
    n = yt.size
    counts = confusion_counts(yt, yp, n_classes=n_classes)
    row_marginal = counts.sum(axis=1)
    col_marginal = counts.sum(axis=0)
    idx = np.arange(n_classes)
    weights = (idx[:, None] - idx[None, :]) ** 2 / float((n_classes - 1) ** 2)
    expected = np.outer(row_marginal, col_marginal) / n
    expected_disagreement = float(np.sum(weights * expected))
    if expected_disagreement == 0.0:
        return float("nan")
    observed_disagreement = float(np.sum(weights * counts))
    return 1.0 - observed_disagreement / expected_disagreement
