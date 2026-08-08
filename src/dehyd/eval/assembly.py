"""Milestone-10 assembly — one explicit run map, one adapter per experiment schema.

Two rules shape this whole module, and both exist because the alternative is a quiet wrong
answer rather than a crash.

**Run directories are supplied explicitly and never discovered.** There is no glob, no "latest"
directory, no fallback search. A milestone's final tables are lineage-bearing: a headline number
that came from whichever directory happened to sort last is not traceable, and the failure is
invisible in the output. `build_run_manifest` takes the caller's explicit mapping, records the
run path, the required relative artifacts, their SHA-256, the source commit and the resolved
config hash, and fails closed on anything missing.

**There is no uniform experiment schema, and this module never pretends there is.** The five
experiments genuinely answer different questions and their artifacts say so:

    A   `metrics_exp_a_{band}.json`  subject_balanced_mae / session_rmse / pooled_pearson_r,
                                     each a CI dict, plus a session-index baseline comparison
    B   `metrics_exp_b_{band}.json`  a session-weighted PRIMARY aggregate that may be
                                     unavailable, a subject-weighted complete-case companion,
                                     and a per-session Holm-4 breakdown
    C   `metrics_exp_c_{band}.json`  two ARMS, each with ordinal metrics (class-unit MAE,
                                     adjacent accuracy, QWK) and its own selection frequency
    D   `metrics_exp_d_{band}.json`  per-FAMILY metrics, a composite comparison and a
                                     session-index comparison — no generic prediction table
    E   `metrics_exp_e_{band}.json`  descriptive attribution only; no estimate at all
    F   `metrics_exp_f_{band}.json`  contrast summaries with an explicit multiplicity family
    G   `metrics_exp_g.json`         cross-band, so no band suffix and no per-band row

Writing one "generic reader" over those would mean inventing a common shape that none of them
has. Each adapter below reads exactly the keys its own experiment writes, and an experiment
whose artifact does not state an estimate contributes no headline row rather than a fabricated
one — which is why Exp E appears in the manifest and the exclusions but not in
`headline_metrics.csv`.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class AssemblyError(ValueError):
    """A missing/mismatched source artifact, an unknown experiment, or a run directory that
    does not contain what its experiment is required to produce."""


# ------------------------------------------------------------------- the explicit run map


EXPERIMENTS = ("a", "b", "c", "d", "e", "f", "g", "robustness")
# G is cross-band by construction (it fuses the two), so it is keyed with band=None. Every
# other experiment is per-band.
CROSS_BAND = ("g",)


def required_artifacts(experiment: str, band: str | None) -> tuple[str, ...]:
    """The relative artifact names one experiment/band MUST have produced.

    Deliberately per-experiment literals rather than a pattern: the names are not derivable
    from each other (Exp B prefixes its prediction table `predictions_b_`, Exp D has no
    prediction table at all, Exp G has no band suffix), and a pattern that happened to work
    for four of them would silently accept a wrong file for the fifth.
    """
    if experiment == "a":
        return (f"metrics_exp_a_{band}.json", f"predictions_{band}.csv",
                f"selection_table_{band}.csv")
    if experiment == "b":
        return (f"metrics_exp_b_{band}.json", f"predictions_b_{band}.csv",
                f"selection_table_b_{band}.csv", f"dropped_sessions_{band}.csv")
    if experiment == "c":
        return (f"metrics_exp_c_{band}.json", f"predictions_{band}.csv",
                f"selection_table_{band}.csv", f"confusion_{band}.csv")
    if experiment == "d":
        return (f"metrics_exp_d_{band}.json", f"composite_{band}.csv")
    if experiment == "e":
        return (f"metrics_exp_e_{band}.json", f"importance_folds_{band}.csv",
                f"path_metadata_{band}.csv", f"importance_summary_{band}.csv",
                f"ridge_coefficients_{band}.csv", f"exclusions_e_{band}.csv")
    if experiment == "f":
        return (f"metrics_exp_f_{band}.json", "confound_availability.csv",
                f"predictions_f_{band}.csv", f"selection_f_{band}.csv",
                f"contrasts_f_{band}.csv", f"exclusions_f_{band}.csv")
    if experiment == "g":
        return ("metrics_exp_g.json", "per_subject_g.csv", "predictions_g.csv",
                "matched_population.csv", "exclusions_g.csv")
    if experiment == "robustness":
        return ("metrics_robustness.json", "robustness_summary.csv",
                "robustness_replicates.csv")
    raise AssemblyError(f"unknown experiment {experiment!r} (expected one of {EXPERIMENTS})")


@dataclass(frozen=True)
class SourceRun:
    """One explicitly supplied authoritative run directory."""

    experiment: str
    band: str | None
    run_dir: Path

    def key(self) -> str:
        return self.experiment if self.band is None else f"{self.experiment}_{self.band}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _config_sha256(provenance: dict) -> str:
    """Hash the resolved config exactly as `provenance.json` recorded it — the same
    sorted-keys JSON dump `exp_b.config_fingerprint` uses, so the two agree by construction."""
    return hashlib.sha256(
        json.dumps(provenance.get("config", {}), sort_keys=True).encode()
    ).hexdigest()


def build_run_manifest(sources) -> dict:
    """The explicit experiment/band -> run directory map, with lineage for every artifact.

    Fails closed, and names what is missing: a run directory that does not exist, an artifact
    the experiment is required to have produced, or a `provenance.json` without a commit. None
    of those is recoverable by looking somewhere else, which is the whole point.
    """
    sources = list(sources)
    if not sources:
        raise AssemblyError("build_run_manifest got no sources — assembly never discovers runs")
    seen = set()
    entries = {}
    for source in sources:
        if source.experiment not in EXPERIMENTS:
            raise AssemblyError(
                f"unknown experiment {source.experiment!r} (expected one of {EXPERIMENTS})")
        if (source.experiment in CROSS_BAND) != (source.band is None):
            raise AssemblyError(
                f"{source.experiment!r} is "
                f"{'cross-band and takes band=None' if source.experiment in CROSS_BAND else 'per-band and requires a band'}"
                f", got band={source.band!r}"
            )
        if source.key() in seen:
            raise AssemblyError(
                f"{source.key()} was supplied twice — one experiment/band maps to exactly ONE "
                "authoritative run"
            )
        seen.add(source.key())

        run_dir = Path(source.run_dir)
        if not run_dir.is_dir():
            raise AssemblyError(f"{source.key()}: run directory does not exist: {run_dir}")
        provenance_path = run_dir / "provenance.json"
        if not provenance_path.is_file():
            raise AssemblyError(f"{source.key()}: no provenance.json in {run_dir}")
        provenance = _read_json(provenance_path)
        commit = (provenance.get("git") or {}).get("commit")
        if not commit:
            raise AssemblyError(f"{source.key()}: {provenance_path} records no git commit")

        artifacts = {}
        for name in required_artifacts(source.experiment, source.band):
            path = run_dir / name
            if not path.is_file():
                raise AssemblyError(
                    f"{source.key()}: required artifact {name!r} is missing from {run_dir}")
            artifacts[name] = {"path": str(path), "sha256": _sha256(path)}

        entries[source.key()] = {
            "experiment": source.experiment,
            "band": source.band,
            "run_dir": str(run_dir),
            "source_commit": commit,
            "resolved_config_sha256": _config_sha256(provenance),
            "artifacts": artifacts,
        }

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "discovery": "explicit_only_no_glob",
        "runs": entries,
    }


MANIFEST_SCHEMA_VERSION = "milestone10_run_manifest_v1"


def validate_manifest(manifest: dict) -> None:
    """Re-hash every registered artifact and refuse any that moved since the manifest was
    written. Assembly is the last step before the chapter; a source that changed underneath it
    must stop the milestone rather than quietly produce different final tables."""
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise AssemblyError(
            f"run manifest schema is {manifest.get('schema_version')!r}, "
            f"expected {MANIFEST_SCHEMA_VERSION!r}")
    for key, entry in sorted(manifest["runs"].items()):
        for name, record in sorted(entry["artifacts"].items()):
            path = Path(record["path"])
            if not path.is_file():
                raise AssemblyError(f"{key}: registered artifact vanished: {path}")
            digest = _sha256(path)
            if digest != record["sha256"]:
                raise AssemblyError(
                    f"{key}: {name} hashes to {digest} but the manifest registered "
                    f"{record['sha256']} — the source changed after registration")


def _metrics_of(entry: dict):
    """The one metrics JSON of a registered run, read through the manifest's own record so a
    reader can never open a file the manifest did not hash."""
    name = next(n for n in entry["artifacts"] if n.startswith("metrics_") and n.endswith(".json"))
    return _read_json(Path(entry["artifacts"][name]["path"])), name


def _require(metrics: dict, key: str, entry: dict, artifact: str):
    """Read a key an experiment's metrics JSON is required to carry, or fail closed naming it.

    A bare `KeyError` deep inside an adapter tells a reader nothing about which source was
    malformed; assembly is the last step before publication, so a truncated or half-written
    metrics file has to be diagnosable from the message alone.
    """
    if key not in metrics:
        raise AssemblyError(
            f"{entry['experiment']}"
            f"{'/' + entry['band'] if entry['band'] else ''}: {artifact} has no {key!r} — "
            f"it is not a complete {entry['experiment']} metrics file "
            f"(source: {entry['run_dir']})"
        )
    return metrics[key]


def _artifact(entry: dict, name: str) -> Path:
    if name not in entry["artifacts"]:
        raise AssemblyError(f"{entry['experiment']}: {name!r} is not a registered artifact")
    return Path(entry["artifacts"][name]["path"])


def _read_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ------------------------------------------------------------------------- shared shapes


HEADLINE_COLUMNS = (
    "experiment", "band", "model_or_contrast", "metric", "estimate", "ci_low", "ci_high",
    "ci_method", "n_subjects", "n_sessions", "status", "primary_or_secondary", "source_run",
    "source_artifact",
)
PER_SUBJECT_COLUMNS = (
    "experiment", "band", "subject", "model_or_contrast", "metric", "value", "n_sessions",
    "source_run",
)
PAIRED_COLUMNS = (
    "experiment", "band", "comparison", "direction", "estimate", "ci_low", "ci_high",
    "ci_method", "p_value", "p_value_adjusted", "multiplicity_family", "n_pairs",
    "n_nonzero_pairs", "n_ties", "test", "primary_or_secondary", "source_run", "source_artifact",
)
EXCLUSIONS_COLUMNS = ("experiment", "band", "unit", "identifier", "reason", "source_run")

_BLANK = ""


def _ci(record) -> dict:
    """A CI dict as A/B/C/D all write it -> the four headline columns. Missing pieces stay
    BLANK rather than becoming zeros; a 0.0 CI bound would read as a measurement."""
    if not isinstance(record, dict):
        return {"estimate": record, "ci_low": _BLANK, "ci_high": _BLANK, "ci_method": _BLANK}
    return {
        "estimate": record.get("point", _BLANK),
        "ci_low": record.get("low", _BLANK),
        "ci_high": record.get("high", _BLANK),
        "ci_method": record.get("method", _BLANK),
    }


def _headline(entry, artifact, model_or_contrast, metric, ci_record, *, n_subjects, n_sessions,
              status, tier) -> dict:
    return {
        "experiment": entry["experiment"], "band": entry["band"] or _BLANK,
        "model_or_contrast": model_or_contrast, "metric": metric,
        **_ci(ci_record),
        "n_subjects": n_subjects, "n_sessions": n_sessions,
        "status": status, "primary_or_secondary": tier,
        "source_run": entry["run_dir"], "source_artifact": artifact,
    }


# ------------------------------------------------------------------ per-experiment adapters


def adapt_a(entry) -> dict:
    """Exp A: subject-balanced MAE is primary; RMSE and pooled r are secondary. The
    session-index baseline comparison is a paired row, not a headline row."""
    metrics, artifact = _metrics_of(entry)
    n_subjects = _require(metrics, "n_eval_subjects", entry, artifact)
    n_sessions = _require(metrics, "n_sessions", entry, artifact)
    status = "conditional_exploratory" if metrics.get("conditional_exploratory") else "reported"
    headline = [
        _headline(entry, artifact, "selected_radar", "subject_balanced_mae",
                  _require(metrics, "subject_balanced_mae", entry, artifact), n_subjects=n_subjects, n_sessions=n_sessions,
                  status=status, tier="primary"),
        _headline(entry, artifact, "selected_radar", "session_rmse", _require(metrics, "session_rmse", entry, artifact),
                  n_subjects=n_subjects, n_sessions=n_sessions, status=status, tier="secondary"),
        _headline(entry, artifact, "selected_radar", "pooled_pearson_r",
                  _require(metrics, "pooled_pearson_r", entry, artifact), n_subjects=n_subjects, n_sessions=n_sessions,
                  status=status, tier="secondary"),
    ]
    baseline = _require(metrics, "baseline_session_index_only", entry, artifact)
    per_subject = [{
        "experiment": "a", "band": entry["band"], "subject": int(subject),
        "model_or_contrast": "selected_radar", "metric": "subject_mae", "value": float(value),
        "n_sessions": _BLANK, "source_run": entry["run_dir"],
    } for subject, value in sorted(metrics["per_subject_mae"].items(), key=lambda kv: int(kv[0]))]
    per_subject += [{
        "experiment": "a", "band": entry["band"], "subject": int(subject),
        "model_or_contrast": "session_index_baseline", "metric": "subject_mae",
        "value": float(value), "n_sessions": _BLANK, "source_run": entry["run_dir"],
    } for subject, value in sorted(baseline["per_subject_mae"].items(), key=lambda kv: int(kv[0]))]
    paired = [_paired(
        entry, artifact, "radar_minus_session_index_baseline",
        baseline["mean_difference_radar_minus_baseline"],
        p_value=baseline["wilcoxon_p"], n_pairs=n_subjects, tier="primary",
        family="single_uncorrected",
    )]
    return {"headline": headline, "per_subject": per_subject, "paired": paired, "exclusions": []}


def _paired(entry, artifact, comparison, ci_record, *, p_value, n_pairs, tier, family,
            adjusted=_BLANK, n_nonzero=_BLANK, n_ties=_BLANK,
            test="wilcoxon_signed_rank") -> dict:
    ci = _ci(ci_record)
    estimate = ci["estimate"]
    direction = _BLANK
    if isinstance(estimate, (int, float)) and not (isinstance(estimate, float) and np.isnan(estimate)):
        direction = "negative_favours_first_term" if estimate < 0 else "positive_favours_second_term"
    return {
        "experiment": entry["experiment"], "band": entry["band"] or _BLANK,
        "comparison": comparison, "direction": direction, **ci,
        "p_value": p_value, "p_value_adjusted": adjusted, "multiplicity_family": family,
        "n_pairs": n_pairs, "n_nonzero_pairs": n_nonzero, "n_ties": n_ties, "test": test,
        "primary_or_secondary": tier, "source_run": entry["run_dir"], "source_artifact": artifact,
    }


def adapt_b(entry) -> dict:
    """Exp B: the session-weighted primary aggregate — which may legitimately be UNAVAILABLE,
    and is then reported as a status rather than omitted — plus the subject-weighted
    complete-case companion and the per-session Holm-4 exploratory breakdown."""
    metrics, artifact = _metrics_of(entry)
    n_subjects = _require(metrics, "n_eval_subjects_aggregate", entry, artifact)
    n_sessions = _require(metrics, "n_rows", entry, artifact)
    headline, paired, exclusions = [], [], []

    if metrics.get("primary_viable") and metrics.get("primary_aggregate"):
        aggregate = metrics["primary_aggregate"]
        for name, tier in (("radar", "primary"), ("baseline", "secondary")):
            headline.append(_headline(
                entry, artifact, name, "equal_session_residual_mae", aggregate[name],
                n_subjects=n_subjects, n_sessions=n_sessions,
                status="conditional_exploratory", tier=tier))
        paired.append(_paired(
            entry, artifact, "radar_minus_baseline_equal_session_aggregate",
            aggregate["difference_radar_minus_baseline"], p_value=_BLANK, n_pairs=n_subjects,
            tier="primary", family="single_uncorrected", test="subject_cluster_bootstrap_ci"))
    else:
        headline.append(_headline(
            entry, artifact, "radar", "equal_session_residual_mae", {},
            n_subjects=n_subjects, n_sessions=n_sessions,
            status="primary_unavailable", tier="primary"))
        exclusions.append({
            "experiment": "b", "band": entry["band"], "unit": "run", "identifier": "primary_aggregate",
            "reason": metrics.get("primary_unavailable_reason") or "primary aggregate not viable",
            "source_run": entry["run_dir"],
        })

    paired_record = metrics.get("paired_subject_weighted_complete_case") or {}
    if paired_record:
        paired.append(_paired(
            entry, artifact, "radar_minus_baseline_subject_weighted_complete_case",
            paired_record.get("mean_difference_radar_minus_baseline", {}),
            p_value=paired_record.get("wilcoxon_p", _BLANK),
            n_pairs=paired_record.get("n_complete_case", _BLANK),
            tier="secondary", family="single_uncorrected"))

    for session, record in sorted((metrics.get("per_session_exploratory") or {}).items()):
        if not isinstance(record, dict) or "mean_difference" not in record:
            continue
        paired.append(_paired(
            entry, artifact, f"radar_minus_baseline_session_{session}", record["mean_difference"],
            p_value=record.get("wilcoxon_p", _BLANK), adjusted=record.get("holm_p", _BLANK),
            n_pairs=record.get("n_eval", _BLANK), tier="secondary",
            family="holm_4_expb_per_session"))

    for subject, dropped in sorted((metrics.get("dropped_sessions") or {}).get("outer_by_fold", {}).items()):
        if dropped:
            exclusions.append({
                "experiment": "b", "band": entry["band"], "unit": "session",
                "identifier": f"fold_{subject}", "reason": f"degenerate_session_means_dropped={dropped}",
                "source_run": entry["run_dir"],
            })
    return {"headline": headline, "per_subject": [], "paired": paired, "exclusions": exclusions}


def adapt_c(entry) -> dict:
    """Exp C: two ARMS. Class-unit MAE is the primary ordinal metric per arm; adjacent accuracy
    and QWK are secondary and — per §2.4 — carry no interval, so their CI columns stay blank."""
    metrics, artifact = _metrics_of(entry)
    n_subjects = _require(metrics, "n_eval_subjects", entry, artifact)
    n_sessions = _require(metrics, "n_rows", entry, artifact)
    headline, per_subject = [], []
    for arm, record in sorted((metrics.get("arms") or {}).items()):
        headline.append(_headline(
            entry, artifact, f"arm_{arm}", "class_unit_mae", _require(record, "class_unit_mae", entry, artifact),
            n_subjects=n_subjects, n_sessions=n_sessions, status="conditional_exploratory",
            tier="primary"))
        for metric in ("adjacent_accuracy", "quadratic_weighted_kappa"):
            value = record.get(metric)
            headline.append(_headline(
                entry, artifact, f"arm_{arm}", metric,
                {"point": (value or {}).get("point", _BLANK)} if isinstance(value, dict) else value,
                n_subjects=n_subjects, n_sessions=n_sessions,
                status="conditional_exploratory_no_interval", tier="secondary"))
        for subject, value in sorted((record.get("per_subject_class_mae") or {}).items(),
                                     key=lambda kv: int(kv[0])):
            per_subject.append({
                "experiment": "c", "band": entry["band"], "subject": int(subject),
                "model_or_contrast": f"arm_{arm}", "metric": "class_unit_mae",
                "value": float(value), "n_sessions": _BLANK, "source_run": entry["run_dir"],
            })
    return {"headline": headline, "per_subject": per_subject, "paired": [], "exclusions": []}


def adapt_d(entry) -> dict:
    """Exp D: per-FAMILY baselines plus the two frozen comparisons. There is no generic
    prediction table here — the per-subject values come from the radar block."""
    metrics, artifact = _metrics_of(entry)
    n_subjects = _require(metrics, "n_eval", entry, artifact)
    headline, per_subject, paired = [], [], []

    radar = metrics.get("radar") or {}
    if radar:
        headline.append(_headline(
            entry, artifact, "selected_radar", "subject_balanced_mae",
            radar.get("subject_balanced_mae", {}), n_subjects=n_subjects, n_sessions=_BLANK,
            status="conditional_exploratory", tier="primary"))
        for subject, value in sorted((radar.get("per_subject_mae") or {}).items(),
                                     key=lambda kv: int(kv[0])):
            per_subject.append({
                "experiment": "d", "band": entry["band"], "subject": int(subject),
                "model_or_contrast": "selected_radar", "metric": "subject_mae",
                "value": float(value), "n_sessions": _BLANK, "source_run": entry["run_dir"],
            })

    for family, record in sorted((metrics.get("per_family_metrics") or {}).items()):
        headline.append(_headline(
            entry, artifact, family, "subject_balanced_mae",
            record.get("subject_balanced_mae", record), n_subjects=n_subjects,
            n_sessions=_BLANK, status="conditional_exploratory", tier="secondary"))

    for key, comparison, family, tier in (
        ("primary_vs_session_index", "radar_minus_session_index", "single_uncorrected", "primary"),
        ("composite", "radar_minus_composite", "single_uncorrected", "secondary"),
    ):
        record = metrics.get(key) or {}
        if not record:
            continue
        difference = (record.get("mean_difference_radar_minus_baseline")
                      or record.get("mean_difference_radar_minus_composite") or {})
        paired.append(_paired(
            entry, artifact, comparison, difference,
            p_value=record.get("wilcoxon_p", _BLANK), n_pairs=record.get("n_eval", n_subjects),
            tier=tier, family=record.get("correction", family)))
    return {"headline": headline, "per_subject": per_subject, "paired": paired, "exclusions": []}


def adapt_e(entry) -> dict:
    """Exp E: DESCRIPTIVE ONLY, so it contributes no headline row at all.

    This is deliberate and is the module's clearest example of not inventing a shape. E's
    artifact states no estimate: its content is a per-path importance distribution, which lives
    in `importance_summary_{band}.csv` and belongs in the chapter's attribution section, not in
    a table of headline estimates with CIs. Assembly registers, validates and hashes E's
    artifacts, records its exclusions, and points at its own tables.
    """
    metrics, _ = _metrics_of(entry)
    exclusions = [{
        "experiment": "e", "band": entry["band"], "unit": "outer_fold",
        "identifier": str(row.get("test_subject", "")), "reason": row.get("reason", ""),
        "source_run": entry["run_dir"],
    } for row in _read_csv(_artifact(entry, f"exclusions_e_{entry['band']}.csv"))]
    return {"headline": [], "per_subject": [], "paired": [], "exclusions": exclusions,
            "descriptive": {
                "n_paths": metrics.get("n_paths"),
                "n_evaluable_outer_folds": metrics.get("n_evaluable_outer_folds"),
                "status": metrics.get("status"),
                "tables": [f"importance_summary_{entry['band']}.csv",
                           f"path_metadata_{entry['band']}.csv"],
            }}


def adapt_f(entry) -> dict:
    """Exp F: the not-estimable HR record travels as a status, and every contrast becomes a
    paired row carrying the multiplicity family its own summary assigned — the Holm-2 primary
    pair, the individually-reported exploratory pair, and the sensitivity variants."""
    metrics, artifact = _metrics_of(entry)
    n_subjects = metrics.get("n_subjects_f", _BLANK)
    paired = []
    for record in metrics.get("contrasts", []):
        variant = record["analysis_variant"]
        tier = ("primary" if record["multiplicity_family"].startswith("holm_")
                else "secondary")
        paired.append(_paired(
            entry, artifact, f"{record['contrast_id']}::{variant}", record,
            p_value=record.get("p_value_unadjusted", _BLANK),
            adjusted=record.get("p_value_holm", _BLANK),
            n_pairs=record.get("n_paired_subjects", _BLANK),
            n_nonzero=record.get("n_nonzero_pairs", _BLANK),
            n_ties=record.get("n_ties", _BLANK),
            tier=tier, family=record["multiplicity_family"]))
        paired[-1]["estimate"] = record.get("mean_difference", _BLANK)
        paired[-1]["ci_low"] = record.get("ci_low", _BLANK)
        paired[-1]["ci_high"] = record.get("ci_high", _BLANK)
        paired[-1]["ci_method"] = record.get("ci_method", _BLANK)

    per_subject = [{
        "experiment": "f", "band": entry["band"], "subject": int(row["subject"]),
        "model_or_contrast": f"{row['contrast_id']}::{row['analysis_variant']}",
        "metric": "difference_with_minus_without",
        "value": float(row["difference_with_minus_without"]),
        "n_sessions": int(row["n_sessions"]), "source_run": entry["run_dir"],
    } for row in _read_csv(_artifact(entry, f"contrasts_f_{entry['band']}.csv"))]

    hr = metrics.get("heart_rate_question") or {}
    exclusions = [{
        "experiment": "f", "band": entry["band"], "unit": "variable", "identifier": variable,
        "reason": f"{hr.get('status')} (n_hr_observations={hr.get('n_hr_observations')})"
        if variable == "heart_rate" else "uncontrolled: not measured",
        "source_run": entry["run_dir"],
    } for variable in hr.get("uncontrolled_variables", [])]
    return {"headline": [], "per_subject": per_subject, "paired": paired,
            "exclusions": exclusions}


def adapt_g(entry) -> dict:
    """Exp G: cross-band, so its rows carry a blank band. The fusion contrast is the estimand;
    the per-condition MAEs come from the saved per-subject table."""
    metrics, artifact = _metrics_of(entry)
    n_subjects = metrics.get("n_subjects_g", _BLANK)
    per_subject = []
    for row in _read_csv(_artifact(entry, "per_subject_g.csv")):
        for condition in ("10", "77", "equal_weight", "fused"):
            per_subject.append({
                "experiment": "g", "band": _BLANK, "subject": int(row["subject"]),
                "model_or_contrast": condition, "metric": "subject_mae",
                "value": float(row[f"mae_{condition}"]),
                "n_sessions": int(row["n_sessions"]), "source_run": entry["run_dir"],
            })
    # G's `primary` is a WRAPPER (estimand, direction, sign, n_subjects) around the CI dict —
    # not itself a CI dict. Reading it as one would silently blank every interval column.
    primary = metrics.get("primary") or {}
    paired = []
    if primary:
        paired.append(_paired(
            entry, artifact, "fused_minus_10ghz",
            primary.get("mean_difference_fused_minus_10", {}),
            p_value=_BLANK,          # G's primary is a CI/sign estimand, with no paired test
            n_pairs=primary.get("n_subjects", n_subjects),
            tier="primary", family="single_uncorrected",
            test="subject_cluster_bootstrap_ci"))
    return {"headline": [], "per_subject": per_subject, "paired": paired, "exclusions": []}


def adapt_robustness(entry) -> dict:
    """H robustness: an empirical selection-variance RANGE, never a BCa interval.

    The label is carried through verbatim from the artifact rather than reconstructed, because
    mislabelling this as a confidence interval is the single most consequential error available
    in the whole assembly (§7: "its empirical range is not mislabeled BCa").
    """
    _, artifact = _metrics_of(entry)
    headline = []
    for row in _read_csv(_artifact(entry, "robustness_summary.csv")):
        headline.append({
            "experiment": "robustness", "band": row.get("band") or _BLANK,
            "model_or_contrast": row.get("arm_or_contrast", _BLANK),
            "metric": f"{row.get('experiment', '')}_selection_variance",
            "estimate": row.get("original_point", _BLANK),
            "ci_low": row.get("range_low", _BLANK),
            "ci_high": row.get("range_high", _BLANK),
            "ci_method": row.get("range_label", "selection_variance_empirical_95pct_range"),
            "n_subjects": _BLANK, "n_sessions": _BLANK,
            "status": row.get("status", _BLANK), "primary_or_secondary": "secondary",
            "source_run": entry["run_dir"], "source_artifact": artifact,
        })
    return {"headline": headline, "per_subject": [], "paired": [], "exclusions": []}


ADAPTERS = {
    "a": adapt_a, "b": adapt_b, "c": adapt_c, "d": adapt_d,
    "e": adapt_e, "f": adapt_f, "g": adapt_g, "robustness": adapt_robustness,
}


# ------------------------------------------------------------------------- final tables


def assemble(manifest: dict) -> dict:
    """Run every registered source through its OWN adapter and collect the final tables."""
    validate_manifest(manifest)
    tables = {"headline": [], "per_subject": [], "paired": [], "exclusions": []}
    descriptive = {}
    for key, entry in sorted(manifest["runs"].items()):
        adapter = ADAPTERS[entry["experiment"]]
        produced = adapter(entry)
        for name in tables:
            tables[name].extend(produced.get(name, []))
        if "descriptive" in produced:
            descriptive[key] = produced["descriptive"]
    tables["descriptive"] = descriptive
    return tables


def milestone_summary(manifest: dict, tables: dict) -> dict:
    """`metrics_milestone10.json`: the lineage, the counts, and what is deliberately absent."""
    return {
        "schema_version": "milestone10_summary_v1",
        "discovery": manifest["discovery"],
        "sources": {
            key: {"experiment": entry["experiment"], "band": entry["band"],
                  "run_dir": entry["run_dir"], "source_commit": entry["source_commit"],
                  "resolved_config_sha256": entry["resolved_config_sha256"],
                  "artifact_sha256": {n: r["sha256"] for n, r in sorted(entry["artifacts"].items())}}
            for key, entry in sorted(manifest["runs"].items())
        },
        "counts": {name: len(tables[name]) for name in
                   ("headline", "per_subject", "paired", "exclusions")},
        "descriptive_only": tables.get("descriptive", {}),
        "notes": [
            "Experiment E contributes no headline row: its artifact states no estimate. Its "
            "per-path attribution lives in importance_summary_{band}.csv and is descriptive.",
            "The robustness rows report an empirical selection-variance range, NOT a BCa "
            "confidence interval; the label is carried through from the source artifact.",
            "Every row traces to an explicit run directory; no source was discovered by glob.",
            "All intervals are conditional/exploratory in this 16-subject feasibility study.",
        ],
    }


def _write_csv(path, columns, rows) -> Path:
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def write_assembly_reports(manifest, tables, summary, out_dir) -> dict:
    """`run_manifest.json` plus the four final tables and the milestone summary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    manifest_path = out_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    paths["run_manifest"] = manifest_path
    paths["headline"] = _write_csv(out_dir / "headline_metrics.csv", HEADLINE_COLUMNS,
                                   tables["headline"])
    paths["per_subject"] = _write_csv(out_dir / "per_subject_results.csv", PER_SUBJECT_COLUMNS,
                                      tables["per_subject"])
    paths["paired"] = _write_csv(out_dir / "paired_comparisons.csv", PAIRED_COLUMNS,
                                 tables["paired"])
    paths["exclusions"] = _write_csv(out_dir / "analysis_exclusions.csv", EXCLUSIONS_COLUMNS,
                                     tables["exclusions"])
    metrics_path = out_dir / "metrics_milestone10.json"
    metrics_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    paths["metrics"] = metrics_path
    return paths


def assemble_and_report(sources, out_dir) -> dict:
    """The whole step: explicit sources -> manifest -> adapters -> final tables."""
    manifest = build_run_manifest(sources)
    tables = assemble(manifest)
    summary = milestone_summary(manifest, tables)
    return write_assembly_reports(manifest, tables, summary, out_dir)
