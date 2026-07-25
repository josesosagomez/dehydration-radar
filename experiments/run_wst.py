"""Milestone 4 — run the frozen WST over the eligible 10 GHz cohort (first contact).

Writes the per-session WST diagnostics: feature dimensionality (nominal / effective /
raw), cohort finiteness across every variant, the PRE-LOG order-0/1/2 coefficient scale
(does epsilon = 1e-6 sit negligibly below it?), and per-session timing — plus a
provenance record carrying the measured `wst_spec`.

    uv run python experiments/run_wst.py --config configs/exp_a_regression.yaml

**Diagnostic only — this script selects nothing.** Every constant is frozen before it
runs; a surprising distribution is a finding for HISTORY.md and the owner, never a licence
to retune (the M2/M3 doctrine). All extraction logic lives in `dehyd.features.extraction`
(library code); `main()` only sequences it, so the reusable wiring is tested directly and
the M7 harness imports a module, not this script.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dehyd.config import load_config
from dehyd.data.ground_truth import load_ground_truth
from dehyd.data.loader_10ghz import load_10ghz_file
from dehyd.data.manifest import apply_qc, build_manifest, eligible_frames, resolve_path
from dehyd.features.extraction import canonical_spec_guard, extract_session_variants
from dehyd.features.pooling import feature_layout, flat_layout
from dehyd.features.wst import wst_spec
from dehyd.provenance import record_run

REPORT_NAME = "wst_diagnostics_10ghz.csv"

REDUCTIONS = ("a", "b")
CHANNELS = ("mag", "iq")


def _channel_count(channel: str) -> int:
    return 1 if channel == "mag" else 2


def session_diagnostics(cube, frame_indices, pre, wst_cfg, spec) -> dict:
    """Every per-session WST number, from the scatter-once variant pass per (reduction, channel).

    Dimensionality is read from the recorded layouts (nominal 6 stats/path, effective per
    the >=2-sample segment-std rule, raw = C*n_paths*n_time); finiteness spans every
    (reduction x channel x tiling x log x family) variant; the pre-log scale is the
    order-0/1/2 statistic (identical under both log states by construction).
    """
    qc_cube = cube[:, :, list(frame_indices)]
    row: dict = {"n_eligible_frames": len(frame_indices)}
    all_finite = True
    t0 = time.perf_counter()

    prelog = {ti: [] for ti in range(len(wst_cfg.tilings))}
    for reduction in REDUCTIONS:
        for channel in CHANNELS:
            result = extract_session_variants(
                qc_cube, pre, wst_cfg, reduction=reduction, channel=channel
            )
            all_finite = all_finite and result.all_finite
            for ti in range(len(wst_cfg.tilings)):
                prelog[ti].append(result.prelog_scale[ti])

    # Pre-log coefficient scale, median over the (reduction, channel) passes per tiling.
    for ti in range(len(wst_cfg.tilings)):
        arr = np.asarray(prelog[ti])  # [4 passes, 3 orders]
        for order in (0, 1, 2):
            row[f"t{ti}_prelog_order{order}"] = float(np.median(arr[:, order]))

    row["all_variants_finite"] = all_finite
    row["wst_seconds"] = time.perf_counter() - t0
    return row


def dimension_summary(wst_cfg, spec) -> list[dict]:
    """Constant-across-sessions feature dimensions per (tiling, channel), from the layouts."""
    rows = []
    for ti, t in enumerate(spec["tilings"]):
        n_paths, n_time = t["n_paths"], t["n_time"]
        for channel in CHANNELS:
            c = _channel_count(channel)
            # A dummy meta with the right path count is all the layouts need.
            meta = {"order": np.zeros(n_paths, dtype=int)}
            nominal = c * n_paths * 6
            effective = len(feature_layout(meta, n_time, c))
            raw = len(flat_layout(meta, n_time, c))
            rows.append(
                {
                    "tiling": ti,
                    "channel": channel,
                    "n_paths": n_paths,
                    "n_time": n_time,
                    "pooled_nominal": nominal,
                    "pooled_effective": effective,
                    "pooled_session_dim": 2 * effective,
                    "raw_frame_dim": raw,
                    "raw_session_dim": 2 * raw,
                }
            )
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", action="append", required=True, metavar="PATH",
                        help="config YAML; repeatable, later files win")
    args = parser.parse_args(argv)

    config = load_config(*args.config)
    canonical_spec_guard(config)  # before any I/O
    pre, wst_cfg = config.preprocess, config.wst
    spec = wst_spec(wst_cfg, pre)

    print(f"config  : {', '.join(args.config)}")
    print(f"backend : {wst_cfg.backend} | max_order {wst_cfg.max_order} | eps {wst_cfg.log_epsilon}")
    for ti, t in enumerate(spec["tilings"]):
        print(
            f"  T{ti+1} Q={t['q']} {t['requested_ms']} ms -> T={t['t_samples']} "
            f"(realized {t['realized_ms']:.4f} ms, err {t['realized_error_frac']*100:.3f}%) "
            f"J={t['J']} -> {t['n_paths']} paths x {t['n_time']} time, pad {t['pad_left']}/{t['pad_right']}"
        )

    dims = dimension_summary(wst_cfg, spec)
    print("\nfeature dimensions (constant across sessions):")
    for d in dims:
        print(
            f"  T{d['tiling']+1} {d['channel']:<3} pooled nominal {d['pooled_nominal']} "
            f"effective {d['pooled_effective']} (session {d['pooled_session_dim']}) | "
            f"raw frame {d['raw_frame_dim']} (session {d['raw_session_dim']})"
        )

    gt = load_ground_truth(config.paths.weight_xlsx)
    manifest = build_manifest(config.paths, gt)
    manifest_qc = apply_qc(manifest, config.paths, config)
    population = eligible_frames(manifest_qc)
    print(
        f"\npopulation: {len(population)} eligible frames, "
        f"{population.groupby(['subject', 'session_idx']).ngroups} sessions, "
        f"{population.subject.nunique()} subjects"
    )

    rows = []
    for (subject, session_idx), group in population.groupby(["subject", "session_idx"]):
        cube = load_10ghz_file(resolve_path(config.paths, group["rel_path"].iloc[0]))
        diag = session_diagnostics(cube, group["frame_idx"].tolist(), pre, wst_cfg, spec)
        rows.append(
            {
                "subject": int(subject),
                "session_idx": int(session_idx),
                "session_name": group["session_name"].iloc[0],
                **diag,
            }
        )
        print(
            f"  s{subject:<2} {group['session_name'].iloc[0]:<5} "
            f"n={diag['n_eligible_frames']:<3} finite={diag['all_variants_finite']} "
            f"{diag['wst_seconds']:.1f}s"
        )

    report = pd.DataFrame(rows)
    print("\ncohort summary")
    print(f"  sessions            : {len(report)}")
    print(f"  all variants finite : {bool(report['all_variants_finite'].all())}")
    print(f"  total WST time      : {report['wst_seconds'].sum():.1f}s")
    # The pre-log scale is a FINDING, reported with the eps/scale ratio so the reader
    # judges rather than being told: eps is negligible vs order 1 (~1e-3) but a material
    # fraction of the tiny order-2 scale (~1e-6). eps stays frozen; log on/off is the M6
    # inner-CV axis that decides whether this matters. (See HISTORY / SECOND_CHAPTER 3.)
    for ti in range(len(wst_cfg.tilings)):
        o1 = report[f"t{ti}_prelog_order1"].median()
        o2 = report[f"t{ti}_prelog_order2"].median()
        print(
            f"  T{ti+1} pre-log scale   : order0 {report[f't{ti}_prelog_order0'].median():+.3e} "
            f"order1 {o1:.3e} (eps/scale {wst_cfg.log_epsilon / abs(o1):.1e}) "
            f"order2 {o2:.3e} (eps/scale {wst_cfg.log_epsilon / abs(o2):.2f})"
        )

    out_dir = Path(config.paths.results_dir) / "wst"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / REPORT_NAME
    report.to_csv(report_path, index=False)
    written = pd.read_csv(report_path)
    assert len(written) == len(report), "diagnostics report lost rows on write"
    assert bool(written["all_variants_finite"].all()), "a variant produced non-finite output"
    print(f"\nreport    : {report_path}")

    provenance_path = record_run(
        config, manifest_qc, folds=None,
        extra={
            "stage": "milestone-4-wst",
            "analysis_role": "primary",
            "backend": wst_cfg.backend,
            "wst_spec": spec,
            "feature_dimensions": dims,
            "n_sessions": len(report),
            "n_eligible_frames": int(report["n_eligible_frames"].sum()),
        },
    )
    print(f"provenance: {provenance_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
