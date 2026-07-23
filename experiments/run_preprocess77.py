"""Milestone 5 — 77 GHz preprocessing chain diagnostics (band 2).

A SINGLE cohort job (not sharded): the chain steps 1-5 are light vs WST, so this writes
exactly one CSV and needs no shard/merge protocol (avoids the array-race hazard). Per
(subject, session) it records the per-stage energies (à la the audit's chain_stages), the
MTI removal fraction, and the gate-crop energy ratio -> results/preprocess/
preprocess_diagnostics_77ghz.csv. Every file is axis-certified before it is touched.

    uv run python experiments/run_preprocess77.py --config configs/exp_77ghz.yaml [--subject N --session S]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dehyd.config import load_config, require_77ghz_dir  # noqa: E402
from dehyd.data.ground_truth import load_ground_truth  # noqa: E402
from dehyd.data.loader_77ghz import load_77ghz_file  # noqa: E402
from dehyd.data.manifest_77 import build_manifest_77, resolve_path_77  # noqa: E402
from dehyd.data.sessions import SESSION_NAMES  # noqa: E402
from dehyd.preprocess.pipeline_77 import chain_stages_77  # noqa: E402
from dehyd.provenance import record_run  # noqa: E402
from dehyd.qc.axis_check_77 import require_accepted_axis  # noqa: E402

DIAG_NAME = "preprocess_diagnostics_77ghz.csv"


def _session_row(cube, pre77, subject, session_name, rel_path):
    """Mean per-stage energies over the session's frames + derived ratios."""
    per_frame = [chain_stages_77(cube[i], pre77) for i in range(cube.shape[0])]
    names = [s["stage"] for s in per_frame[0]]
    mean_energy = {n: float(np.mean([f[k]["energy"] for f in per_frame]))
                   for k, n in enumerate(names)}
    raw, mti, crop = mean_energy["raw_frame"], mean_energy["mti"], mean_energy["range_gate_crop"]
    return {
        "subject": subject, "session_name": session_name, "rel_path": rel_path,
        "n_frames": cube.shape[0],
        **{f"energy_{n}": mean_energy[n] for n in names},
        "mti_removal_fraction": 1.0 - mti / raw if raw else float("nan"),
        "gate_crop_ratio_to_raw": crop / raw if raw else float("nan"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", action="append", required=True, metavar="PATH")
    parser.add_argument("--subject", type=int, default=None)
    parser.add_argument("--session", choices=SESSION_NAMES, default=None)
    args = parser.parse_args(argv)

    config = load_config(*args.config)
    require_77ghz_dir(config)
    print(f"config       : {', '.join(args.config)}")

    gt = load_ground_truth(config.paths.weight_xlsx)
    manifest = build_manifest_77(config.paths, gt)
    files = manifest[["subject", "session_name", "rel_path"]].drop_duplicates()
    if args.subject is not None:
        files = files[files["subject"] == args.subject]
    if args.session is not None:
        files = files[files["session_name"] == args.session]

    survival = Path(config.paths.results_dir) / "qc" / "qc_survival_77ghz.csv"
    rows = []
    for f in files.itertuples(index=False):
        path = resolve_path_77(config.paths, f.rel_path)
        require_accepted_axis(path, config, survival_csv=survival if survival.exists() else None)
        cube = load_77ghz_file(path)
        rows.append(_session_row(cube, config.preprocess77, f.subject, f.session_name, f.rel_path))
        print(f"  {f.rel_path}: MTI removes {rows[-1]['mti_removal_fraction']:.3f}, "
              f"gate/raw {rows[-1]['gate_crop_ratio_to_raw']:.3e}")

    out_dir = Path(config.paths.results_dir) / "preprocess"
    out_dir.mkdir(parents=True, exist_ok=True)
    diag = out_dir / DIAG_NAME
    pd.DataFrame(rows).to_csv(diag, index=False)
    assert len(pd.read_csv(diag)) == len(rows), "diagnostics lost rows on write"
    print(f"diagnostics  : {diag}")

    record_run(config, manifest, folds=None, data_dir=require_77ghz_dir(config),
               extra={"stage": "milestone-5-preprocess77", "n_sessions": len(rows)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
