"""Persistent per-session feature store + fail-closed fingerprint validation (M7).

The 77 GHz session feature vectors do not exist as a persistent artifact (only the M5
diagnostics CSV), and the tuned-ε branch cannot be precomputed as a session vector — it
needs the RAW pre-log scattering tensors. So the store holds, per (band, subject, session):

  * data-independent pooled session vectors (log off + on_frozen_eps),
  * the per-order pre-log scales,
  * the RAW pre-log per-frame scattering tensors (the tuned-ε input),
  * the per-tiling kymatio meta order,

as one `.npz` plus a `.fingerprint.json` sidecar. Keys are flat (`a__b__c`) so nothing
depends on zip-internal path handling.

The fingerprint BINDS a store to the exact code/config/QC/data that produced it, and
`validate_store` is fail-closed (the proven `--merge-shards` doctrine): it refuses to run
if any expected session is missing or any sidecar disagrees with what THIS run would
produce — including the exact QC-selected frame membership (`frame_ids_sha256`, C4) and the
building git commit vs the analysis commit (C16).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..config import Config, config_to_dict
from ..provenance import _git_info, sha256_file

STORE_VERSION = 1
POOLING_CONTRACT_VERSION = "pool_stats_v1"  # bump if the pooling element order ever changes
FRAME_SELECTION = "qc_pass_frames_of_eligible_sessions"


class StoreError(RuntimeError):
    """A missing session file, an unreadable store, or a stale/mismatched fingerprint."""


# ------------------------------------------------------------------- key builders


def vec_key(g, r, c, t, branch):        # branch in {"off", "frozen"}
    return f"vec__g{g}__{r}__{c}__t{t}__{branch}"


def raw_key(g, r, c, t):
    return f"raw__g{g}__{r}__{c}__t{t}"


def prelog_key(g, r, c, t):
    return f"prelog__g{g}__{r}__{c}__t{t}"


def order_key(t):
    return f"order__t{t}"


def session_stem(subject, session) -> str:
    return f"s{int(subject)}_{session}"


# 77 GHz keys: fixed reduction/channel/gate and the single primary fusion "mean", so keyed
# by tiling + branch only.
def vec77_key(t, branch):
    return f"vec__t{t}__{branch}"


def raw77_key(t):
    return f"raw__t{t}"


def prelog77_key(t):
    return f"prelog__t{t}"


# --------------------------------------------------------------------- fingerprint


def _sorted_json_sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def spec_hash(config: Config, band: str) -> str:
    """Bind the store to the band's canonical feature spec + gate whitelist + pooling order."""
    cfg = config_to_dict(config)
    if band == "10ghz":
        spec = {
            "preprocess": cfg["preprocess"],
            "wst": cfg["wst"],
            "range_gate_m": cfg["search_10ghz"]["range_gate_m"],
        }
    elif band == "77ghz":
        spec = {
            "preprocess77": cfg["preprocess77"],
            "wst77": cfg["wst77"],
            "gate_m": cfg["search_77ghz"]["gate_m"],
        }
    else:
        raise StoreError(f"unknown band {band!r}")
    spec["pooling_contract"] = POOLING_CONTRACT_VERSION
    return _sorted_json_sha(spec)


def qc_config_hash(config: Config, band: str) -> str:
    cfg = config_to_dict(config)
    qc = cfg["qc"] if band == "10ghz" else cfg["qc77"]
    return _sorted_json_sha(qc)


def frame_ids_sha256(frame_ids) -> str:
    """Canonical hash of the ORDERED selected frame indices — catches a same-count
    membership substitution that `n_frames` + `raw_sha256` alone would miss (C4)."""
    arr = np.asarray(sorted(int(i) for i in frame_ids), dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def compute_fingerprint(config: Config, band: str, *, frame_ids, raw_path, session_eligible: bool) -> dict:
    return {
        "git": _git_info(),
        "spec_hash": spec_hash(config, band),
        "qc_config_hash": qc_config_hash(config, band),
        "frame_selection": FRAME_SELECTION,
        "frame_ids_sha256": frame_ids_sha256(frame_ids),
        "n_frames": int(len(list(frame_ids))),
        "raw_sha256": sha256_file(raw_path),
        "session_eligible": bool(session_eligible),
        "store_version": STORE_VERSION,
    }


# ------------------------------------------------------------------ write / read


def band_dir(store_dir, band) -> Path:
    return Path(store_dir) / "features" / band


def write_session_store(band, subject, session, npz_dict, fingerprint, store_dir) -> Path:
    out_dir = band_dir(store_dir, band)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = session_stem(subject, session)
    npz_path = out_dir / f"{stem}.npz"
    np.savez(npz_path, **npz_dict)
    (out_dir / f"{stem}.fingerprint.json").write_text(
        json.dumps(fingerprint, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return npz_path


class LazyStore:
    """Lazy per-key access to one session's `.npz` (arrays load on first access)."""

    def __init__(self, npz_path: Path, fingerprint: dict):
        self._npz = np.load(npz_path, allow_pickle=False)
        self.fingerprint = fingerprint

    def __getitem__(self, key):
        return self._npz[key]

    def __contains__(self, key):
        return key in self._npz.files

    def keys(self):
        return list(self._npz.files)

    def close(self):
        self._npz.close()


def read_fingerprint(store_dir, band, subject, session) -> dict:
    path = band_dir(store_dir, band) / f"{session_stem(subject, session)}.fingerprint.json"
    if not path.is_file():
        raise StoreError(f"missing fingerprint for {band} {subject}/{session}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_session_store(band, subject, session, store_dir) -> LazyStore:
    stem = session_stem(subject, session)
    npz_path = band_dir(store_dir, band) / f"{stem}.npz"
    if not npz_path.is_file():
        raise StoreError(f"missing store file for {band} {subject}/{session}: {npz_path}")
    return LazyStore(npz_path, read_fingerprint(store_dir, band, subject, session))


# ------------------------------------------------------------------ validation


def _check_match(sidecar: dict, expected: dict, analysis_commit, where: str) -> None:
    for key in ("spec_hash", "qc_config_hash", "frame_ids_sha256", "n_frames",
                "frame_selection", "store_version", "session_eligible"):
        if sidecar.get(key) != expected.get(key):
            raise StoreError(
                f"stale/mismatched store at {where}: {key} is {sidecar.get(key)!r}, "
                f"expected {expected.get(key)!r} — refusing to run (fail-closed)"
            )
    store_commit = (sidecar.get("git") or {}).get("commit")
    if analysis_commit is not None and store_commit != analysis_commit:
        raise StoreError(
            f"store/analysis commit mismatch at {where}: store built at {store_commit!r}, "
            f"analysis at {analysis_commit!r} — a store must back only its own revision (C16)"
        )


def assert_clean_tree() -> None:
    """Refuse to build a feature store unless the working tree is clean and committed (C7/C16).

    A store must be attributable to a single clean revision, so BOTH producers (the local
    `extract_features.py` and the IBEX shard) call this before writing anything. A dirty tree
    or an unrecorded commit would make `DEHYD_GIT_COMMIT` point at the wrong (or no) code.
    """
    git = _git_info()
    if git.get("dirty"):
        raise StoreError(
            "refusing to build a feature store from a DIRTY tree — commit the milestone-7 "
            "code first (C7/C16); a store must be attributable to a clean revision"
        )
    if git.get("commit") is None:
        raise StoreError("refusing to build a feature store with no git commit recorded")


def build_session_npz_10ghz(cube: np.ndarray, frame_ids, config: Config) -> dict:
    """Package one 10 GHz session's store arrays: per gate x reduction x channel x tiling the
    off/frozen pooled session vectors, the raw pre-log tensor, the pre-log scales, and the
    per-tiling meta order. Extraction runs once per (gate, reduction, channel) via the
    M4-tested `extract_session_variants(keep_raw=True)` — no parallel loop."""
    import dataclasses

    from .extraction import extract_session_variants

    sub = cube[:, :, list(frame_ids)]
    npz: dict = {}
    for gi, gate in enumerate(config.search_10ghz.range_gate_m):
        pre = dataclasses.replace(config.preprocess, model_gate_m=tuple(gate))
        for r in config.search_10ghz.reduction:      # ("A", "B")
            for c in config.search_10ghz.channel:    # ("mag", "iq")
                res = extract_session_variants(
                    sub, pre, config.wst, reduction=r.lower(), channel=c, keep_raw=True
                )
                for ti in range(len(config.wst.tilings)):
                    npz[vec_key(gi, r, c, ti, "off")] = res.vectors[(ti, False, "pooled")]
                    npz[vec_key(gi, r, c, ti, "frozen")] = res.vectors[(ti, True, "pooled")]
                    npz[raw_key(gi, r, c, ti)] = res.raw[ti]["S"]
                    npz[prelog_key(gi, r, c, ti)] = np.asarray(res.prelog_scale[ti], dtype=float)
                    npz[order_key(ti)] = res.raw[ti]["order"]
    return npz


def build_session_npz_77ghz(cube: np.ndarray, config: Config) -> dict:
    """Package one 77 GHz session's store arrays (primary fusion "mean"): per tiling the
    off/frozen pooled vectors, raw pre-log tensor, pre-log scale, and meta order. The 77 GHz
    chain has no gate/reduction/channel search axes (Exp G freezes them), so keys are keyed by
    tiling + branch only. NB: 77 GHz frame QC/eligibility is applied by the caller before this
    (the cube passed in is already the eligible-frame cube)."""
    from .extraction_77 import PRIMARY_FUSION_77, extract_session_variants_77

    res = extract_session_variants_77(cube, config.preprocess77, config.wst77, keep_raw=True)
    fusion = PRIMARY_FUSION_77
    npz: dict = {}
    for ti in range(len(config.wst77.tilings)):
        npz[vec77_key(ti, "off")] = res.vectors[(ti, "off", fusion, "pooled")]
        npz[vec77_key(ti, "frozen")] = res.vectors[(ti, "on_frozen_eps", fusion, "pooled")]
        npz[raw77_key(ti)] = res.raw[ti]["S"]
        npz[prelog77_key(ti)] = np.asarray(res.prelog_scale[(ti, fusion)], dtype=float)
        npz[order_key(ti)] = res.raw[ti]["order"]
    return npz


def validate_store(band, store_dir, expected_fingerprints: dict, *, analysis_commit) -> None:
    """Fail closed unless every expected session is present and its sidecar matches exactly.

    `expected_fingerprints`: {(subject, session): fingerprint} the CALLER computed from the
    QC'd manifest + config (so `frame_ids_sha256` reflects the frames THIS run selected). A
    changed/stale QC manifest, a config drift, or a store built at a different commit all
    fail here rather than silently backing the analysis.
    """
    for (subject, session), expected in expected_fingerprints.items():
        where = f"{band} {subject}/{session}"
        sidecar = read_fingerprint(store_dir, band, subject, session)  # raises if missing
        _check_match(sidecar, expected, analysis_commit, where)
