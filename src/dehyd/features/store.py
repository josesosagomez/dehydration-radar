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
from ..provenance import _git_info, _package_versions, sha256_file

STORE_VERSION = 2
POOLING_CONTRACT_VERSION = "pool_stats_v1"  # bump if the pooling element order ever changes
FRAME_SELECTION = "qc_pass_frames_of_eligible_sessions"

# Schema v2 (milestone 9): the per-frame SIGNAL arrays Experiment D consumes, stored in the
# same canonical QC-passed frame order as the raw WST tensors. They are kept UNSTANDARDIZED
# where the frozen input definition says "raw" — the physics baseline's power ratio is an
# absolute-magnitude quantity that a robust z would destroy (MILESTONE_9_PLAN §5 trap 13).
# The names are also written in `models/cnn.py::FRAME_INPUT` and
# `eval/exp_d.py::PHYSICS_SIGNAL_KEY`; the two sources are cross-checked by test, never
# derived from each other.
SIG_RAW_BEAT_10GHZ = "sig__raw_beat"        # [N, 534] complex128
SIG_RAW_SLOWTIME_77GHZ = "sig__raw_slowtime"  # [N, 256] float64
SIG_MATCHED_IQ = "sig__matched_iq"          # 10 GHz [N, 2, 470]; 77 GHz [N, 2, 256]; float64


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
    """The fingerprint binding a store to what produced it.

    Every field above `packages` records what the store was built FROM — the code revision,
    the spec, the QC config, the exact frame membership, the raw bytes. `packages` records
    what it was built WITH, and it is deliberately NOT one of the fields `_check_match`
    compares.

    Why it exists: M9 lost two days to a 5.14e-14 divergence between an M7 Exp A artifact and
    its re-run. Data, splits, seeds, model selection, node hardware, run-to-run determinism
    and the store rebuild were each eliminated by direct test, and M7's own code was shown to
    reproduce the store bit-for-bit — so the cause lay in the realized numerical environment.
    That could not be checked, because nothing recorded it. The declared versions in uv.lock
    were identical throughout; what differed was what was actually installed.

    Why it does not fail closed: a store must be attributable to a clean revision, not to a
    byte-identical environment. Refusing a store because scipy moved a patch version would
    make every environment refresh a full re-extraction, and the numerical differences at
    stake are ~1e-13 and provably never changed a model selection. This is diagnostic
    evidence for a future investigator, not an acceptance criterion.
    """
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
        "packages": _package_versions(),
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


def session_signals_10ghz(sub: np.ndarray, config: Config) -> dict:
    """The two 10 GHz Exp D signal arrays for one session's SELECTED frames.

    `sub` is `cube[:, :, frame_ids]` — the QC-passed frames in canonical order — so both
    outputs are row-aligned with `frame_ids` and with the raw WST tensors built from the
    same `sub`.

      * `sig__raw_beat` — the chirp mean (`reduce_option_a`) of the **raw, ungated** frame:
        no bandpass, no EdgeTrim, no standardization, 534 complex samples. This is the
        frozen "raw beat" input of baseline (i) *and* the physics baseline's input
        (`implementation_plan.md:891-896`, `:921-935`); the physics power ratio needs
        absolute magnitudes, which is why nothing is normalized here.
      * `sig__matched_iq` — literally `preprocess_cube(..., reduction="a", channel="iq")` at
        the DEFAULT model gate, i.e. the same bandpassed/trimmed/robust-standardized
        470-sample I/Q signal the WST chain consumes (`:902-904`). One definition, two
        consumers: the CNN ablation reads this array untouched.
    """
    from ..preprocess.pipeline import preprocess_cube
    from ..preprocess.reduce import reduce_option_a

    raw_beat = np.stack(
        [reduce_option_a(sub[:, :, i]) for i in range(sub.shape[2])]
    ).astype(np.complex128)
    matched = preprocess_cube(sub, config.preprocess, reduction="a", channel="iq")
    return {
        SIG_RAW_BEAT_10GHZ: raw_beat,
        SIG_MATCHED_IQ: np.asarray(matched, dtype=np.float64),
    }


def session_signals_77ghz(cube: np.ndarray, config: Config) -> dict:
    """The two 77 GHz Exp D signal arrays for one session's eligible-frame cube
    ([N, n_fast, n_chirp, n_rx] — the caller has already selected the frames).

      * `sig__raw_slowtime` — A-M6-2 (i): a plain mean over the 256 fast-time bins and the
        16 Rx of the RAW real cube, leaving the one axis the band's design rests on (chirp /
        slow time). Unstandardized, for the same physics-baseline reason as 10 GHz.
      * `sig__matched_iq` — A-M6-2 (i-ablation): chain steps 1-5 (`preprocess_frame_77`),
        the single fixed representative **Rx 0**, mean over that Rx's 27 gate range bins,
        as {real, imag}. Stored PRE-standardization: `models/cnn.py::matched_input_77`
        applies the frozen robust per-channel z at load, and the physics path must never
        see a standardized array.
    """
    from ..preprocess.pipeline_77 import preprocess_frame_77

    cube = np.asarray(cube)
    raw_slowtime = cube.mean(axis=(1, 3)).astype(np.float64)
    matched = []
    for i in range(cube.shape[0]):
        gate_mean = preprocess_frame_77(cube[i], config.preprocess77)[:, :, 0].mean(axis=0)
        matched.append(np.stack([gate_mean.real, gate_mean.imag]))
    return {
        SIG_RAW_SLOWTIME_77GHZ: raw_slowtime,
        SIG_MATCHED_IQ: np.asarray(np.stack(matched), dtype=np.float64),
    }


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
    npz.update(session_signals_10ghz(sub, config))   # schema v2: the Exp D per-frame signals
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
    npz.update(session_signals_77ghz(cube, config))  # schema v2: the Exp D per-frame signals
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
