"""Bit-exact comparison of two per-session WST feature stores.

    python scripts/compare_stores.py <reference_band_dir> <candidate_band_dir>

Each argument is a band directory — `<results>/features/<band>` — holding `s<subj>_<sess>.npz`.

Written for M9's O-M9-5 investigation, where the question was whether a store rebuilt at a
later commit was value-identical to the one an earlier commit produced. The answer has to be
BIT-exact, not "close": the whole point is to decide whether a downstream float difference
could have originated in the features, and an `np.allclose` here would beg that question.

The reference store is authoritative: every key it holds must exist in the candidate and
match byte-for-byte. Keys the CANDIDATE alone holds are reported, not failed — a store schema
can legitimately gain arrays (v2 added `sig__raw_beat` / `sig__matched_iq` for Exp D, which
M7 never wrote), and failing on that would make the tool useless across schema versions.

Comparison is on `ndarray.tobytes()` rather than `==`, so NaN-vs-NaN registers as equal and
+0.0-vs-−0.0 registers as different — both the opposite of what float comparison would give,
and both the behaviour you want when asking "did this bit pattern change?".

To compare a store against one an OLD commit builds, check that commit out in a throwaway
worktree, build into a scratch results dir, and point this at both. Two traps, each of which
silently invalidates the result:
  * the paths overlay must live OUTSIDE the worktree — provenance derives `dirty` from
    `git status --porcelain`, which counts untracked files, so an overlay inside it trips
    `assert_clean_tree`;
  * `dehyd` is installed editable, so PYTHONPATH must point at the worktree's `src` or the
    old checkout silently runs NEW code. Verify before trusting anything: each store's
    `.fingerprint.json` records the commit of the tree the code was imported from, because
    `provenance._git_info` runs git in the importing module's own directory.
"""
import sys
from pathlib import Path

import numpy as np


def compare_stores(reference_dir, candidate_dir) -> int:
    reference_dir, candidate_dir = Path(reference_dir), Path(candidate_dir)
    stems = sorted(p.stem for p in reference_dir.glob("*.npz"))
    if not stems:
        print(f"ERROR: no .npz under {reference_dir}", file=sys.stderr)
        return 2

    n_arrays, differences, missing, candidate_only_keys = 0, [], [], set()
    for stem in stems:
        candidate_path = candidate_dir / f"{stem}.npz"
        if not candidate_path.is_file():
            missing.append(stem)
            continue
        with np.load(reference_dir / f"{stem}.npz") as reference, np.load(candidate_path) as candidate:
            reference_keys, candidate_keys = set(reference.files), set(candidate.files)
            candidate_only_keys |= candidate_keys - reference_keys
            for key in sorted(reference_keys - candidate_keys):
                differences.append((stem, key, "in the reference, ABSENT from the candidate"))
            for key in sorted(reference_keys & candidate_keys):
                a, b = reference[key], candidate[key]
                n_arrays += 1
                if a.shape != b.shape or a.dtype != b.dtype:
                    differences.append(
                        (stem, key, f"{a.shape}/{a.dtype} vs {b.shape}/{b.dtype}")
                    )
                elif a.tobytes() != b.tobytes():
                    delta = np.abs(a.astype(float) - b.astype(float))
                    differences.append((stem, key, f"max|delta| = {np.nanmax(delta):.3e}"))

    print(f"sessions compared : {len(stems) - len(missing)}")
    print(f"shared arrays     : {n_arrays}")
    print(f"candidate-only keys (schema additions): {sorted(candidate_only_keys)}")
    if missing:
        print(f"MISSING from the candidate : {missing}")
    ok = not differences and not missing
    print("RESULT: BIT-IDENTICAL" if ok else f"RESULT: {len(differences)} DIFFERING arrays")
    for stem, key, detail in differences[:40]:
        print(f"  {stem:16s} {key:28s} {detail}")
    if len(differences) > 40:
        print(f"  ... and {len(differences) - 40} more")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(compare_stores(sys.argv[1], sys.argv[2]))
