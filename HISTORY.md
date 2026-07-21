# HISTORY — implementation log

Running record of every attempt, newest-first. Each entry: what was tried, whether it
succeeded/failed **and why**, and the concrete parameter values + reasoning. Failures
stay in the log. A new session reads only the most recent entries to orient.

---

## 2026-07-21 — Planning phase complete; plan approved and hardened. Pre-implementation.

**State:** No implementation code written yet. The design is locked in
`plans/implementation_plan.md` and is the spec milestone 1 builds against.

**What was done.** Read CLAUDE.md/AGENTS.md + ROADMAP.md in full, the paper
(`paper/`), and the MATLAB reference (`matlab/`). Inspected a real 10 GHz file
byte-for-byte (not assumed from the paper) and parsed the weight workbook. Produced the
implementation plan, then hardened it across **7 rounds of independent (Codex) review**
— every comment resolved; reviewer's final verdict was "no further comments,
implementation-ready."

**Verified data facts (not assumed).**
- 10 GHz: `data/10ghz/subject_<1..16>_<8am|10am|12pm|2pm|4pm>.mat`, 80 files, MAT v5,
  little-endian, zlib. Var `framesRadar` = MATLAB **double** `[534 fast-time × 20 chirps
  × 100 frames]`, complex; on disk the elements are `miINT16` (space optimization) so
  `scipy.io.loadmat` returns **complex128**. Also `framesRadarIQ` [20834×2×100] (raw IQ,
  unused). One file = one subject/session = 100 frames.
- 77 GHz: MAT **v7.3/HDF5**, ~285 MB each (~23 GB), needs `h5py` (not yet installed —
  milestone-2 audit installs it). h5py-reported shape (reviewer-sampled)
  `(16,256,256,125)=(Nrx,Nchirps,Nfast,Nframes)`; full axis reversal →
  `(Nframes,Nfast,Nchirps,Nrx)`. Fast-time↔chirp (both 256) disambiguated by a raw-data
  signal-domain check, not shape alone.
- Ground truth: `data/weight/metadata_subjects_info.xlsx`, sheet `MetaData`, rows 3–18.
  Two-row merged header → parse by fixed cell addresses. Cols E–I = weights 8am→4pm.
  Signed target `Δm% = (m(s) − m(S0))/m(S0)×100` (negative = loss), ≈0 to ≈−2%.
- Subject identity: radar `subject_N` = workbook "Subject N" (owner-confirmed; old
  MATLAB 5–20 numbering was renumbered to 1–16 for cleanliness, same subjects/order).

**Key locked decisions & why (see plan for full detail).**
- MATLAB is a **design reference only** — Python is the sole source of all reported
  numbers; correctness via Python-native self-consistency checks, not numeric diffs.
- Headline = **fluid-loss (Δm%) regression under LOSO**; 5-class demoted to secondary
  **ordinal**. Analysis unit is **session-level** (aggregate per-frame WST features to
  one vector/session) to kill pseudo-replication; per-frame is diagnostic only.
- Deliberate departures from the reference (logged here as they're implemented):
  robust standardize = median/MAD (not the reference's mean/MAD mix); range gate is a
  config parameter (default 1–2 m); WST log transform = order-aware
  (`log(S+ε)` on orders 1–2, ε=1e-6; order 0 left linear); EdgeTrim=32 **after**
  reduction.
- Scoring counts use **N_eval** (evaluable subjects), never a hard 16; session
  eligibility = `≥ ceil(0.5 × actual_frame_count)` QC-passing frames, no imputation.
- 77 GHz primary feature = **slow-time (Doppler) I/Q WST, per-Rx, feature-space fused**
  (magnitude discards Doppler phase; coherent Rx averaging risks phase cancellation).
- Stats: subject-cluster bootstrap (B=10000), seeds collapsed (metric-type-aware),
  all CIs/p-values labeled **conditional/exploratory**; effect sizes + per-subject
  spread carry interpretation.

**Outcome:** success (planning). **Next:** milestone 1 — repo scaffold, config system,
manifest + nested-LOSO splitter + provenance, and `tests/test_no_leakage.py` green
before any modeling.
