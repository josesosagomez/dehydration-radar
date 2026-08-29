# Running the 77 GHz front-end on IBEX (milestone 5)

Self-contained runbook — every command is literal (Claude has no ssh; the owner runs these).
IBEX is CPU-only here: numpy is the canonical backend for every reported feature (frozen
policy), so no GPU is requested. `configs/ibex.yaml` is a **paths-only overlay**; because it
changes only `paths.*`, an axis certificate written on one machine stays valid after rsync.

## 0. One-time setup on IBEX — **on the LOGIN node**

Everything in this section must run on the **login** node, not in a batch job: it needs
outbound network, which compute nodes generally do not have.

```bash
# uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"   # this session; the installer only
                                                        # edits ~/.bashrc, which batch jobs
                                                        # never source (see the note below)

# clone + the pinned environment (uv.lock pins scipy <1.17; the torch CPU wheel rides along)
git clone https://github.com/josesosagomez/dehydration-radar.git dehy_radar_new && cd dehy_radar_new
git checkout v1_milestone_5
uv sync --frozen          # creates .venv/ — the batch jobs use it directly

# kymatio import smoke (numpy frontend)
.venv/bin/python -c "from kymatio.numpy import Scattering1D; print('kymatio ok')"
```

`configs/ibex.yaml` now carries the owner's literal root (`/ibex/user/floresge/dehy_radar_new`), so
there is nothing to edit on IBEX. Keep it that way: since M9 the IBEX tree is a git **clone**, and
a hand-edit there would leave the tree dirty, which `assert_clean_tree` and `submit_ibex.sh` both
refuse. If the root ever moves, change it here, commit, and pull — do not edit it on IBEX.

> **Why the sbatch scripts do not just call `uv`.** A Slurm batch script runs in a
> **non-interactive** shell, which never sources `~/.bashrc` — so the PATH entry the uv
> installer appends there is absent and `uv run ...` fails with `uv: command not found`.
> The scripts therefore (a) add `~/.local/bin` and `~/.cargo/bin` to PATH explicitly, and
> (b) prefer `.venv/bin/python` directly, falling back to `uv run --no-sync`. Running
> `uv sync --frozen` on the login node first is what makes (b) work, and it also guarantees
> no job ever tries to resolve or download a package from a compute node.

## 1. Stage the data (once)

```bash
rsync -av --progress data/77ghz/  floresge@ilogin.ibex.kaust.edu.sa:/ibex/user/floresge/dehy_radar_new/data/77ghz/   # ~22 GB
rsync -av --progress data/10ghz/  floresge@ilogin.ibex.kaust.edu.sa:/ibex/user/floresge/dehy_radar_new/data/10ghz/
rsync -av --progress data/weight/ floresge@ilogin.ibex.kaust.edu.sa:/ibex/user/floresge/dehy_radar_new/data/weight/
```

## 2. Cohort QC + axis certification (the authoritative eligibility gate)

```bash
sbatch scripts/ibex/qc77.sbatch     # single job; writes results/qc/qc_survival_77ghz.csv
```

This is the authoritative survival/eligibility pass (owner outcome (b), the mechanism-corrected
exclude-range-bin-0 flatline rule). It fails closed on any non-ACCEPTED axis verdict. Inspect
`results/qc/qc_survival_77ghz.csv` before proceeding — surprising survival is a finding for
HISTORY.md, never a reason to move a threshold.

## 3. WST feature extraction (job array over the 80 cells) — AFTER the gate

```bash
sbatch scripts/ibex/wst77.sbatch    # --array=0-79; one shard per eligible cell
```

Each task loads its own file, re-certifies the axis, and (if the cell is eligible) writes
`results/wst/shards/wst77_s<subj>_<sess>.csv` + a fingerprint sidecar. Optionally the light
chain diagnostics: `sbatch scripts/ibex/preprocess77.sbatch`.

## 4. Bring results back and merge

```bash
rsync -av floresge@ilogin.ibex.kaust.edu.sa:/ibex/user/floresge/dehy_radar_new/results/ results/
uv run python experiments/run_wst77.py --config configs/exp_77ghz.yaml --merge-shards
```

`--merge-shards` verifies exactly the eligible shards are present, rejects any duplicate or any
fingerprint disagreement (a stale retry, a different code/config revision, or a changed raw
file), and only then writes the curated `results/wst/wst_diagnostics_77ghz.csv`.

## Optional: run the realdata test suite on IBEX

```bash
DEHYD_REALDATA=1 uv run pytest -q        # exercises the loader/QC/extraction on real files
```

## Exploratory path-40 LOSO versus random-session comparison

This light CPU job reads the already completed compact WST-order tables from the separate
diagnostic repository. It runs one subject-level LOSO score and one deliberately leaky
random-session score. Both are post-selection exploratory and neither is confirmatory.

```bash
cd /path/to/dehydration_radar_2
uv sync --frozen
sbatch scripts/ibex/run_path40_exploratory.sbatch
```

The diagnostic root defaults to
`/ibex/user/sosagojm/dehydration_loso_diagnostic`. Override it only if that repository moved:

```bash
DIAGNOSTIC_ROOT=/new/diagnostic/root \
  sbatch --export=ALL,DIAGNOSTIC_ROOT scripts/ibex/run_path40_exploratory.sbatch
```

Results go only to `results/exploratory_path40/`; stdout and stderr go to
`logs/path40_exploratory_<jobid>.out` and `.err`.

## Quality-aware training sensitivity (10 GHz)

This CPU job compares baseline training, training-only removal of the five negative
in-band-margin sessions, and appending that raw margin as one train-scaled feature.  LOSO
is primary; the separately written subject-overlap session split is an optimistic post-hoc
diagnostic only.

The feature store must first be rebuilt at the implementation commit.  The store validator
rejects an older build commit rather than weakening provenance.

```bash
IBEX_CONFIG=configs/ibex_sosagojm.yaml sbatch scripts/ibex/extract10.sbatch
# Wait for all 80 extraction array tasks and validate the rebuilt store, then:
IBEX_CONFIG=configs/ibex_sosagojm.yaml \
  sbatch scripts/ibex/run_quality_training_sensitivity.sbatch
```

Outputs are isolated below `results/quality_training_sensitivity_10ghz/loso/` and
`results/quality_training_sensitivity_10ghz/subject_overlap_session_cv/`.

## Notes / gotchas (pre-paid, see plans/MILESTONE_5_PLAN.md §5)

- `HDF5_USE_FILE_LOCKING=FALSE` is set in the sbatch env (GPFS can spuriously fail read-opens).
- The `.gitattributes` `scripts/ibex/* eol=lf` rule keeps these scripts LF on checkout, so a
  Windows CRLF never reaches the Linux shebang (`/bin/bash^M: bad interpreter`).
- `uv sync --frozen` must be used so the lockfile prevents scipy resolving to >= 1.17.
