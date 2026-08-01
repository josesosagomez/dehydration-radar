"""T-M9-sbatch: static checks on milestone 9's IBEX shell artifacts, re-applying the
`test_exp_b_ibex_scripts.py` patterns (C23/C24/C25) to the Exp D CNN fold array.

What pytest can verify without a real SLURM instance: the STAGE-dispatch file's #SBATCH
header carries no per-stage resource directive (C24); `sbatch --parsable`'s possible
";cluster" suffix is stripped before a job id is used in a path or a --dependency (C25); the
array is never submitted after a failed init (C23); the run_dir handoff is robust to
preflight noise; and the wrapper is git-free from day one (the M8 step-10.5 lesson). The
scheduler-dependent behaviour itself (allocation, --array/--gres semantics, dependency
resolution) is a code-review + real-dry-run check, as the plan itself acknowledges.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
IBEX = REPO_ROOT / "scripts" / "ibex"
CNN_SBATCH = IBEX / "run_exp_d_cnn.sbatch"
SUBMIT_SH = IBEX / "submit_exp_d_cnn.sh"
CHEAP_SBATCH = IBEX / "run_exp_d_cheap.sbatch"
EXP_C_SBATCH = IBEX / "run_exp_c.sbatch"

RESOURCE_DIRECTIVES = ("--cpus-per-task", "--mem", "--time", "--array", "--gres",
                       "--output", "--error")


def _sbatch_header_lines(path: Path) -> list[str]:
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("#SBATCH"):
            if line.strip() and not line.startswith("#!"):
                break   # header ends at the first non-#SBATCH, non-shebang, non-blank line
            continue
        lines.append(line)
    return lines


def test_cnn_sbatch_header_carries_no_resource_directives():
    """(C24) The three stages differ by more than a factor of ten and by whether a GPU is
    requested at all; #SBATCH is parsed before STAGE exists, so one header cannot serve them
    and must carry none."""
    header = _sbatch_header_lines(CNN_SBATCH)
    assert any(line.startswith("#SBATCH --job-name") for line in header)
    for line in header:
        for directive in RESOURCE_DIRECTIVES:
            assert directive not in line, f"unexpected resource directive in shared header: {line!r}"


@pytest.mark.parametrize("path", [EXP_C_SBATCH, CHEAP_SBATCH])
def test_single_stage_sbatch_files_do_carry_their_own_fixed_resources(path):
    """Sanity contrast: these are single, non-dispatching jobs, so (unlike the CNN file) they
    legitimately carry their own fixed #SBATCH resources."""
    joined = "\n".join(_sbatch_header_lines(path))
    assert "--cpus-per-task" in joined and "--mem" in joined and "--time" in joined


def test_cnn_fold_stage_maps_the_array_task_id_to_a_zero_based_fold():
    """Fold ids are POSITIONS in the selectable-fold list (0-based) while a SLURM array is
    1-based; the offset must exist exactly once, in the sbatch."""
    text = CNN_SBATCH.read_text(encoding="utf-8")
    assert "FOLD=$((SLURM_ARRAY_TASK_ID - 1))" in text
    assert '--fold "$FOLD"' in text


def test_cnn_sbatch_carries_a_cpu_smoke_stage_that_never_touches_a_run_group():
    """Step 10 requires a CPU mechanism smoke for every Exp D family in BOTH bands, but the
    77 GHz store lives only on IBEX — so the smoke needs a batch path, and the STAGE-dispatch
    file is where it belongs. It runs `--subset 6subjects` in one process, so it must NOT
    accept or write a --run-dir: a smoke that landed inside a real run group would put
    mechanism-only artifacts where the merge looks for shards."""
    text = CNN_SBATCH.read_text(encoding="utf-8")
    smoke = text.split("smoke)")[1].split(";;")[0]
    assert "--subset" in smoke and "6subjects" in smoke
    assert "--run-dir" not in smoke and "RUN_DIR" not in smoke
    assert "configs/smoke.yaml" in smoke      # the seed_set=[1] overlay, or it costs 5x


def test_the_gpu_overlay_is_applied_to_every_run_group_stage_and_to_no_other():
    """(§5 trap 11 + the config_hash coupling) `ibex.yaml` is paths-only and nothing else sets
    `device: cuda`, so the fold array would allocate a GPU and train on CPU. The overlay fixes
    that — but because `run.device` is hashed into `config_fingerprint`, init/fold/merge must
    ALL load it or the array fails `_validate_group_lineage` on config_hash. The smoke is
    standalone (it joins no run group) and step 10 specifies it on CPU, so it must NOT."""
    text = CNN_SBATCH.read_text(encoding="utf-8")
    body = text.split("case \"$STAGE\" in")[1]
    stages = {name: body.split(f"{name})")[1].split(";;")[0]
              for name in ("init", "fold", "merge", "smoke")}
    group_args = {name: ("GROUP_ARGS" in stage) for name, stage in stages.items()}
    assert group_args == {"init": True, "fold": True, "merge": True, "smoke": False}
    assert "GROUP_ARGS=(\"${CONFIG_ARGS[@]}\" --config configs/gpu.yaml)" in text
    assert "configs/gpu.yaml" not in stages["smoke"]


def test_gpu_overlay_names_only_the_device():
    import yaml

    from dehyd.config import M6_SECTIONS

    overlay = yaml.safe_load((REPO_ROOT / "configs" / "gpu.yaml").read_text(encoding="utf-8"))
    assert set(overlay) == {"run"} and overlay["run"] == {"device": "cuda"}
    assert not set(overlay) & set(M6_SECTIONS)


def test_the_smoke_overlay_config_reduces_run_level_fields_only():
    """CLAUDE.md's smoke rule: a smoke differs from the full job by run-level config alone.
    Any M6-frozen section appearing here would make the smoke exercise different science."""
    import yaml

    from dehyd.config import M6_SECTIONS

    overlay = yaml.safe_load((REPO_ROOT / "configs" / "smoke.yaml").read_text(encoding="utf-8"))
    assert set(overlay) == {"run"}
    assert overlay["run"] == {"seed_set": [1]}
    assert not set(overlay) & set(M6_SECTIONS)


def test_submit_script_requests_a_gpu_for_the_fold_stage_only():
    text = SUBMIT_SH.read_text(encoding="utf-8")
    gpu_lines = [line for line in text.splitlines() if "--gres=gpu" in line]
    assert len(gpu_lines) == 1 and "--array=" in gpu_lines[0]


def test_the_array_span_defaults_to_all_sixteen_folds():
    """Step 10 needs a SINGLE-fold GPU task to measure ARRAY_TIME from (the C8 rule: size the
    array from measurement, never a guess), so the span is an env knob — but its DEFAULT must
    stay the full 16, or a forgotten ARRAY_SPEC would silently under-run step 13's cohort."""
    text = SUBMIT_SH.read_text(encoding="utf-8")
    assert '--array="${ARRAY_SPEC:-1-16}"' in text


def test_submit_script_normalizes_every_parsable_job_id():
    """(C25) `sbatch --parsable` can return "jobid;cluster" on a multi-cluster SLURM; every
    capture must strip from the first ';' before the value is used in a path or a
    --dependency spec."""
    text = SUBMIT_SH.read_text(encoding="utf-8")
    for name in ("init", "array", "merge"):
        assert re.search(rf'{name}_job_id="\$\{{{name}_raw%%;\*\}}"', text), name
    assert 'logs/exp_d_cnn_init_${tag}_${init_job_id}.out' in text
    assert '--dependency=afterany:"$array_job_id"' in text


def test_submit_script_blocks_on_init_so_a_failure_never_reaches_the_array():
    """(C23) `--wait` plus `set -e`: the array submission is textually after the init capture,
    and a nonzero init exit terminates the script before it."""
    text = SUBMIT_SH.read_text(encoding="utf-8")
    assert "set -euo pipefail" in text
    assert "sbatch --wait --parsable" in text
    assert text.index("init_raw=") < text.index("array_raw=") < text.index("merge_raw=")
    assert '[ -d "$run_dir" ]' in text


def test_submit_script_is_git_free_and_requires_a_revision_file():
    """The M8 step-10.5 lesson, baked in from day one: on IBEX the tree is COPIED, so `git`
    is absent and every call to it would fail the job."""
    text = SUBMIT_SH.read_text(encoding="utf-8")
    # comments and single-quoted literals cannot invoke anything; the wrapper's REVISION help
    # text legitimately QUOTES the `git rev-parse` command the user runs elsewhere.
    code = "\n".join(
        re.sub(r"'[^']*'", "", line)
        for line in text.splitlines()
        if not line.lstrip().startswith("#")
    )
    assert not re.search(r"(^|[^-\w])git\s", code), "the submit wrapper must never invoke git"
    assert "[ -f REVISION ]" in text
    assert "unset DEHYD_GIT_COMMIT DEHYD_GIT_BRANCH DEHYD_GIT_DIRTY" in text


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available on PATH")
def test_run_dir_handoff_is_the_last_stdout_line_and_survives_preflight_noise(tmp_path):
    """`--init-run-group` prints the run dir LAST; the wrapper reads it with `tail -n1`, so
    any amount of config/validation chatter before it is harmless. Exercised for real."""
    assert 'run_dir=$(tail -n1 ' in SUBMIT_SH.read_text(encoding="utf-8")
    script = (
        'printf "config : a, b  band 10ghz\\nslurm  : job 7\\n/results/runs/20260731_abc\\n" '
        "> out.txt; run_dir=$(tail -n1 out.txt); echo \"$run_dir\""
    )
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True,
                         cwd=str(tmp_path))
    assert out.stdout.strip() == "/results/runs/20260731_abc"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available on PATH")
def test_percent_percent_semicolon_star_strips_cluster_suffix_in_bash():
    """(C25) The exact idiom, run against bash on a "12345;ibex" fixture — not just grepped."""
    out = subprocess.run(["bash", "-c", 'raw="12345;ibex"; echo "${raw%%;*}"'],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "12345"


def test_every_new_ibex_script_parses_with_bash_dash_n():
    """A cheap syntax gate — it catches the quoting class of bug this repo's own history had
    (an apostrophe inside a ${VAR:?message} broke bash's parser even within double quotes)."""
    if shutil.which("bash") is None:
        pytest.skip("bash not available on PATH")
    for path in (EXP_C_SBATCH, CHEAP_SBATCH, CNN_SBATCH, SUBMIT_SH):
        result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        assert result.returncode == 0, f"{path.name}: {result.stderr}"
