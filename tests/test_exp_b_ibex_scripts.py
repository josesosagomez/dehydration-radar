"""T-M8-variant (C24/C25): static checks on the IBEX shell artifacts that pytest CAN verify
without a real SLURM instance -- the #SBATCH header carries no per-stage resource directives
(C24), and the job-ID normalization strips a possible ";cluster" suffix from `sbatch
--parsable` before it is used in a path or --dependency (C25). The scheduler-dependent
behaviour itself (actual allocation, log naming, --array/--dependency semantics) is a code-
review + real-dry-run check, per the plan's own acknowledgement -- not something pytest can
execute (plan §2.5 Acceptance).
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VARIANT_SBATCH = REPO_ROOT / "scripts" / "ibex" / "run_exp_b_variant.sbatch"
SUBMIT_SH = REPO_ROOT / "scripts" / "ibex" / "submit_exp_b_variant.sh"
PRIMARY_SBATCH = REPO_ROOT / "scripts" / "ibex" / "run_exp_b.sbatch"

RESOURCE_DIRECTIVES = ("--cpus-per-task", "--mem", "--time", "--array", "--output", "--error")


def _sbatch_header_lines(path: Path) -> list[str]:
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("#SBATCH"):
            if line.strip() and not line.startswith("#!"):
                break   # header ends at the first non-#SBATCH, non-shebang, non-blank line
            continue
        lines.append(line)
    return lines


def test_variant_sbatch_header_carries_no_resource_directives():
    """(C24) The shared STAGE-dispatch file's #SBATCH header must carry ONLY --job-name --
    #SBATCH lines are parsed before STAGE is ever evaluated, so a single header cannot vary
    per stage; every resource flag must instead come from the submit script's per-stage CLI."""
    header = _sbatch_header_lines(VARIANT_SBATCH)
    assert any(line.startswith("#SBATCH --job-name") for line in header)
    for line in header:
        for directive in RESOURCE_DIRECTIVES:
            assert directive not in line, f"unexpected resource directive in shared header: {line!r}"


def test_primary_sbatch_header_does_carry_its_own_fixed_resources():
    """Sanity contrast: the PRIMARY path's sbatch is a single, non-dispatching job, so (unlike
    the variant) it legitimately DOES carry its own fixed #SBATCH resource directives."""
    header = _sbatch_header_lines(PRIMARY_SBATCH)
    joined = "\n".join(header)
    assert "--cpus-per-task" in joined and "--mem" in joined and "--time" in joined


def test_submit_script_normalizes_both_init_and_array_job_ids():
    """(C25) `sbatch --parsable` can return "jobid;cluster" on a multi-cluster SLURM config;
    both job-ID captures must strip everything from the first ';' before use in a path or a
    --dependency spec."""
    text = SUBMIT_SH.read_text(encoding="utf-8")
    assert re.search(r'init_job_id="\$\{init_raw%%;\*\}"', text)
    assert re.search(r'array_job_id="\$\{array_raw%%;\*\}"', text)
    # the normalized init_job_id must be what locates the %j-named log file, not the raw value.
    assert 'exp_b_variant_init_${init_job_id}.out' in text
    # the normalized array_job_id must be what feeds --dependency, not the raw value.
    assert '--dependency=afterany:"$array_job_id"' in text


def run_bash(script: str, **kwargs) -> subprocess.CompletedProcess:
    """Run a snippet through bash with the script on STDIN, never as an argument.

    Why not `bash -c <script>`: on a Windows checkout `shutil.which("bash")` resolves to the
    WindowsApps **WSL app-execution-alias stub**, not Git Bash, and invoking bare `"bash"` with
    the script as an argument silently swallows the output — rc=0 with stdout `"\\n"` — so the
    assertion below compared `''` against `'12345'` and the test had been red since M8. (The
    long-standing diagnosis in HISTORY/HANDOFF, "Git Bash eats backslashes / inline `bash -c`
    comes back empty", named the symptom rather than the cause; corrected 2026-08-08.)

    Feeding the script on stdin returns correctly under the stub, under Git Bash, and on Linux,
    so this tests bash rather than testing which bash the PATH happened to find.
    """
    return subprocess.run(["bash", "-s"], input=script.encode(), capture_output=True, **kwargs)


def bash_syntax_check(path) -> subprocess.CompletedProcess:
    """`bash -n` on a shell artifact, with its BYTES piped in rather than its path.

    Same root cause: a `C:\\...` path handed to the resolved interpreter comes back as rc=127
    with the backslashes eaten. LF is forced because this repo has `core.autocrlf=true`, so the
    working-tree copy can carry CRLF even though `.gitattributes` checks these files out LF on
    Linux — and a CRLF after a line-continuation backslash breaks bash's parser.
    """
    script = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return subprocess.run(["bash", "-n", "-"], input=script, capture_output=True)


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available on PATH")
def test_percent_percent_semicolon_star_strips_cluster_suffix_in_bash():
    """(C25) The exact normalization idiom, exercised for real against bash -- not just
    grepped for -- on a fixture supplying "12345;ibex" as sbatch --parsable's raw value."""
    out = run_bash('raw="12345;ibex"; stripped="${raw%%;*}"; echo "$stripped"')
    assert out.returncode == 0, out.stderr
    assert out.stdout.decode().strip() == "12345"

    # a bare numeric ID (single-cluster SLURM) must pass through unchanged.
    out_bare = run_bash('raw="12345"; stripped="${raw%%;*}"; echo "$stripped"')
    assert out_bare.returncode == 0, out_bare.stderr
    assert out_bare.stdout.decode().strip() == "12345"


def test_all_three_ibex_scripts_parse_with_bash_dash_n():
    """A cheap syntax gate for the shell artifacts this milestone adds -- not a semantic
    check, but catches the class of quoting bug this file's own history had (an apostrophe
    inside a ${VAR:?message} broke bash's parser even within double quotes)."""
    if shutil.which("bash") is None:
        pytest.skip("bash not available on PATH")
    for path in (PRIMARY_SBATCH, VARIANT_SBATCH, SUBMIT_SH):
        result = bash_syntax_check(path)
        assert result.returncode == 0, f"{path.name}: {result.stderr.decode('utf-8', 'replace')}"
