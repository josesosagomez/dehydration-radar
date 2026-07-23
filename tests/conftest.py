"""Shared test fixtures and the `realdata` gate.

Registering the `realdata` marker in pyproject only silences pytest's unknown-marker
warning — it does not skip anything. The gating is here:

  uv run pytest              -> realdata tests are SKIPPED (clean checkout is green)
  uv run pytest --realdata   -> they RUN, and missing/incomplete data is a HARD FAILURE

The asymmetry is deliberate: the whole point of the --realdata command is to prove the
real cohort validates, so silently skipping when the data are absent would defeat it.
"""

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_10GHZ_DIR = REPO_ROOT / "data" / "10ghz"
DATA_77GHZ_DIR = REPO_ROOT / "data" / "77ghz"
WEIGHT_XLSX = REPO_ROOT / "data" / "weight" / "metadata_subjects_info.xlsx"

N_SUBJECTS = 16
N_SESSIONS = 5
N_RADAR_FILES = N_SUBJECTS * N_SESSIONS


def pytest_addoption(parser):
    parser.addoption(
        "--realdata",
        action="store_true",
        default=False,
        help="run tests that require the private raw dataset",
    )


def _realdata_enabled(config) -> bool:
    # The env var lets IBEX batch jobs enable this without argv plumbing.
    return config.getoption("--realdata") or os.environ.get("DEHYD_REALDATA") == "1"


def pytest_collection_modifyitems(config, items):
    if _realdata_enabled(config):
        return
    skip = pytest.mark.skip(reason="needs the private dataset; pass --realdata to run")
    for item in items:
        if "realdata" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def real_data_paths():
    """Resolved paths to the real dataset.

    Fails (never skips) if the data are absent or incomplete — this fixture only ever
    runs when --realdata was requested.
    """
    missing = [p for p in (DATA_10GHZ_DIR, WEIGHT_XLSX) if not p.exists()]
    if missing:
        pytest.fail(
            "--realdata was requested but these paths are missing: "
            + ", ".join(str(p) for p in missing)
        )

    mat_files = sorted(DATA_10GHZ_DIR.glob("*.mat"))
    if len(mat_files) != N_RADAR_FILES:
        pytest.fail(
            f"expected {N_RADAR_FILES} radar files in {DATA_10GHZ_DIR}, "
            f"found {len(mat_files)}"
        )

    return {
        "data_10ghz_dir": DATA_10GHZ_DIR,
        "weight_xlsx": WEIGHT_XLSX,
        "mat_files": mat_files,
    }


@pytest.fixture(scope="session")
def real_data_77_paths():
    """Resolved paths to the real 77 GHz cohort (band 2, milestone 5).

    Separate from real_data_paths so the 10 GHz realdata tests never depend on the
    22 GB 77 GHz tree being present. Fails (never skips) if the 77 GHz data are absent
    or incomplete — it only runs under --realdata, whose contract is that the full
    cohort is present.
    """
    if not DATA_77GHZ_DIR.exists():
        pytest.fail(
            "--realdata was requested but the 77 GHz data dir is missing: "
            f"{DATA_77GHZ_DIR}"
        )

    mat_files = sorted(DATA_77GHZ_DIR.glob("*.mat"))
    if len(mat_files) != N_RADAR_FILES:
        pytest.fail(
            f"expected {N_RADAR_FILES} 77 GHz files in {DATA_77GHZ_DIR}, "
            f"found {len(mat_files)}"
        )

    return {
        "data_77ghz_dir": DATA_77GHZ_DIR,
        "mat_files": mat_files,
    }
