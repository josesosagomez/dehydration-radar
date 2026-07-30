"""Fold-parallel execution with a progress heartbeat -- the single implementation.

Extracted verbatim from `exp_b._run_folds_parallel` (M8) at milestone 9 step 5, because Exp C
needs the same thing and a per-experiment copy of a process pool is exactly the kind of
duplication that drifts silently (one copy gains a fix, another does not). The outer folds are
independent by construction, so running them in worker processes is a pure speedup: the
results are bit-identical to the serial run, which is asserted by Exp B's and Exp C's own
serial-vs-parallel tests rather than assumed here. Exp D deliberately uses one outer fold per
SLURM array task instead of this in-process pool.

Deliberately generic and deliberately thin: this module knows about tasks and workers, NOT
about folds, providers, scores or configs. The caller builds the tasks (fold construction stays
in `splits.py`-fed experiment code) and the caller sorts the results into canonical order --
results come back in COMPLETION order here. The one property assumed of a result object is a
`.test_subject` attribute, used only for the completion log line; every fold result in this
project has one.
"""

from __future__ import annotations

import time

_POLL_INTERVAL_S = 1     # how often we check for newly-finished tasks (responsive, not a log cadence)
_PROGRESS_INTERVAL_S = 60  # minimum spacing between heartbeat lines when nothing has finished yet


def run_folds_parallel(worker, tasks, n_workers, label):
    """Run `worker(*task)` for each task -- serially, or via a spawn-context Pool -- and return
    the results in completion order.

    `worker` must be a top-level function and every task a picklable argument tuple, since the
    pool branch ships both to a fresh interpreter. `label` prefixes the progress lines
    (`[<label> progress] ...`) so a merged IBEX log says which experiment is talking.

    Prints a progress line to stdout whenever a fold completes, and at least every
    _PROGRESS_INTERVAL_S seconds even if none have, so an IBEX job's log shows the process is
    alive rather than going silent for the whole search. With n_workers roughly equal to the
    fold count, all folds are dispatched together and tend to FINISH together too -- so a
    print-on-completion line alone would show nothing until the very end; the timer is what
    actually distinguishes "still running" from "stuck". Polling itself (_POLL_INTERVAL_S) is
    kept short so this adds negligible latency on fast (e.g. test) workloads -- only the print
    cadence is throttled to _PROGRESS_INTERVAL_S."""
    tasks = list(tasks)
    n_total = len(tasks)
    start = time.monotonic()

    if n_workers <= 1 or n_total <= 1:
        results = []
        for task in tasks:
            results.append(worker(*task))
            print(f"[{label} progress] {len(results)}/{n_total} folds done "
                  f"(test_subject={results[-1].test_subject}), elapsed={time.monotonic() - start:.0f}s",
                  flush=True)
    else:
        import multiprocessing as mp

        # spawn (not fork): a clean worker with no inherited open npz handles or BLAS state, so
        # behaviour is identical on Linux (IBEX) and elsewhere.
        ctx = mp.get_context("spawn")
        n_procs = min(n_workers, n_total)
        with ctx.Pool(processes=n_procs) as pool:
            pending = [pool.apply_async(worker, t) for t in tasks]
            print(f"[{label} progress] {n_total} folds dispatched across {n_procs} workers", flush=True)
            results = []
            last_print = start
            while pending:
                still_pending, newly_done = [], []
                for r in pending:
                    (newly_done if r.ready() else still_pending).append(r)
                pending = still_pending
                results.extend(r.get() for r in newly_done)   # re-raises a worker's exception, if any (C12)
                now = time.monotonic()
                if newly_done or now - last_print >= _PROGRESS_INTERVAL_S:
                    print(f"[{label} progress] {len(results)}/{n_total} folds done, "
                          f"elapsed={now - start:.0f}s", flush=True)
                    last_print = now
                if pending:
                    time.sleep(_POLL_INTERVAL_S)

    return results
