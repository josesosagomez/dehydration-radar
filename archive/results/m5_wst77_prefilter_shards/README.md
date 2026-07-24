# Superseded: 77 GHz WST shards + curated CSV built BEFORE the eligible-frame filter

IBEX array job 49399759 (+ task-22 rerun 49399874), 2026-07-24.

INVALID for analysis. `run_wst77.py` filtered to eligible SESSIONS but extracted all 125
frames of each, instead of the QC-PASSING frames of eligible sessions (`eligible_frames`,
the only view modelling may consume). 34 QC-failing frames (0.38%) entered the features,
in exactly two sessions: subject 7 12pm (used 125, should be 102) and subject 9 12pm
(used 125, should be 114). All other 70 eligible sessions were unaffected (125/125 passing),
so their shards would be numerically identical under the fix — but the whole set is retired
together rather than partially reused, so the artifact has one provenance.

Kept because the geometry and fingerprint consistency they demonstrate are still evidence:
n_paths 424/453/182, n_time 8/4/4 cohort-wide, 80/80 fingerprints identical.

Superseded by the re-run after the fix (fingerprint gains
`frame_selection=qc_pass_frames_of_eligible_sessions`).
