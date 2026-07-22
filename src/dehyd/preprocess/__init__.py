"""Preprocessing: the executable sequence between a QC-passed frame and WST input.

The stages live in the order the plan states them (implementation_plan.md,
"Preprocessing — executable sequence"):

    filters.py      step 3  band gate (zero-phase Butterworth, or the FFT ablation)
    reduce.py       steps 4-5  chirp reduction (Option A / B), then EdgeTrim
    standardize.py  steps 6-7  channel mapping (mag / iq) and per-signal robust z
    pipeline.py     the whole sequence as one linear function

**Nothing here is fitted.** Every stage is a deterministic function of one frame plus
frozen config constants; the only data-dependent numbers -- Option B's peak bin and
the robust z's median/MAD -- come from the signal being processed itself. Preprocessing
therefore introduces no train/test leakage vector and never enters the CV loops.
"""
