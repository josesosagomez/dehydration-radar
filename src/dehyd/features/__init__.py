"""WST feature extraction (milestone 4).

Public API re-exported so the M6 harness imports from `dehyd.features`, not from a CLI
script (the `preprocess/__init__.py` precedent).
"""

from .extraction import (
    CanonicalSpecError,
    SessionVariantResult,
    canonical_spec_guard,
    extract_session_features,
    extract_session_variants,
)
from .pooling import (
    aggregate_session,
    feature_layout,
    flat_layout,
    flatten_series,
    pool_stats,
    session_feature_layout,
)
from .wst import (
    AgreementResult,
    apply_order_log,
    backend_agreement,
    build_scattering,
    octaves_j,
    preprocessed_length,
    scatter_channels,
    scatter_frames,
    scattering_shape,
    t_samples,
    wst_spec,
)

__all__ = [
    # wst
    "AgreementResult",
    "apply_order_log",
    "backend_agreement",
    "build_scattering",
    "octaves_j",
    "preprocessed_length",
    "scatter_channels",
    "scatter_frames",
    "scattering_shape",
    "t_samples",
    "wst_spec",
    # pooling
    "aggregate_session",
    "feature_layout",
    "flat_layout",
    "flatten_series",
    "pool_stats",
    "session_feature_layout",
    # extraction
    "CanonicalSpecError",
    "SessionVariantResult",
    "canonical_spec_guard",
    "extract_session_features",
    "extract_session_variants",
]
